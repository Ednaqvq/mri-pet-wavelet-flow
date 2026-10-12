"""Matched single-source generation controls without a Coarse residual anchor."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .flow_engine import FlowTrainer
from .models import ResBlock
from .data import ClinicalTransform
from .losses import reconstruction, masked_l1, ssim3d


class DirectUNet(nn.Module):
    """Same multiscale predictor for all arms; only input lift width differs."""
    def __init__(self, inputs, latent, width=64, checkpointing=True):
        super().__init__()
        self.checkpointing = checkpointing
        self.lift = nn.Conv3d(inputs, width, 1)
        self.enc = nn.ModuleList(nn.Sequential(ResBlock(c), ResBlock(c))
                                 for c in (width, 2*width, 4*width))
        self.down = nn.ModuleList(nn.Conv3d(a, 2*a, 3, 2, 1) for a in (width, 2*width))
        self.up = nn.ModuleList(nn.Conv3d(2*a, a, 3, padding=1) for a in (width, 2*width))
        self.merge = nn.ModuleList(nn.Conv3d(2*a, a, 1) for a in (width, 2*width))
        self.dec = nn.ModuleList(nn.Sequential(ResBlock(c), ResBlock(c)) for c in (width, 2*width))
        self.output = nn.Conv3d(width, latent, 1)
        # Avoid the undefined direction of an exactly zero vector in cosine loss.
        nn.init.normal_(self.output.weight, std=.001)
        nn.init.zeros_(self.output.bias)

    def block(self, module, x):
        if self.checkpointing and torch.is_grad_enabled():
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, source):
        h = self.lift(source)
        skips = []
        for i, block in enumerate(self.enc):
            h = self.block(block, h)
            skips.append(h)
            if i < 2:
                h = self.down[i](h)
        for i in (1, 0):
            h = self.up[i](F.interpolate(h, size=skips[i].shape[2:], mode='trilinear', align_corners=False))
            h = self.merge[i](torch.cat((h, skips[i]), 1))
            h = self.block(self.dec[i], h)
        return self.output(h).float()


class DirectTrainer(FlowTrainer):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.source_kind = cfg['direct']['source']
        if self.source_kind not in ('downsample', 'aae', 'coarse'):
            raise ValueError('Unknown direct source')
        self.stage_name = 'direct_'+self.source_kind
        self.run = Path(cfg['output_dir'])/f'fold_{self.fold}'/'direct'
        self.run.mkdir(parents=True, exist_ok=True)
        # Keep the inherited optimizer/checkpoint member name, not the Flow model.
        self.gen.velocity = DirectUNet(1 if self.source_kind == 'downsample' else cfg['model']['latent_channels'],
            cfg['model']['latent_channels'], cfg['direct']['width'],
            cfg['runtime'].get('activation_checkpointing', True)).to(self.device)
        self.normalization = None
        self.configure()

    def set_normalization(self, state):
        if state['fit_split'] != 'train' or state['fold'] != self.fold:
            raise ValueError('Normalization must use training fold')
        if self.metadata_hash is not None and state['metadata_hash'] != self.metadata_hash:
            raise ValueError('Normalization metadata mismatch')
        if state['coarse_sha256'] != self.provenance['sha256']:
            raise ValueError('Normalization backbone provenance mismatch')
        mu = torch.tensor(state['channels']['ZP']['mean'], device=self.device)
        std = torch.tensor(state['channels']['ZP']['std'], device=self.device)
        channels = self.cfg['model']['latent_channels']
        if mu.numel() != channels or std.numel() != channels or not torch.isfinite(mu).all() or not torch.isfinite(std).all() or (std < 0).any():
            raise ValueError('Invalid latent moments')
        self.mu = mu.float().view(1, channels, 1, 1, 1)
        self.std = std.float().clamp_min(self.cfg['direct']['std_floor']).view(1, channels, 1, 1, 1)
        self.normalization = state

    def initialize(self, path):
        state = super().initialize(path)
        self.set_normalization(json.loads(Path(self.cfg['direct']['statistics']).read_text()))
        return state

    def encode_coordinates(self, z):
        return (z.float()-self.mu)/self.std

    def decode_coordinates(self, z):
        return z.float()*self.std+self.mu

    @torch.no_grad()
    def source(self, mri):
        if self.source_kind == 'downsample':
            # No learned Coarse structural stem and no MRI-AAE computation.
            shape = tuple(n//4 for n in mri.shape[2:])
            return F.interpolate(mri.float(), size=shape, mode='trilinear', align_corners=False)
        za = self.gen.mri.encode(mri)
        if self.source_kind == 'aae':
            return za
        return self.gen.coarse(mri, za)[1]

    def loss(self, batch):
        with torch.no_grad():
            source = self.source(batch['input'])
            target = self.encode_coordinates(self.gen.pet.encode(batch['target']))
        predicted = self.gen.velocity(source)
        latent = F.l1_loss(predicted, target)
        cosine = 1-F.cosine_similarity(predicted.flatten(1), target.flatten(1), dim=1).mean()
        image = self.gen.pet.decode(self.decode_coordinates(predicted)).float()
        pet = reconstruction(image, batch['target'].float(), batch['brain_mask'], self.cfg['loss'])
        weights = self.cfg['direct']
        total = weights['image_weight']*pet + weights['latent_weight']*latent + weights['cosine_weight']*cosine
        # Shared history keys: endpoint=latent L1, pet=image loss, path=cosine loss.
        return total, dict(flow=0., endpoint=float(latent.detach()), pet=float(pet.detach()), path=float(cosine.detach()))

    @torch.no_grad()
    def generate(self, mri, clinical=None, ids=None, trajectory=False):
        if trajectory:
            raise ValueError('Direct models do not have ODE trajectories')
        self.gen.eval()
        with self.autocast():
            source = self.source(mri)
            zf = self.decode_coordinates(self.gen.velocity(source))
            out = self.gen.pet.decode(zf).float()
        if not torch.isfinite(out).all():
            raise FloatingPointError('Nonfinite direct output')
        return out, dict(source=source.float(), ZF=zf), {}, []

    @torch.no_grad()
    def validate(self, loader):
        self.configure(False)
        rows, started = [], time.monotonic()
        for index, raw in enumerate(loader):
            batch = self.move(raw)
            out, latents, _, _ = self.generate(batch['input'])
            with self.autocast():
                zp = self.gen.pet.encode(batch['target']).float()
            for i, sid in enumerate(batch['sample_id']):
                x, y, mask = out[i:i+1], batch['target'][i:i+1].float(), batch['brain_mask'][i:i+1]
                p, z = zp[i:i+1], latents['ZF'][i:i+1]
                mse = ((x-y).square()*mask).sum()/mask.sum().clamp_min(1)
                rows.append(dict(sample_id=sid, mae=float(masked_l1(x,y,mask)),
                    psnr=float(10*torch.log10(self.cfg['loss']['data_range']**2/mse.clamp_min(1e-12))),
                    ssim=float(ssim3d(x,y,self.cfg['loss']['data_range'],mask=mask)[0]),
                    latent_l1=float(F.l1_loss(z,p)), cosine=float(F.cosine_similarity(z.flatten(1),p.flatten(1)).mean())))
            if index == 0 or (index+1)%50 == 0 or index+1 == len(loader):
                print(json.dumps(dict(event='direct_validation', source=self.source_kind, batch=index+1,
                    batches=len(loader), seconds=round(time.monotonic()-started,1))), flush=True)
        if not rows:
            raise ValueError('Empty evaluation split')
        mean = {k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'sample_id'}
        if not all(np.isfinite(v) for v in mean.values()):
            raise FloatingPointError('Nonfinite direct validation')
        return dict(selection_loss=mean['mae'], samples=len(rows),
                    generation={k:mean[k] for k in ('mae','psnr','ssim')},
                    transport={k:mean[k] for k in ('latent_l1','cosine')}), rows

    def checkpoint(self, *args):
        state = super().checkpoint(*args)
        state.update(format_version='single-source-direct-v1', stage=self.stage_name,
                     source_kind=self.source_kind, normalization=self.normalization)
        return state

    def load(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state.get('format_version') != 'single-source-direct-v1' or state['source_kind'] != self.source_kind:
            raise ValueError('Direct checkpoint source/type mismatch')
        self.verify(state)
        if state['config']['direct'] != self.cfg['direct']:
            raise ValueError('Direct configuration mismatch')
        self.gen.load_state_dict(state['generator'], strict=True)
        self.clinical = ClinicalTransform(state['clinical'])
        self.provenance = state['provenance']
        self.set_normalization(state['normalization'])
        return state
