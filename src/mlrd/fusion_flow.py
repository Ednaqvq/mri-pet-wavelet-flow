"""Two MRI feature branches -> lightweight fusion -> Flow -> frozen PET Decoder.

Optional Coarse source supports frozen pretrained or joint training ablations.
"""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .models import Autoencoder, ResidualUNet, CoarseMapper, strided_mri_stem
from .flow import flow_pair, integrate, source_state
from .flow_engine import FlowTrainer
from .refinement import endpoint_prediction
from .data import ClinicalTransform
from .engine import fixed_noise
from .losses import reconstruction, masked_l1, ssim3d


class MRIFeatureFusion(nn.Module):
    def __init__(self, channels, base=32, structural_downsample='interpolate', fusion_mode='concat_1x1'):
        super().__init__()
        if fusion_mode not in ('concat_1x1', 'gated'):
            raise ValueError('Unknown MRI feature fusion mode')
        self.fusion_mode = fusion_mode
        if structural_downsample not in ('interpolate','strided_conv'):
            raise ValueError('Unknown structural_downsample mode')
        self.structural_downsample=structural_downsample
        self.downsample_projection = (strided_mri_stem(base,channels) if structural_downsample=='strided_conv'
            else nn.Sequential(nn.Conv3d(1,channels,1),nn.SiLU()))
        if fusion_mode == 'gated':
            self.gate = nn.Conv3d(2*channels,channels,1)
        else:
            self.fuse = nn.Conv3d(2*channels,channels,1)

    def forward(self, mri, za):
        down = mri if self.structural_downsample=='strided_conv' else F.interpolate(mri,size=za.shape[2:],mode='trilinear',align_corners=False)
        zs = self.downsample_projection(down)
        if zs.shape!=za.shape:
            raise ValueError('ZS/ZA shape mismatch: strided-convolution MRI inputs must have spatial dimensions divisible by 4')
        joined = torch.cat([zs,za],dim=1)
        if self.fusion_mode == 'gated':
            gate = torch.sigmoid(self.gate(joined))
            return zs, gate*za + (1-gate)*zs
        return zs, self.fuse(joined)


class FusedConditionVelocity(ResidualUNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.condition = nn.Conv3d(kwargs['latent'],kwargs['channels'][0],1)
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def conditions(self, fused):
        levels = [self.condition(fused)]
        for down in self.cond_down:
            levels.append(down(levels[-1]))
        return levels

    def forward(self, z, t, conditions, clinical):
        return super().forward(z,t*1000,conditions,clinical)


class FusedMRIGenerator(nn.Module):
    def __init__(self, cfg, settings=None):
        super().__init__()
        settings=settings or {}
        self.mri = Autoencoder(cfg['ae_base'],cfg['latent_channels'],cfg['checkpointing'])
        self.pet = Autoencoder(cfg['ae_base'],cfg['latent_channels'],cfg['checkpointing'])
        if settings.get('use_coarse',False):
            self.coarse = CoarseMapper(cfg['ae_base'],cfg['latent_channels'],cfg.get('structural_downsample','interpolate'))
        else:
            self.fusion = MRIFeatureFusion(cfg['latent_channels'],cfg['ae_base'],cfg.get('structural_downsample','interpolate'),
                settings.get('fusion', 'concat_1x1'))
        self.velocity = FusedConditionVelocity(latent=cfg['latent_channels'],channels=cfg['unet_channels'],
            time_dim=cfg['time_dim'],heads=cfg['attention_heads'],dropout=cfg['dropout'],
            checkpointing=cfg['checkpointing'],clinical_dim=cfg.get('clinical_dim',8))

    def source(self, mri):
        with torch.no_grad():
            za = self.mri.encode(mri)
        mapper = self.coarse if hasattr(self,'coarse') else self.fusion
        zs, fused = mapper(mri,za)
        return zs, za, fused


class FusionFlowTrainer(FlowTrainer):
    generator_class = FusedMRIGenerator
    stage_name = 'mri_downsample_aae_fusion_flow'

    def __init__(self,cfg,*args,**kwargs):
        settings=cfg['fusion_flow']
        for key in ('use_coarse','freeze_coarse'):
            settings.setdefault(key,False)
            if type(settings[key]) is not bool:raise ValueError(key+' must be boolean')
        settings.setdefault('coarse_checkpoint',None)
        if cfg['fusion_flow']['source'] != 'mri_downsample+aae':
            raise ValueError('This experiment requires BOTH MRI branches')
        super().__init__(cfg,*args,**kwargs)
        self.normalization = None

    def build_generator(self):
        return self.generator_class(self.cfg['model'],self.cfg['fusion_flow'])

    def trainable_modules(self):
        settings=self.cfg['fusion_flow']
        names=['velocity']
        if not settings['use_coarse']:names.insert(0,'fusion')
        elif not settings['freeze_coarse']:names.insert(0,'coarse')
        return names

    def configure(self, training=True):
        self.gen.eval().requires_grad_(False)
        modules=[getattr(self.gen,n) for n in self.trainable_modules()]
        for module in modules:
            module.train(training).requires_grad_(training)
        return [p for m in modules for p in m.parameters()]

    def set_normalization(self,state):
        if state['fit_split']!='train' or state['fold']!=self.fold:
            raise ValueError('Normalization must come from the training fold')
        if self.metadata_hash is not None and state['metadata_hash']!=self.metadata_hash:
            raise ValueError('Normalization metadata mismatch')
        c=self.cfg['model']['latent_channels']
        mu=torch.tensor(state['mean'],device=self.device,dtype=torch.float32)
        std=torch.tensor(state['std'],device=self.device,dtype=torch.float32)
        if mu.numel()!=c or std.numel()!=c or not torch.isfinite(mu).all() or not torch.isfinite(std).all() or (std<0).any():
            raise ValueError('Invalid PET coordinate statistics')
        self.mu=mu.view(1,c,1,1,1)
        self.std=std.clamp_min(self.cfg['fusion_flow']['std_floor']).view(1,c,1,1,1)
        self.normalization=state

    def encode_coordinates(self,z):return (z.float()-self.mu)/self.std
    def decode_coordinates(self,z):return z.float()*self.std+self.mu

    def initialize(self,path):
        path=Path(path)
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['format_version']!='mri-aae-pet-backbone-v1':
            raise ValueError('Expected an AE-only bundle, not a Coarse/Flow checkpoint')
        if state['fold']!=self.fold or state['metadata_hash']!=self.metadata_hash:
            raise ValueError('AE bundle fold/metadata mismatch')
        for k in ('latent_channels','ae_base'):
            if state['backbone_config'][k]!=self.cfg['model'][k]:
                raise ValueError('AE architecture mismatch')
        for name in ('mri','pet'):
            getattr(self.gen,name).load_state_dict(state[name],strict=True)
        self.clinical=ClinicalTransform(state['clinical'])
        self.set_normalization(state['normalization'])
        self.provenance=dict(checkpoint=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             source=state['provenance'])
        settings=self.cfg['fusion_flow']
        if settings['use_coarse']:
            coarse_path=settings['coarse_checkpoint']
            if not coarse_path and settings['freeze_coarse']:
                raise ValueError('Frozen Coarse requires a pretrained coarse_checkpoint')
            if coarse_path:
                coarse_path=Path(coarse_path)
                coarse=torch.load(coarse_path,map_location='cpu',weights_only=False)
                pretrain=coarse.get('format_version')=='generation-clinical4-v1' and coarse.get('stage')=='coarse'
                joint=coarse.get('format_version')=='mri-fusion-flow-v1' and coarse.get('config',{}).get('fusion_flow',{}).get('use_coarse') is True
                if not (pretrain or joint):
                    raise ValueError('Expected a pretrained Coarse-stage or compatible Coarse-enabled fusion Flow checkpoint')
                if coarse['fold']!=self.fold or coarse['metadata_hash']!=self.metadata_hash:
                    raise ValueError('Coarse checkpoint fold/metadata mismatch')
                if coarse['config']['model'].get('structural_downsample','interpolate')!=self.cfg['model'].get('structural_downsample','interpolate'):
                    raise ValueError('Coarse structural_downsample mismatch; convolutional Coarse requires matching pretrained weights')
                weights=coarse['generator']
                for name in ('mri','pet'):
                    actual={k[len(name)+1:]:v for k,v in weights.items() if k.startswith(name+'.')}
                    expected=getattr(self.gen,name).state_dict()
                    if actual.keys()!=expected.keys() or any(not torch.equal(actual[k].cpu(),v.cpu()) for k,v in expected.items()):
                        raise ValueError('Coarse checkpoint uses different AE weights')
                if coarse['clinical']!=state['clinical']:
                    raise ValueError('Coarse clinical transform mismatch')
                self.gen.coarse.load_state_dict({k[7:]:v for k,v in weights.items() if k.startswith('coarse.')},strict=True)
                self.provenance['coarse']=dict(checkpoint=str(coarse_path.resolve()),sha256=hashlib.sha256(coarse_path.read_bytes()).hexdigest())
            else:self.provenance['coarse']=dict(initialization='random_joint_training')
        return state

    def checkpoint(self,*args):
        state=super().checkpoint(*args)
        state.update(format_version='mri-fusion-flow-v1',normalization=self.normalization)
        return state

    def load(self,path):
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['format_version']!='mri-fusion-flow-v1':
            raise ValueError('Expected a two-branch MRI fusion Flow checkpoint')
        self.verify(state)
        if state['config']['fusion_flow']!=self.cfg['fusion_flow']:
            raise ValueError('Fusion model configuration mismatch')
        self.gen.load_state_dict(state['generator'],strict=True)
        self.clinical=ClinicalTransform(state['clinical']);self.provenance=state['provenance']
        self.set_normalization(state['normalization'])
        return state

    def loss(self,b):
        zs,za,fused=self.gen.source(b['input'])
        with torch.no_grad():p=self.encode_coordinates(self.gen.pet.encode(b['target']))
        c=self.encode_coordinates(fused)
        f=self.cfg['flow']
        z0=source_state(c,f['source_sigma'])
        t=torch.rand(c.shape[0],device=self.device)
        t=torch.where(torch.rand_like(t)<self.cfg['fusion_flow']['anchor_probability'],torch.zeros_like(t),t)
        zt,target=flow_pair(z0,p,t)
        conditions=self.gen.velocity.conditions(fused)
        v=self.gen.velocity(zt,t,conditions,b['clinical']).float()
        fm=F.mse_loss(v,target.detach())
        estimate=endpoint_prediction(zt,t,v)
        end=F.l1_loss(estimate,p)
        pet=reconstruction(self.gen.pet.decode(self.decode_coordinates(estimate)).float(),
            b['target'].float(),b['brain_mask'],self.cfg['loss'])
        total=fm+f['endpoint_weight']*end+f['pet_weight']*pet
        return total,{k:float(v.detach()) for k,v in dict(flow=fm,endpoint=end,pet=pet,path=fm.new_zeros(())).items()}

    @torch.no_grad()
    def generate(self,mri,clinical,ids,trajectory=False):
        self.gen.eval();f=self.cfg['flow']
        with self.autocast():
            zs,za,fused=self.gen.source(mri)
            c=self.encode_coordinates(fused)
            z0=source_state(c,f['source_sigma'],fixed_noise(c,ids,self.cfg['seed']))
            conditions=self.gen.velocity.conditions(fused)
            end,states=integrate(self.gen.velocity,z0,conditions,clinical,f['sample_steps'],f['sample_solver'],trajectory)
            zf=self.decode_coordinates(end)
            image=self.gen.pet.decode(zf).float()
            if not torch.isfinite(image).all():raise FloatingPointError('Non-finite fused Flow image')
            payload=dict(ZS=zs.float(),ZA=za.float(),Z0=self.decode_coordinates(z0),ZF=zf)
            if hasattr(self.gen,'coarse'):payload['ZC']=fused.float()
            if trajectory:
                payload.update({f'Z_{t:.3f}':self.decode_coordinates(z) for t,z in states.items()})
        return image,payload,{},[]

    @torch.no_grad()
    def validate(self,loader):
        self.configure(False);rows=[];start=time.monotonic()
        for index,raw in enumerate(loader):
            b=self.move(raw);out,latents,_,_=self.generate(b['input'],b['clinical'],b['sample_id'])
            with self.autocast():
                zp=self.gen.pet.encode(b['target']).float()
                source_image=self.gen.pet.decode(latents['Z0']).float()
            for i,sid in enumerate(b['sample_id']):
                x,y,mask=out[i:i+1],b['target'][i:i+1].float(),b['brain_mask'][i:i+1]
                z0,zf,p=latents['Z0'][i:i+1],latents['ZF'][i:i+1],zp[i:i+1]
                error=(x-y).square();se=(source_image[i:i+1]-y).square()
                peak=self.cfg['loss']['data_range']**2
                psnr=lambda mse:float(10*torch.log10(peak/mse.clamp_min(1e-12)))
                rows.append(dict(sample_id=sid,mae=float(masked_l1(x,y,mask)),
                    full_mae=float(F.l1_loss(x,y)),
                    psnr=psnr((error*mask).sum()/mask.sum()),full_psnr=psnr(error.mean()),
                    ssim=float(ssim3d(x,y,self.cfg['loss']['data_range'],mask=mask)[0]),
                    full_ssim=float(ssim3d(x,y,self.cfg['loss']['data_range'])[0]),
                    source_psnr=psnr((se*mask).sum()/mask.sum()),source_full_psnr=psnr(se.mean()),
                    d_source=float(F.l1_loss(z0,p)),d_final=float(F.l1_loss(zf,p)),
                    displacement=float(F.l1_loss(zf,z0))))
            if index==0 or (index+1)%25==0 or index+1==len(loader):
                print(json.dumps(dict(event='fusion_flow_validation',batch=index+1,batches=len(loader),
                                      seconds=round(time.monotonic()-start,1))),flush=True)
        if not rows:raise ValueError('Empty validation split')
        mean={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='sample_id'}
        if not all(np.isfinite(v) for v in mean.values()):raise FloatingPointError('Invalid metrics')
        keys=('mae','full_mae','psnr','full_psnr','ssim','full_ssim')
        return dict(selection_loss=mean['mae'],samples=len(rows),generation={k:mean[k] for k in keys},
                    transport={k:v for k,v in mean.items() if k not in keys}),rows
