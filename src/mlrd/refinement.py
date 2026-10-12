"""Controlled mapper baseline and normalized, endpoint-supervised Flow experiment."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .flow_engine import FlowTrainer
from .flow import flow_pair, source_state, integrate
from .engine import fixed_noise
from .models import ResBlock
from .data import ClinicalTransform
from .losses import reconstruction,masked_l1,ssim3d


def endpoint_prediction(zt,t,velocity):
    return zt.float()+(1-t.float()).view(-1,1,1,1,1)*velocity.float()


class StaticMapper(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.net=nn.Sequential(nn.Conv3d(3*channels,channels,1),ResBlock(channels),ResBlock(channels),
                               ResBlock(channels),nn.Conv3d(channels,channels,1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def conditions(self,zs,za,zc): return (zs,za,zc)
    def forward(self,z,t,conditions,clinical): return self.net(torch.cat(conditions,1))


class RefinementTrainer(FlowTrainer):
    def __init__(self,cfg,*args,**kwargs):
        super().__init__(cfg,*args,**kwargs)
        self.kind=cfg['refinement']['kind']
        if self.kind not in ('flow','mapper'): raise ValueError('Unknown refinement model')
        if self.kind=='mapper': self.gen.velocity=StaticMapper(cfg['model']['latent_channels']).to(self.device)
        self.normalization=None
        self.configure()

    def set_normalization(self,state):
        c=self.cfg['model']['latent_channels']
        if state['fit_split']!='train' or state['fold']!=self.fold:
            raise ValueError('Normalization must be fitted on the training fold only')
        if self.metadata_hash and state['metadata_hash']!=self.metadata_hash:
            raise ValueError('Normalization metadata mismatch')
        if state['coarse_sha256']!=self.provenance['sha256']:
            raise ValueError('Normalization encoder/coarse provenance mismatch')
        mu=torch.tensor(state['channels']['ZP']['mean'],device=self.device,dtype=torch.float32)
        std=torch.tensor(state['channels']['ZP']['std'],device=self.device,dtype=torch.float32)
        if mu.numel()!=c or std.numel()!=c or not torch.isfinite(mu).all() or not torch.isfinite(std).all() or (std<0).any():
            raise ValueError('Invalid channel statistics')
        self.mu=mu.view(1,c,1,1,1)
        self.std=std.clamp_min(self.cfg['refinement'].get('std_floor',1e-3)).view(1,c,1,1,1)
        self.normalization=state

    def encode_coordinates(self,z): return (z.float()-self.mu)/self.std
    def decode_coordinates(self,z): return z.float()*self.std+self.mu

    def initialize(self,path):
        state=super().initialize(path)
        self.set_normalization(json.loads(Path(self.cfg['refinement']['statistics']).read_text()))
        return state

    def checkpoint(self,*args):
        state=super().checkpoint(*args)
        state.update(format_version='mlt-refinement-v2',refinement_kind=self.kind,normalization=self.normalization)
        return state

    def load(self,path):
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state.get('format_version')!='mlt-refinement-v2' or state['refinement_kind']!=self.kind:
            raise ValueError('Refinement checkpoint type mismatch')
        self.verify(state)
        if state['config']['refinement']!=self.cfg['refinement']:
            raise ValueError('Refinement configuration mismatch')
        self.gen.load_state_dict(state['generator'],strict=True)
        self.clinical=ClinicalTransform(state['clinical']); self.provenance=state['provenance']
        self.set_normalization(state['normalization'])
        return state

    def loss(self,b):
        f=self.cfg['flow']; settings=self.cfg['refinement']
        with torch.no_grad():
            zs,za,zc=self.gen.source(b['input']); zp=self.gen.pet.encode(b['target'])
            c,p=self.encode_coordinates(zc),self.encode_coordinates(zp)
        conditions=self.gen.velocity.conditions(zs,za,zc)
        if self.kind=='mapper':
            estimate=c+self.gen.velocity(c,None,conditions,b['clinical']).float()
            end=F.l1_loss(estimate,p)
            cosine=1-F.cosine_similarity(estimate.flatten(1),p.flatten(1),dim=1).mean()
            return end+.1*cosine,dict(flow=0.,endpoint=float(end.detach()),pet=0.,path=0.)
        z0=source_state(c,f['source_sigma'])
        t=torch.rand(c.shape[0],device=self.device)
        # Expose exact inference anchors as well as teacher-path states.
        t=torch.where(torch.rand_like(t)<settings.get('anchor_probability',.25),torch.zeros_like(t),t)
        zt,target=flow_pair(z0,p,t)
        v=self.gen.velocity(zt,t,conditions,b['clinical']).float()
        fm=F.mse_loss(v,target)
        estimate=endpoint_prediction(zt,t,v)
        end=F.l1_loss(estimate,p)
        # This is explicitly a teacher-path endpoint proxy, not a solved ODE endpoint.
        pet=reconstruction(self.gen.pet.decode(self.decode_coordinates(estimate)).float(),b['target'].float(),b['brain_mask'],self.cfg['loss'])
        loss=fm+f['endpoint_weight']*end+f['pet_weight']*pet
        return loss,{k:float(x.detach()) for k,x in dict(flow=fm,endpoint=end,pet=pet,path=fm.new_zeros(())).items()}

    @torch.no_grad()
    def generate(self,mri,clinical,ids,trajectory=False):
        self.gen.eval(); f=self.cfg['flow']
        with self.autocast():
            zs,za,zc=self.gen.source(mri); c=self.encode_coordinates(zc)
            conditions=self.gen.velocity.conditions(zs,za,zc)
            z0=source_state(c,f['source_sigma'],fixed_noise(c,ids,self.cfg['seed']))
            if self.kind=='mapper':
                endpoint=c+self.gen.velocity(c,None,conditions,clinical).float(); states={}
            else:
                endpoint,states=integrate(self.gen.velocity,z0,conditions,clinical,f['sample_steps'],f['sample_solver'],trajectory)
            zf=self.decode_coordinates(endpoint)
            pet=self.gen.pet.decode(zf).float()
            if not torch.isfinite(pet).all(): raise FloatingPointError('Non-finite refinement output')
            payload=dict(ZS=zs.float(),ZA=za.float(),ZC=zc.float(),Z0=self.decode_coordinates(z0),ZF=zf)
            routes,velocities=[],{}
            if trajectory:
                for t,z in states.items():
                    if any(abs(t-k)<1e-7 for k in (0.,.25,.5,.75,1.)):
                        payload[f'Z_{t:.2f}']=self.decode_coordinates(z)
                        v,rs=self.gen.velocity(z,z.new_full((z.shape[0],),t),conditions,clinical,True)
                        velocities[f'V_{t:.2f}']=v.float()*self.std
                        for block,rec in enumerate(rs): routes.append(dict(t=t,block=block,level=rec['level'],weights=rec['weights'].float().cpu().tolist()))
        return pet,payload,velocities,routes

    @torch.no_grad()
    def validate(self,loader):
        self.configure(False); rows=[]; start=time.monotonic()
        for index,raw in enumerate(loader):
            b=self.move(raw); out,latents,_,_=self.generate(b['input'],b['clinical'],b['sample_id'])
            with self.autocast():
                zp=self.gen.pet.encode(b['target']).float()
                coarse=self.gen.pet.decode(latents['ZC']).float()
            for i,sid in enumerate(b['sample_id']):
                x,y,mask=out[i:i+1],b['target'][i:i+1].float(),b['brain_mask'][i:i+1]
                zc,zf,p=latents['ZC'][i:i+1],latents['ZF'][i:i+1],zp[i:i+1]
                mse=((x-y).square()*mask).sum()/mask.sum()
                cmse=((coarse[i:i+1]-y).square()*mask).sum()/mask.sum()
                rows.append(dict(sample_id=sid,mae=float(masked_l1(x,y,mask)),
                    psnr=float(10*torch.log10(self.cfg['loss']['data_range']**2/mse.clamp_min(1e-12))),
                    ssim=float(ssim3d(x,y,self.cfg['loss']['data_range'],mask=mask)[0]),
                    coarse_psnr=float(10*torch.log10(self.cfg['loss']['data_range']**2/cmse.clamp_min(1e-12))),
                    d_C=float(F.l1_loss(zc,p)),d_F=float(F.l1_loss(zf,p)),displacement=float(F.l1_loss(zf,zc)),
                    cos_C=float(F.cosine_similarity(zc.flatten(1),p.flatten(1)).mean()),
                    cos_F=float(F.cosine_similarity(zf.flatten(1),p.flatten(1)).mean())))
            if index==0 or (index+1)%25==0 or index+1==len(loader):
                print(json.dumps(dict(event='refinement_validation',kind=self.kind,batch=index+1,batches=len(loader),
                    mae=float(np.mean([r['mae'] for r in rows])),seconds=round(time.monotonic()-start,1))),flush=True)
        mean={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='sample_id'}
        if not all(np.isfinite(v) for v in mean.values()): raise FloatingPointError('Nonfinite validation')
        return dict(selection_loss=mean['mae'],samples=len(rows),generation={k:mean[k] for k in ('mae','psnr','ssim')},
                    transport={k:v for k,v in mean.items() if k not in ('mae','psnr','ssim')}),rows
