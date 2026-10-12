"""Joint Coarse/Flow ablation; MRI/PET autoencoders stay frozen."""
import time
import numpy as np
import torch
from torch.nn import functional as F
from .refinement import RefinementTrainer, endpoint_prediction
from .flow import source_state, flow_pair
from .losses import reconstruction, masked_l1, ssim3d


class JointCoarseFlowTrainer(RefinementTrainer):
    stage_name = 'joint_coarse_flow'

    def __init__(self, cfg, *args, **kwargs):
        if cfg['refinement'].get('joint_coarse') is not True or cfg['refinement']['kind'] != 'flow':
            raise ValueError('Joint experiment requires flow and joint_coarse=true')
        super().__init__(cfg, *args, **kwargs)

    def configure(self, training=True):
        self.gen.eval().requires_grad_(False)
        for module in (self.gen.velocity, self.gen.coarse):
            module.train(training).requires_grad_(training)
        return list(self.gen.velocity.parameters()) + list(self.gen.coarse.parameters())

    def loss(self, b):
        f, settings = self.cfg['flow'], self.cfg['refinement']
        with torch.no_grad():
            za = self.gen.mri.encode(b['input'])
            p = self.encode_coordinates(self.gen.pet.encode(b['target']))
        # Unlike frozen V2, preserve the graph through both Coarse branches.
        zs, zc = self.gen.coarse(b['input'], za)
        c = self.encode_coordinates(zc)
        conditions = self.gen.velocity.conditions(zs, za, zc)
        z0 = source_state(c, f['source_sigma'])
        t = torch.rand(c.shape[0], device=self.device)
        t = torch.where(torch.rand_like(t) < settings.get('anchor_probability', .25), torch.zeros_like(t), t)
        zt, target = flow_pair(z0, p, t)
        v = self.gen.velocity(zt, t, conditions, b['clinical']).float()
        # The supervised velocity target is fixed within an update. Coarse gradients
        # still flow via zt, conditions and the endpoint/image prediction branches.
        fm = F.mse_loss(v, target.detach())
        estimate = endpoint_prediction(zt, t, v)
        end = F.l1_loss(estimate, p)
        pet = reconstruction(self.gen.pet.decode(self.decode_coordinates(estimate)).float(),
                             b['target'].float(), b['brain_mask'], self.cfg['loss'])
        loss = fm + f['endpoint_weight']*end + f['pet_weight']*pet
        return loss, {k:float(x.detach()) for k,x in dict(flow=fm, endpoint=end, pet=pet, path=fm.new_zeros(())).items()}

    @torch.no_grad()
    def validate(self, loader):
        import json
        self.configure(False)
        rows, started = [], time.monotonic()
        for index, raw in enumerate(loader):
            b = self.move(raw)
            out, latent, _, _ = self.generate(b['input'], b['clinical'], b['sample_id'])
            with self.autocast():
                zp = self.gen.pet.encode(b['target']).float()
                coarse = self.gen.pet.decode(latent['ZC']).float()
            for i, sid in enumerate(b['sample_id']):
                x, y, mask = out[i:i+1], b['target'][i:i+1].float(), b['brain_mask'][i:i+1]
                cx = coarse[i:i+1]
                zc, zf, p = latent['ZC'][i:i+1], latent['ZF'][i:i+1], zp[i:i+1]
                error, cerror = (x-y).square(), (cx-y).square()
                peak = self.cfg['loss']['data_range']**2
                psnr = lambda mse: float(10*torch.log10(peak/mse.clamp_min(1e-12)))
                rows.append(dict(sample_id=sid, mae=float(masked_l1(x,y,mask)),
                    psnr=psnr((error*mask).sum()/mask.sum()), full_psnr=psnr(error.mean()),
                    ssim=float(ssim3d(x,y,self.cfg['loss']['data_range'],mask=mask)[0]),
                    coarse_mae=float(masked_l1(cx,y,mask)),
                    coarse_psnr=psnr((cerror*mask).sum()/mask.sum()), coarse_full_psnr=psnr(cerror.mean()),
                    d_C=float(F.l1_loss(zc,p)), d_F=float(F.l1_loss(zf,p)), displacement=float(F.l1_loss(zf,zc))))
            if index == 0 or (index+1)%25 == 0 or index+1 == len(loader):
                print(json.dumps(dict(event='joint_validation',batch=index+1,batches=len(loader),
                    seconds=round(time.monotonic()-started,1))),flush=True)
        if not rows:
            raise ValueError('Empty validation split')
        mean = {k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'sample_id'}
        if not all(np.isfinite(x) for x in mean.values()):
            raise FloatingPointError('Non-finite joint validation metric')
        return dict(selection_loss=mean['mae'],samples=len(rows),
                    generation={k:mean[k] for k in ('mae','psnr','full_psnr','ssim')},
                    transport={k:v for k,v in mean.items() if k not in ('mae','psnr','full_psnr','ssim')}), rows
