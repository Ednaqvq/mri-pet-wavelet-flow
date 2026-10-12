"""Differentiable signed Haar supervision and physically aligned template ROI metrics."""
import itertools,math
import numpy as np
import torch
from torch.nn import functional as F
BANDS=[''.join(x) for x in itertools.product('LH',repeat=3)]
def haar(x):
 assert x.ndim==5 and all(n%2==0 for n in x.shape[2:])
 # Orthonormal separable filters, L=(a+b)/sqrt(2), H=(a-b)/sqrt(2).
 filters=[]
 for band in BANDS:
  a=[torch.tensor([1.,1. if c=='L' else -1.],device=x.device,dtype=torch.float32)/math.sqrt(2) for c in band]
  filters.append(a[0][:,None,None]*a[1][None,:,None]*a[2][None,None,:])
 w=torch.stack(filters)[:,None]
 with torch.autocast(device_type=x.device.type,enabled=False):return F.conv3d(x.float(),w.repeat(x.shape[1],1,1,1,1),stride=2,groups=x.shape[1])
def inverse(waves):
 filters=[]
 for band in BANDS:
  a=[torch.tensor([1.,1. if c=='L' else -1.],device=waves.device,dtype=torch.float32)/math.sqrt(2) for c in band]
  filters.append(a[0][:,None,None]*a[1][None,:,None]*a[2][None,None,:])
 w=torch.stack(filters)[:,None];c=waves.shape[1]//8
 with torch.autocast(device_type=waves.device.type,enabled=False):return F.conv_transpose3d(waves.float(),w.repeat(c,1,1,1,1),stride=2,groups=c)
def band_metrics(pred,target,mask):
 wp,wt=haar(pred),haar(target);support=F.avg_pool3d(mask.float(),2,2)
 denom=support.flatten(2).sum(-1).clamp_min(1)
 mae=((wp-wt).abs()*support).flatten(2).sum(-1)/denom
 sq=(((wp-wt).square()*support).flatten(2).sum(-1)/denom).sqrt()
 ref=((wt.square()*support).flatten(2).sum(-1)/denom).sqrt()
 nrmse=sq/ref.clamp_min(1e-8)
 return mae,nrmse
class Atlas:
 def __init__(self,cfg,metadata,device):
  import nibabel as nib
  from nibabel.processing import resample_from_to
  from mlrd.data import select_rows
  import pandas as pd
  from pathlib import Path
  row=select_rows(metadata,'train',0).iloc[0];p=str(row.t1_path)
  for a,b in cfg['data'].get('path_replacements',{}).items():p=p.replace(a,b)
  native=nib.load(p);shape=np.array(cfg['data']['spatial_size']);scale=np.array(native.shape)/shape
  transform=np.eye(4);transform[:3,:3]=np.diag(scale);transform[:3,3]=(scale-1)/2
  affine=native.affine@transform
  atlas=resample_from_to(nib.load(cfg['wave_roi']['atlas']),(tuple(shape),affine),order=0)
  labels=np.rint(atlas.get_fdata()).astype(np.int64)
  assert set(np.unique(labels))==set(range(117))
  self.labels=torch.as_tensor(labels,device=device).flatten();self.spacing=nib.affines.voxel_sizes(affine)
  self.geometry=dict(training_reference=p,shape=shape.tolist(),native_shape=list(native.shape),native_affine=native.affine.tolist(),affine=affine.tolist(),spacing=self.spacing.tolist(),atlas=cfg['wave_roi']['atlas'])
  self.lut=pd.read_csv(cfg['wave_roi']['lut']);self.groups={p:self.lut[self.lut.region_name.str.startswith(p)].region_id.astype(int).tolist() for p in ['Cingulum_Post','Precuneus','Angular','Parietal_Inf','Temporal_Mid','Hippocampus']}
 def errors(self,pred,target,mask):
  diffs=[];maes=[];valid=[]
  for x,y,m in zip(pred,target,mask):
   weight=m.flatten().float();count=torch.zeros(117,device=x.device).scatter_add(0,self.labels,weight)
   bias=torch.zeros_like(count).scatter_add(0,self.labels,(x-y).flatten().float()*weight)/count.clamp_min(1)
   mae=torch.zeros_like(count).scatter_add(0,self.labels,(x-y).flatten().float().abs()*weight)/count.clamp_min(1)
   diffs.append(bias[1:]);maes.append(mae[1:]);valid.append(count[1:]>0)
  return torch.stack(diffs),torch.stack(maes),torch.stack(valid)
 def loss(self,pred,target,mask):
  bias,_,valid=self.errors(pred,target,mask)
  return ((bias.abs()*valid).sum(1)/valid.sum(1).clamp_min(1)).mean()
 def metrics(self,pred,target,mask):
  bias,mae,valid=self.errors(pred,target,mask)
  out={'roi_mean_abs_bias':float((bias.abs()*valid).sum()/valid.sum())}
  for k,ids in self.groups.items():
   # Equal left/right ROI weighting, explicitly distinct from pooled-voxel means.
   ix=torch.tensor(ids,device=pred.device)-1
   out[k+'_bias']=float(bias[:,ix].mean());out[k+'_mae']=float(mae[:,ix].mean())
  return out

def unit_tests():
 torch.manual_seed(1);x=torch.randn(2,1,8,12,16,requires_grad=True);w=haar(x);back=inverse(w)
 err=float((back-x).abs().max());assert err<2e-6
 assert torch.allclose(x.square().sum(),w.square().sum(),rtol=1e-6)
 y=torch.randn_like(x);mae,n=band_metrics(x,y,torch.ones_like(x));loss=mae.mean();loss.backward();assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
 const=haar(torch.ones(1,1,8,8,8));assert const[:,1:].abs().max()<1e-6
 ramp=torch.arange(8.).view(1,1,8,1,1).expand(1,1,8,8,8);assert haar(ramp)[:,4].mean()<0
 return {'status':'passed','max_reconstruction_error':err,'parseval':True,'differentiable':True,'constant_details_zero':True,'sign_order_verified':True,'bands':BANDS}
if __name__=='__main__':
 import json;print(json.dumps(unit_tests()))
