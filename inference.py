"""Portable MRI-only generation from a released paired checkpoint."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='8'
import sys,json,argparse,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'src'))
from common import torch,np,sha
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
from pipeline import SourceFlow
from supported_flow import SupportedFlow
from paired_t1_fdg_dataset import _load_volume,_resize_volume
from output_support import finalize_pet
import nibabel as nib

class Predictor:
 def __init__(self,weights,device='cuda:0',work_dir='./runs/inference'):
  weights=Path(weights);ap=weights/'anchor.pt';rp=weights/'residual.pt'
  self.device=torch.device(device);a=torch.load(ap,map_location='cpu',weights_only=True);r=torch.load(rp,map_location='cpu',weights_only=True)
  assert r['anchor_sha256']==sha(ap) and r['arm']=='F_multih_no_boundary' and r['clinical_dim']==0
  cfg=copy.deepcopy(a['config']);cfg['output_dir']=str(Path(work_dir).resolve());self.cfg=cfg
  self.anchor=SourceFlow(cfg,0,str(self.device),inference_only=True);self.anchor.load(ap);self.anchor.gen.eval().requires_grad_(False)
  self.residual=SupportedFlow('mri').to(self.device).eval().requires_grad_(False);self.residual.load_state_dict(r['model'],strict=True)
  self.digests={'anchor':sha(ap),'residual':sha(rp)}
 @torch.no_grad()
 def tensor(self,mri,sample_id='input'):
  mri=mri.to(self.device)
  if not torch.isfinite(mri).all() or mri.min()<-1.001 or mri.max()>1.001:raise ValueError('Input must be a preprocessed MRI in [-1,1]')
  c=torch.empty(len(mri),0,device=self.device)
  a=self.anchor.generate(mri,c,[sample_id]*len(mri))[0]
  with self.anchor.autocast():raw=self.residual.predict_image(mri,a,c,.1)
  return finalize_pet(raw,mri)
 @torch.no_grad()
 def nifti(self,mri_path,output):
  output=Path(output);sidecar=output.with_suffix(output.suffix+'.json')
  if output.exists() or sidecar.exists():raise FileExistsError(output)
  native=nib.load(mri_path);shape=self.cfg['data']['spatial_size'];mri=_resize_volume(_load_volume(mri_path),shape)[None]
  out=self.tensor(mri,Path(mri_path).stem)
  scale=np.asarray(native.shape[:3])/np.asarray(shape);transform=np.eye(4);transform[:3,:3]=np.diag(scale);transform[:3,3]=(scale-1)/2
  affine=native.affine@transform;output.parent.mkdir(parents=True,exist_ok=True)
  nib.save(nib.Nifti1Image(out[0,0].float().cpu().numpy(),affine),str(output))
  info=dict(method='F_multih_no_boundary',cohort=self.cfg['cohort'],seed=self.cfg['seed'],shape=shape,
   checkpoints=self.digests,output_policy='mri_support_final_output_v1',output_grid='resized input grid',intensity='scaled PET [-1,1] training scale; not absolute SUVR')
  sidecar.write_text(json.dumps(info,indent=2));return info

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--weights',required=True);p.add_argument('--mri',required=True);p.add_argument('--output',required=True)
 p.add_argument('--device',default='cuda:0');a=p.parse_args();print(json.dumps(Predictor(a.weights,a.device,Path(a.output).parent/'runtime').nifti(a.mri,a.output),indent=2))
if __name__=='__main__':main()
