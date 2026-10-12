import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='8'
import hashlib,json,random,time,uuid
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from mlrd.data import TrainingDataset,ClinicalTransform,select_rows
from mlrd.losses import masked_l1,ssim3d
from wave_tools import band_metrics
torch.set_num_threads(8)
CAM=Path(os.environ.get('WAVE_FLOW_CAMPAIGN', './campaign')).resolve()
METRICS=['mae','ssim','psnr','full_mae','full_ssim','full_psnr','wave_detail_mae']
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
 tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(p)
def checkpoint(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');torch.save(obj,tmp);tmp.replace(p)
def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def digest(model):
 h=hashlib.sha256()
 for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def check_stop():
 if (CAM/'STOP_REQUESTED').exists():raise InterruptedError('Campaign stop requested; committed checkpoints retained')
def move(b):return {k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in b.items()}
def loader(cfg,part,max_samples=None):
 df=pd.read_csv(Path(cfg['data']['prepared_dir'])/'metadata.csv');rows=select_rows(df,part,0)
 if max_samples:rows=rows.head(max_samples)
 assert len(rows)>0
 return DataLoader(TrainingDataset(rows,cfg['data'],ClinicalTransform.fit(None)),batch_size=1,num_workers=0,pin_memory=True)
def metric(pred,target,mask,b):
 assert torch.isfinite(pred).all()
 e=(pred.float()-target.float()).square();w,_=band_metrics(pred,target,mask)
 return dict(sample_id=b['sample_id'][0],subject_id=str(b['subject_id'][0]),
   mae=float(masked_l1(pred,target,mask)),ssim=float(ssim3d(pred,target,2.,mask=mask)[0]),
   psnr=float(10*torch.log10(4/((e*mask).sum()/mask.sum()).clamp_min(1e-12))),
   full_mae=float((pred-target).abs().mean()),full_ssim=float(ssim3d(pred,target,2.)[0]),
   full_psnr=float(10*torch.log10(4/e.mean().clamp_min(1e-12))),wave_detail_mae=float(w[:,1:].mean()))
def summary(rows):
 df=pd.DataFrame(rows);assert len(df)==df.sample_id.nunique()
 assert np.isfinite(df[METRICS].to_numpy()).all()
 return dict(scans=len(df),subjects=int(df.subject_id.nunique()),scan=df[METRICS].mean().to_dict(),
   subject=df.groupby('subject_id')[METRICS].mean().mean().to_dict())
def verify_sources():
 m=read(CAM/'manifest.json');code=Path(m['code'])
 for p,h in m['source_sha256'].items():assert sha(code/p)==h,('Code changed',p)
 for ds in m['datasets'].values():assert sha(ds['metadata'])==ds['metadata_sha256']
 return m
