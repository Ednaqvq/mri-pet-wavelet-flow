"""Train MRI AAE -> PET AE -> joint Coarse/Flow -> frozen Coarse/Flow -> F residual."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='8'
import argparse,json,hashlib,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--metadata',required=True,help='CSV: subject_id,sample_id,t1_path,fdg_pet_path,has_fdg_pet,split,fold')
 p.add_argument('--dataset-root',required=True);p.add_argument('--run-dir',required=True)
 p.add_argument('--cohort',choices=['ADNI','NACC'],required=True);p.add_argument('--seed',type=int,default=42)
 p.add_argument('--config',default=str(Path(__file__).parent/'configs/default.json'))
 p.add_argument('--smoke',action='store_true',help='Two scans per partition, one epoch per stage; writes under preflight/')
 a=p.parse_args();run=Path(a.run_dir).resolve();cam=run/'campaign';os.environ['WAVE_FLOW_CAMPAIGN']=str(cam)
 sys.path.insert(0,str(Path(__file__).parent/'src'))
 from common import pd,torch,sha,save,read
 from pipeline import run_job
 import residual_pipeline
 residual_pipeline.ARMS={'F_multih_no_boundary':([3,5,6,7],False)}
 if torch.cuda.device_count()!=1:raise ValueError('Select exactly one GPU with CUDA_VISIBLE_DEVICES')
 df=pd.read_csv(a.metadata,dtype={'subject_id':str,'sample_id':str})
 required=['subject_id','sample_id','t1_path','fdg_pet_path','has_fdg_pet','split','fold']
 if not set(required)<=set(df):raise ValueError('Missing metadata columns')
 df=df[required].copy();base=Path(a.dataset_root).resolve()
 if df.sample_id.duplicated().any():raise ValueError('sample_id must be unique')
 if df[required].isna().any().any():raise ValueError('Missing metadata values')
 for col in ['t1_path','fdg_pet_path']:
  df[col]=df[col].map(lambda v:str((base/v).resolve()) if not Path(v).is_absolute() else str(Path(v).resolve()))
  if not all(Path(x).is_relative_to(base) and Path(x).is_file() for x in df[col]):raise ValueError('Missing image or path outside dataset root')
 from mlrd.data import select_rows
 groups={k:set(select_rows(df,k,0).subject_id) for k in ['train','val','final_val']}
 if not groups['train'] or not groups['val']:raise ValueError('Training and validation must be nonempty')
 if any(groups[x]&groups[y] for x,y in [('train','val'),('train','final_val'),('val','final_val')]):raise ValueError('Subjects overlap between partitions')
 if not set(df['split'])<= {'train','final_val'}:raise ValueError('Use train/fold!=0 for train, train/fold=0 for validation, final_val for test')
 cam.mkdir(parents=True,exist_ok=True);prepared=cam/'prepared';prepared.mkdir(exist_ok=True)
 text=df.to_csv(index=False);mp=prepared/'metadata.csv'
 if mp.exists() and mp.read_text()!=text:raise ValueError('Existing run has different metadata')
 mp.write_text(text)
 cfg=read(a.config);cfg.update(seed=a.seed,cohort=a.cohort,output_dir=str(run/'ae'))
 cfg['data'].update(dataset_root=str(base),prepared_dir=str(prepared),cache_dir=str(cam/'image_cache'),clinical_features=[],clinical_csv=None)
 cfg['model']['clinical_dim']=0;cfg['training'].update(num_workers=0,batch_size=1,accumulation_steps=4)
 cfg['runtime']['cpu_threads']=8;cfg['flow']['init_checkpoint']=str(run/'ae_bundle.pt')
 cfg['fusion_flow'].update(use_coarse=True,freeze_coarse=False,coarse_checkpoint=None)
 cp=cam/'config.json'
 if cp.exists() and read(cp)!=cfg:raise ValueError('Existing run has different configuration')
 save(cp,cfg);src=Path(__file__).parent/'src'
 hashes={str(f.relative_to(src)):sha(f) for f in sorted(src.rglob('*.py'))}
 manifest=dict(code=str(src.resolve()),source_sha256=hashes,datasets={a.cohort:dict(metadata=str(mp),metadata_sha256=sha(mp),root=str(base))},
  jobs=[dict(cohort=a.cohort,seed=a.seed,run=str(run),config=str(cp),config_sha256=sha(cp))])
 mf=cam/'manifest.json'
 if mf.exists() and read(mf)!=manifest:raise ValueError('Existing run has different source/configuration; use a new directory')
 save(mf,manifest);run_job(str(cp),a.smoke)

if __name__=='__main__':main()
