"""Evaluate a released model: per-scan and subject-equal metrics on a chosen cohort."""
import argparse,copy,json
from pathlib import Path
from inference import Predictor,torch
from common import pd,metric,summary
from mlrd.data import TrainingDataset,ClinicalTransform,select_rows
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--weights',required=True);p.add_argument('--metadata',required=True)
 p.add_argument('--dataset-root',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--split',choices=['val','final_val'],default='final_val')
 a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 if (out/'per_scan.csv').exists() or (out/'summary.json').exists():raise FileExistsError(out)
 pred=Predictor(a.weights,work_dir=out/'runtime');cfg=copy.deepcopy(pred.cfg['data']);cfg.update(dataset_root=str(Path(a.dataset_root).resolve()),cache_dir=None)
 frame=pd.read_csv(a.metadata,dtype={'sample_id':str,'subject_id':str})
 for key in ['t1_path','fdg_pet_path']:
  frame[key]=frame[key].map(lambda x:str((Path(a.dataset_root)/x).resolve()) if not Path(x).is_absolute() else x)
 ds=TrainingDataset(select_rows(frame,a.split,0),cfg,ClinicalTransform.fit(None));dl=torch.utils.data.DataLoader(ds,batch_size=1,num_workers=0)
 rows=[]
 for i,b in enumerate(dl):
  mri=b['input'].to(pred.device);x=pred.tensor(mri,b['sample_id'][0]);rows.append(metric(x,b['target'].to(pred.device),b['brain_mask'].to(pred.device),b))
  print(f'{i+1}/{len(ds)}',flush=True)
 result=summary(rows);result.update(source=pred.cfg['cohort'],seed=pred.cfg['seed'],method='F_multih_no_boundary',output_policy='mri_support_final_output_v1',checkpoints=pred.digests)
 pd.DataFrame(rows).to_csv(out/'per_scan.csv',index=False);(out/'summary.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
