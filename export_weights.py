"""Export a trained F run as a portable pair, retaining every model tensor."""
import argparse,copy,json,hashlib
from pathlib import Path
import torch
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def export(run,output):
 run=Path(run);out=Path(output)
 if out.exists():raise FileExistsError(out)
 ap=run/'frozen/fold_0/flow/best.pt';rp=run/'F_multih_no_boundary/best.pt'
 a=torch.load(ap,map_location='cpu',weights_only=False);r=torch.load(rp,map_location='cpu',weights_only=False)
 assert r['anchor_sha256']==sha(ap) and r['arm']=='F_multih_no_boundary' and r['clinical_dim']==0
 assert a['config']['model']['clinical_dim']==0 and a['config']['seed']==r['seed']
 cfg=copy.deepcopy(a['config']);cfg['output_dir']='runs/inference'
 cfg['data'].update(dataset_root='data',prepared_dir='data/prepared',cache_dir=None,clinical_features=[],clinical_csv=None)
 cfg['flow']['init_checkpoint']='ae_bundle.pt';cfg['fusion_flow']['coarse_checkpoint']='joint/fold_0/flow/best.pt'
 clean={k:a[k] for k in ['format_version','stage','epoch','fold','metadata_hash','generator','clinical','normalization']}
 clean.update(config=cfg,provenance=dict(cohort=cfg['cohort'],seed=cfg['seed'],source_anchor_sha256=sha(ap),release='v1.0.0'))
 out.mkdir(parents=True);torch.save(clean,out/'anchor.pt')
 rc={k:r[k] for k in ['model','epoch','arm','seed','clinical_dim']};rc['anchor_sha256']=sha(out/'anchor.pt');torch.save(rc,out/'residual.pt')
 for original,key,name in [(a,'generator','anchor.pt'),(r,'model','residual.pt')]:
  saved=torch.load(out/name,map_location='cpu',weights_only=True)
  assert original[key].keys()==saved[key].keys()
  for k,v in original[key].items():assert torch.equal(v,saved[key][k]) and torch.isfinite(v).all(),k
 info=dict(cohort=cfg['cohort'],seed=cfg['seed'],method=r['arm'],files={f:dict(sha256=sha(out/f),bytes=(out/f).stat().st_size) for f in ['anchor.pt','residual.pt']})
 (out/'weights.json').write_text(json.dumps(info,indent=2));return info
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(export(a.run_dir,a.output),indent=2))
