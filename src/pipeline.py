"""Ten independent source-only pipelines; selected weights never fit target data."""
from common import *
import argparse,copy,gc,socket,traceback
from mlrd.engine import Trainer
from mlrd.fusion_flow import FusionFlowTrainer
from mlrd.models import Autoencoder

def subject_selection(metrics,rows,metadata):
 mapping=metadata.set_index('sample_id').subject_id.to_dict()
 frame=pd.DataFrame(rows);frame['subject_id']=frame.sample_id.map(mapping)
 assert frame.subject_id.notna().all()
 metrics['selection_loss']=float(frame.groupby('subject_id').mae.mean().mean())
 metrics['selection']='subject_equal_foreground_mae'
 return metrics,rows
class SourceAE(Trainer):
 def validate(self,stage,dl):
  return subject_selection(*super().validate(stage,dl),self.metadata)
class SourceFlow(FusionFlowTrainer):
 def validate(self,dl):return subject_selection(*super().validate(dl),self.metadata)
 def initialize(self,path):
  before=digest(self.gen.velocity);v=super().initialize(path)
  assert before==digest(self.gen.velocity),'Velocity was not initialized fresh'
  save(self.run/'initialization.json',dict(fresh_flow=True,velocity_digest=before,clinical_dim=0,
      modules={n:digest(getattr(self.gen,n)) for n in ['mri','pet','coarse','velocity']},trainable=self.trainable_modules()))
  return v
def clear():gc.collect();torch.cuda.empty_cache()

def export_bundle(cfg,limit=None):
 out=Path(cfg['flow']['init_checkpoint']);meta=Path(cfg['data']['prepared_dir'])/'metadata.csv';mh=sha(meta)
 if out.exists():
  old=torch.load(out,map_location='cpu',weights_only=False);assert old['metadata_hash']==mh;return
 weights={};parents={}
 for name,stage in [('mri','mri_aae'),('pet','pet_ae')]:
  path=Path(cfg['output_dir'])/'fold_0'/stage/'best.pt';st=torch.load(path,map_location='cpu',weights_only=False)
  assert st['metadata_hash']==mh and st['config']['seed']==cfg['seed'] and st['clinical']['features']==[]
  weights[name]={k[len(name)+1:]:v for k,v in st['generator'].items() if k.startswith(name+'.')}
  parents[name]=dict(path=str(path),sha256=sha(path),epoch=st['epoch']+1);del st
 pet=Autoencoder(cfg['model']['ae_base'],cfg['model']['latent_channels'],False).cuda().eval()
 pet.load_state_dict(weights['pet']);tot=torch.zeros(64,dtype=torch.float64,device='cuda');sq=torch.zeros_like(tot);count=0;ids=[]
 with torch.no_grad():
  for raw in loader(cfg,'train',limit):
   check_stop();b=move(raw)
   with torch.autocast('cuda',dtype=torch.bfloat16):z=pet.encode(b['target']).float()
   tot+=z.sum((0,2,3,4),dtype=torch.float64);sq+=z.double().square().sum((0,2,3,4));count+=z.numel()//64;ids+=b['sample_id']
 mean=tot/count;std=(sq/count-mean.square()).clamp_min(0).sqrt();assert torch.isfinite(std).all()
 bundle=dict(format_version='mri-aae-pet-backbone-v1',fold=0,metadata_hash=mh,clinical=ClinicalTransform.fit(None).state,
  backbone_config={k:cfg['model'][k] for k in ['ae_base','latent_channels']},
  normalization=dict(fit_split='train',fold=0,metadata_hash=mh,mean=mean.cpu().tolist(),std=std.cpu().tolist()),
  provenance=dict(cohort=cfg['cohort'],seed=cfg['seed'],initialization='random_source_only',parents=parents,
    statistic_sample_ids=ids,clinical_dim=0,imported_target_weights=False),**weights)
 checkpoint(out,bundle);save(out.with_suffix('.json'),dict(sha256=sha(out),parents=parents,train_scans=len(ids)))
 del pet,bundle,weights;clear()

def verify_frozen(t):
 initial=read(t.run/'initialization.json');now={n:digest(getattr(t.gen,n)) for n in initial['modules']}
 for n in initial['modules']:
  assert (now[n]!=initial['modules'][n]) if n in t.trainable_modules() else (now[n]==initial['modules'][n]),n
 assert t.gen.velocity.clinical is None
 save(t.run/'freeze_check.json',dict(passed=True,before=initial['modules'],after=now,trainable=t.trainable_modules()))

def run_job(config,preflight=False):
 manifest=verify_sources();cfg=read(config);root=Path(cfg['output_dir']).parent
 job=next(j for j in manifest['jobs'] if j['cohort']==cfg['cohort'] and j['seed']==cfg['seed'])
 assert sha(config)==job['config_sha256']
 if preflight:
  root=root/'preflight';cfg['output_dir']=str(root/'ae');cfg['flow']['init_checkpoint']=str(root/'ae_bundle.pt')
  cfg['training']['epochs']={k:1 for k in cfg['training']['epochs']}
 limit=2 if preflight else None
 root.mkdir(parents=True,exist_ok=True)
 status=dict(status='running',pid=os.getpid(),host=socket.gethostname(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],
  cohort=cfg['cohort'],seed=cfg['seed'],started=time.time(),preflight=preflight,no_clinical=True)
 def record(**kw):status.update(kw);save(root/'status.json',status)
 try:
  for stage in ['mri_aae','pet_ae']:
   check_stop();done=root/(stage+'_complete.json')
   if done.exists():continue
   record(stage=stage)
   t=SourceAE(cfg,0,'cuda:0');assert t.gen.denoiser.clinical is None
   directory=t.run/stage;t.train(stage,resume=(directory/'last.pt').exists(),max_samples=limit)
   best=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
   save(done,dict(checkpoint=str(directory/'best.pt'),sha256=sha(directory/'best.pt'),epoch=best['epoch']+1))
   del best,t;clear()
  record(stage='train_only_latent_statistics');export_bundle(cfg,limit)
  for frozen in [False,True]:
   stage='frozen' if frozen else 'joint';done=root/(stage+'_complete.json');fc=copy.deepcopy(cfg)
   fc['output_dir']=str(root/stage);fc['fusion_flow']['freeze_coarse']=frozen
   if frozen:fc['fusion_flow']['coarse_checkpoint']=str(root/'joint/fold_0/flow/best.pt')
   if done.exists():continue
   record(stage=stage);check_stop()
   t=SourceFlow(fc,0,'cuda:0');t.train(resume=(t.run/'last.pt').exists(),max_samples=limit)
   t.load(t.run/'best.pt');verify_frozen(t)
   save(done,dict(checkpoint=str(t.run/'best.pt'),sha256=sha(t.run/'best.pt')))
   del t;clear()
  fc=copy.deepcopy(cfg);fc['output_dir']=str(root/'frozen');fc['fusion_flow'].update(freeze_coarse=True,coarse_checkpoint=str(root/'joint/fold_0/flow/best.pt'))
  save(root/'anchor_config.json',fc)
  from residual_pipeline import cache_anchor,train_residuals,evaluate_tests
  record(stage='source_anchor_cache');cache_anchor(fc,root,limit)
  record(stage='residual_training');train_residuals(fc,root,preflight)
  # Evaluation is invoked explicitly with evaluate.py after training.
  record(status='complete',stage='complete',finished=time.time())
 except BaseException as e:
  record(status='stopped' if isinstance(e,InterruptedError) else 'failed',error=repr(e));raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args()
 assert torch.cuda.device_count()==1 and torch.get_num_threads()==8
 try:run_job(a.config,a.preflight)
 except BaseException:traceback.print_exc();raise
