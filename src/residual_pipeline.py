"""Source-trained Haar residuals, matched ablations, locked target-test evaluation."""
from common import *
import gc,copy,math
from torch.nn import functional as F
from torch.utils.data import Dataset
from supported_flow import SupportedFlow,mri_support
from multih_huber_loss import selective_huber_fm_loss
from wave_tools import haar,Atlas
SCALE=.1
ARMS={'C_mse':None,'D_multih_boundary':([3,5,6,7],True),'E_all_boundary':(list(range(8)),True),
 'F_multih_no_boundary':([3,5,6,7],False),'G_all_no_boundary':(list(range(8)),False)}

def cache_anchor(cfg,root,limit=None):
 from pipeline import SourceFlow
 folder=root/'anchor_cache';folder.mkdir(exist_ok=True);mp=folder/'manifest.json'
 anchor_path=root/'frozen/fold_0/flow/best.pt';ah=sha(anchor_path)
 if mp.exists():assert read(mp)['anchor_sha256']==ah;return
 t=SourceFlow(cfg,0,'cuda:0');t.load(anchor_path);t.gen.eval().requires_grad_(False)
 records=[];sq=torch.zeros(8,dtype=torch.float64,device='cuda');den=0.;train_ids=[]
 with torch.no_grad():
  for part in ['train','val']:
   for i,raw in enumerate(loader(cfg,part,limit)):
    check_stop();b=move(raw);path=folder/(b['sample_id'][0]+'.pt')
    if path.exists():
     old=torch.load(path,map_location='cpu',weights_only=False);assert old['anchor_sha256']==ah
     pred=old['anchor'][None].cuda()
    else:
     pred=t.generate(b['input'],b['clinical'],b['sample_id'])[0]
     checkpoint(path,dict(anchor=pred[0].cpu(),sample_id=b['sample_id'][0],anchor_sha256=ah))
    if part=='train':
     waves=haar((b['target']-pred).float());support=F.avg_pool3d(b['brain_mask'],2,2)
     sq+=(waves.double().square()*support).sum((0,2,3,4));den+=float(support.sum());train_ids+=b['sample_id']
    records.append(dict(sample_id=b['sample_id'][0],subject_id=b['subject_id'][0],part=part,path=str(path)))
    if i%50==0:print(json.dumps(dict(event='anchor_cache',part=part,index=i+1)),flush=True)
 rms=(sq/den).sqrt();assert torch.isfinite(rms).all() and (rms>0).all()
 save(mp,dict(records=records,anchor_sha256=ah,train_only_rms=rms.cpu().tolist(),statistic_sample_ids=train_ids,
  clinical_dim=0,test_accessed=False,metadata_sha256=sha(Path(cfg['data']['prepared_dir'])/'metadata.csv')))
 del t;gc.collect();torch.cuda.empty_cache()

class Cached(Dataset):
 def __init__(self,cfg,root,part):
  self.cfg=cfg;self.man=read(root/'anchor_cache/manifest.json');recs=[r for r in self.man['records'] if r['part']==part]
  meta=pd.read_csv(Path(cfg['data']['prepared_dir'])/'metadata.csv').set_index('sample_id')
  self.rows=meta.loc[[r['sample_id'] for r in recs]].reset_index();self.records=recs
  self.ds=TrainingDataset(self.rows,cfg['data'],ClinicalTransform.fit(None))
 def __len__(self):return len(self.records)
 def __getitem__(self,i):
  b=self.ds[i];s=torch.load(self.records[i]['path'],map_location='cpu',weights_only=False)
  assert s['anchor_sha256']==self.man['anchor_sha256'] and s['sample_id']==b['sample_id'];b['anchor']=s['anchor'];return b

@torch.no_grad()
def evaluate_model(model,dl):
 model.eval();rows=[]
 for raw in dl:
  b=move(raw)
  with torch.autocast('cuda',dtype=torch.bfloat16):x=model.predict_image(b['input'],b['anchor'],b['clinical'],SCALE)
  rows.append(metric(x,b['target'],b['brain_mask'],b))
 return summary(rows),rows

def loss(model,b,time_rng,arm,thresholds):
 m=b['brain_mask'].float();assert torch.equal(m,mri_support(b['input']))
 target=b['target'].float();a=b['anchor'].float();cond=model.conditions(b['input'],a)
 r=model.project_support(model.encode((target-a)/SCALE),m).detach()
 t=torch.rand(1,generator=time_rng);t=torch.where(torch.rand(1,generator=time_rng)<.5,0.,t).cuda();tt=t[:,None,None,None,None]
 v=model.velocity(r*tt,t,cond,b['clinical'],m);err=model.decode(v-r)
 spec=ARMS[arm]
 fm=((err.square()*m).sum()/m.sum()) if spec is None else selective_huber_fm_loss(err,m,thresholds,selected=spec[0],protect_boundary=spec[1])
 end=a+SCALE*model.decode(r*tt+(1-tt)*v)
 final=model.integrate_supported(r.shape,cond,b['clinical'],m,2);pred=a+SCALE*model.decode(final)
 return fm+.25*masked_l1(end,target,m)/SCALE+masked_l1(pred,target,m)/SCALE+.2*(1-ssim3d(pred,target,2.,mask=m).mean())

def train_residuals(cfg,root,preflight=False):
 cache=read(root/'anchor_cache/manifest.json');train_ds=Cached(cfg,root,'train');val_ds=Cached(cfg,root,'val')
 val=DataLoader(val_ds,batch_size=1,num_workers=0,pin_memory=True);hashes=read(CAM/'manifest.json')['source_sha256']
 for arm,spec in ARMS.items():
  check_stop();out=root/arm;out.mkdir(exist_ok=True)
  if (out/'complete.json').exists():continue
  seed_all(cfg['seed']);model=SupportedFlow('mri').cuda();assert model.net.clinical is None
  initial=digest(model);opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
  sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=3,threshold=1e-5,threshold_mode='abs',min_lr=1e-6)
  thresholds=None if spec is None else torch.tensor([cache['train_only_rms'][i]/SCALE for i in spec[0]],dtype=torch.float64,device='cuda')
  start=1;best=math.inf;stale=0;history=[];best_epoch=0
  if (out/'last.pt').exists():
   saved=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
   assert saved['source_sha256']==hashes and saved['anchor_sha256']==cache['anchor_sha256'] and saved['initial_digest']==initial
   model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);sched.load_state_dict(saved['scheduler'])
   start=saved['epoch']+1;best=saved['best'];stale=saved['stale'];history=saved['history'];best_epoch=saved['best_epoch']
   torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng']);random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);del saved
  else:
   q,rows=evaluate_model(model,val);best=q['subject']['mae']
   checkpoint(out/'best.pt',dict(model=model.state_dict(),epoch=0,metrics=q,arm=arm,seed=cfg['seed'],source_sha256=hashes,
      anchor_sha256=cache['anchor_sha256'],clinical_dim=0,initial_digest=initial))
   pd.DataFrame(rows).to_csv(out/'best_val_per_scan.csv',index=False)
  for epoch in range(start,(1 if preflight else 60)+1):
   check_stop()
   if stale>=12:break
   started=time.time();model.train();rng=torch.Generator().manual_seed(cfg['seed']+1000003*epoch)
   dl=DataLoader(train_ds,batch_size=1,shuffle=True,num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(cfg['seed']+epoch))
   order=hashlib.sha256();total=0.;updates=0;opt.zero_grad(set_to_none=True)
   for i,raw in enumerate(dl):
    b=move(raw);order.update((b['sample_id'][0]+'\n').encode());div=min(4,len(dl)-(i//4)*4)
    with torch.autocast('cuda',dtype=torch.bfloat16):v=loss(model,b,rng,arm,thresholds)
    assert torch.isfinite(v);(v/div).backward();total+=float(v.detach())
    if (i+1)%4==0 or i+1==len(dl):
     torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True);updates+=1
    if i%50==0:
     save(out/'status.json',dict(status='running',epoch=epoch,batch=i+1,batches=len(dl),pid=os.getpid()))
     print(json.dumps(dict(event='residual',arm=arm,epoch=epoch,batch=i+1,loss=total/(i+1))),flush=True)
   assert updates==math.ceil(len(dl)/4) and digest(model)!=initial
   row=dict(epoch=epoch,train_loss=total/len(dl),updates=updates,tail_group=((len(dl)-1)%4)+1,seconds=time.time()-started,order_sha256=order.hexdigest())
   if epoch==1 or epoch%2==0 or epoch==60:
    q,rows=evaluate_model(model,val);score=q['subject']['mae'];row['validation']=q;sched.step(score)
    if score<best:
     best=score;best_epoch=epoch;stale=0
     checkpoint(out/'best.pt',dict(model=model.state_dict(),epoch=epoch,metrics=q,arm=arm,seed=cfg['seed'],source_sha256=hashes,
       anchor_sha256=cache['anchor_sha256'],clinical_dim=0,initial_digest=initial))
     pd.DataFrame(rows).to_csv(out/'best_val_per_scan.csv',index=False)
    else:stale+=1
   history.append(row)
   checkpoint(out/'last.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=epoch,
    best=best,best_epoch=best_epoch,stale=stale,history=history,initial_digest=initial,source_sha256=hashes,anchor_sha256=cache['anchor_sha256'],
    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state()))
   (out/'history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
  save(out/'complete.json',dict(best_epoch=best_epoch,stop_epoch=history[-1]['epoch'],best=best,initial_digest=initial,
    best_sha256=sha(out/'best.pt'),anchor_sha256=cache['anchor_sha256'],thresholds=None if thresholds is None else thresholds.cpu().tolist(),
    train_scans=len(train_ds),val_scans=len(val_ds),clinical_dim=0,preflight=preflight))
  save(out/'status.json',dict(status='complete',best_epoch=best_epoch))
  del model,opt,sched;gc.collect();torch.cuda.empty_cache()

def evaluate_tests(*args, **kwargs):
 raise RuntimeError('Use evaluate.py for corrected, MRI-support-constrained evaluation')
