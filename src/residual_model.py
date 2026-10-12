import torch
from torch import nn
from torch.nn import functional as F
from mlrd.models import ResidualUNet
from wave_tools import haar,inverse

def pack(x):
 b,c,d,h,w=x.shape;assert d%2==h%2==w%2==0
 return x.reshape(b,c,d//2,2,h//2,2,w//2,2).permute(0,1,3,5,7,2,4,6).reshape(b,c*8,d//2,h//2,w//2)
def unpack(x):
 b,c,d,h,w=x.shape;assert c%8==0
 return x.reshape(b,c//8,2,2,2,d,h,w).permute(0,1,5,2,6,3,7,4).reshape(b,c//8,d*2,h*2,w*2)
class ResidualFlow(nn.Module):
 def __init__(self,kind):
  super().__init__();self.kind=kind
  self.net=ResidualUNet(latent=8,channels=(24,48,72,96),time_dim=128,heads=4,dropout=0.,checkpointing=True,clinical_dim=0)
  self.net.condition=nn.Conv3d(16,24,1)
  nn.init.zeros_(self.net.output[-1].weight);nn.init.zeros_(self.net.output[-1].bias)
 def encode(self,x):return pack(x.float()) if self.kind=='s2c' else haar(x)
 def decode(self,x):return unpack(x.float()) if self.kind=='s2c' else inverse(x)
 def project(self,x):
  if self.kind!='haar_high':return x
  return torch.cat([torch.zeros_like(x[:,:1]),x[:,1:]],dim=1)
 def conditions(self,mri,anchor):
  cs=[self.net.condition(torch.cat([self.encode(mri),self.encode(anchor)],dim=1))]
  for layer in self.net.cond_down:cs.append(layer(cs[-1]))
  return cs
 def forward(self,y,t,conditions,clinical):return self.project(self.net(y,t*1000,conditions,clinical).float())
 def integrate(self,shape,conditions,clinical,steps):
  y=torch.zeros(shape,device=clinical.device,dtype=torch.float32)
  for i in range(steps):y=y+self(y,torch.full((shape[0],),i/steps,device=y.device),conditions,clinical)/steps
  return y

def unit_tests():
 torch.set_num_threads(8);torch.manual_seed(42);x=torch.randn(1,1,12,16,20,requires_grad=True);out={}
 for kind in ['s2c','haar','haar_high']:
  m=ResidualFlow(kind);w=m.encode(x);z=m.decode(w)
  assert (z-x).abs().max()<2e-6 and torch.allclose(w.square().sum(),x.square().sum(),rtol=1e-6)
  r=m.decode(m.project(w));r.square().mean().backward(retain_graph=True)
  if kind=='haar_high':assert F.avg_pool3d(r,2,2).abs().max()<1e-6
  out[kind]={'inverse_max_error':float((z-x).abs().max().detach()),'parameters':sum(p.numel() for p in m.parameters())}
 assert x.grad.abs().sum()>0;return out
