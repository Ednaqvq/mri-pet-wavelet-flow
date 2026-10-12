"""Analytic contracts for Haar, the selected robust loss, and MRI support."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from wave_tools import haar,inverse
from multih_huber_loss import quadratic_matched_huber,selective_huber_fm_loss
from output_support import verify_contract
from supported_flow import SupportedFlow
torch.set_num_threads(8)
class CoreTests(unittest.TestCase):
 def test_haar_inverse_energy_and_gradient(self):
  torch.manual_seed(42);x=torch.randn(2,1,16,20,16,requires_grad=True);w=haar(x)
  self.assertLess(float((inverse(w)-x).abs().max().detach()),2e-6)
  torch.testing.assert_close(w.square().sum(),x.square().sum(),rtol=1e-6,atol=1e-5)
  inverse(w).sum().backward();torch.testing.assert_close(x.grad,torch.ones_like(x),rtol=1e-6,atol=1e-6)
 def test_huber_analytic_value_and_gradient(self):
  x=torch.tensor([-3.,-1.,-.5,0.,.5,1.,3.],requires_grad=True)
  y=quadratic_matched_huber(x,torch.tensor(1.));torch.testing.assert_close(y,torch.tensor([5.,1.,.25,0.,.25,1.,5.]))
  y.sum().backward();torch.testing.assert_close(x.grad,torch.tensor([-2.,-2.,-1.,0.,1.,2.,2.]))
 def test_large_threshold_matches_masked_mse_with_partial_blocks(self):
  torch.manual_seed(42);x=torch.randn(2,1,8,10,12,dtype=torch.float64,requires_grad=True)
  m=(torch.rand_like(x)>.4).double();actual=selective_huber_fm_loss(x,m,torch.full((4,),1e6),selected=[3,5,6,7],protect_boundary=False)
  expected=((x.square()*m).flatten(1).sum(1)/m.flatten(1).sum(1)).mean()
  torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
  actual.backward();self.assertTrue(torch.isfinite(x.grad).all());self.assertEqual(int(torch.count_nonzero(x.grad[m==0])),0)
 def test_final_output_contract(self):self.assertTrue(all(verify_contract().values()))
 def test_velocity_projection_idempotent(self):
  torch.manual_seed(42);model=SupportedFlow('mri');v=torch.randn(1,8,8,10,8);m=(torch.rand(1,1,16,20,16)>.4).float()
  p=model.project_support(v,m);pp=model.project_support(p,m)
  torch.testing.assert_close(p,pp,rtol=2e-6,atol=2e-6)
  self.assertLess(float((model.decode(p)*(1-m)).abs().max()),2e-6)
if __name__=='__main__':unittest.main()
