"""Final, deployable MRI-only support projection; no target is accepted.

Apply AFTER the entire anchor/residual pipeline, never before conditioning.
This changes generated background, not SSIM, reference images, or foreground.
"""
import torch

def finalize_pet(prediction, mri):
    if prediction.shape != mri.shape or prediction.ndim != 5 or prediction.shape[1] != 1:
        raise ValueError('Expected matching [B,1,D,H,W] MRI and PET tensors')
    if prediction.device != mri.device or not prediction.is_floating_point() or not mri.is_floating_point():
        raise ValueError('MRI and prediction must be floating tensors on one device')
    if not torch.isfinite(prediction).all() or not torch.isfinite(mri).all():
        raise ValueError('Non-finite prediction or MRI')
    support = mri.detach().abs() > 1e-6
    if not support.flatten(1).any(1).all():
        raise ValueError('Empty MRI support')
    return torch.where(support, prediction, torch.zeros_like(prediction))

def verify_contract():
    torch.manual_seed(42)
    mri=torch.randn(2,1,8,10,12);mri[:,:,:2]=0
    mri[:,:,:,0]=1e-7
    pred=torch.randn_like(mri,requires_grad=True)
    out=finalize_pet(pred,mri);mask=mri.abs()>1e-6
    assert torch.equal(out[mask],pred[mask])
    assert torch.count_nonzero(out[~mask])==0
    assert torch.equal(finalize_pet(out,mri),out)
    out.sum().backward();assert torch.equal(pred.grad,mask.to(pred.dtype))
    for bad in [torch.zeros_like(mri),mri*float('nan')]:
        try:finalize_pet(pred,bad)
        except ValueError:pass
        else:raise AssertionError('Invalid MRI accepted')
    return dict(foreground_bitwise_unchanged=True,background_exact_zero=True,
                idempotent=True,gradient_support_correct=True,no_target_argument=True)
