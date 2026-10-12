import torch
from torch.nn import functional as F


def masked_l1(pred, target, mask):
    # Equal weight per scan; an empty MRI foreground is a data error.
    count = mask.flatten(1).sum(1)
    if (count == 0).any():
        raise ValueError("Empty brain mask")
    return (((pred-target).abs()*mask).flatten(1).sum(1)/count).mean()


def ssim3d(pred, target, data_range=1., window=7, mask=None):
    """Local 3D SSIM using a uniform window; fixed configured intensity range."""
    pred, target = pred.float(), target.float()
    window = min(window, *pred.shape[2:])
    window = window if window % 2 else window-1
    def avg(x):
        return F.avg_pool3d(x, window, stride=1, padding=window//2, count_include_pad=False)
    mx, my = avg(pred), avg(target)
    vx, vy = (avg(pred*pred)-mx*mx).clamp_min(0), (avg(target*target)-my*my).clamp_min(0)
    cov = avg(pred*target)-mx*my
    c1, c2 = (.01*data_range)**2, (.03*data_range)**2
    score = ((2*mx*my+c1)*(2*cov+c2))/((mx*mx+my*my+c1)*(vx+vy+c2)).clamp_min(1e-12)
    if mask is None:
        return score.flatten(1).mean(1)
    return (score*mask).flatten(1).sum(1)/mask.flatten(1).sum(1).clamp_min(1)


def gradient_l1(pred, target, mask):
    losses = []
    for dim in [2, 3, 4]:
        a, b = [slice(None)]*5, [slice(None)]*5
        a[dim], b[dim] = slice(1, None), slice(None, -1)
        valid = mask[tuple(a)]*mask[tuple(b)]
        diff = (torch.diff(pred, dim=dim)-torch.diff(target, dim=dim)).abs()
        losses.append((diff*valid).sum()/valid.sum().clamp_min(1))
    return torch.stack(losses).mean()


def reconstruction(pred, target, mask, cfg):
    l1 = masked_l1(pred, target, mask)
    ssim = 1-ssim3d(pred, target, cfg["data_range"], mask=mask).mean()
    grad = gradient_l1(pred, target, mask) if cfg["gradient"] else l1.new_zeros(())
    return cfg["image"]*l1+cfg["ssim"]*ssim+cfg["gradient"]*grad
