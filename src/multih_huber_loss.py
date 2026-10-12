"""Isolated image-only-v4 copy of the diagonal-Haar robust FM prototype.

Numerical change: use explicit quadratic/linear branches to avoid cancellation
of opposing 2*delta gradient terms in the previous minimum-based expression.
The mathematical penalty and support/band definitions are unchanged.

Input error is image-space FM velocity error in residual/scale units. Support
must be the fixed binary MRI-only mask. Complete 2^3 support blocks use Haar
coefficients: LHH, HLH, HHL and HHH get quadratic-matched Huber; the other four
bands retain squares. Partial blocks retain original supported-voxel image MSE.
Normalize each scan by its ORIGINAL MRI voxel count, then average scans equally.

This changes the standard squared FM regression objective. It neither proves
better PET reconstruction nor introduces a proven-new method. Threshold source/unit verification remains a caller
responsibility: accepting split='train' is an assertion, not provenance proof.
"""

import itertools

import torch
from torch.nn import functional as F


BANDS = tuple(''.join(x) for x in itertools.product('LH', repeat=3))
SELECTED = (3, 5, 6, 7)
UNCHANGED = (0, 1, 2, 4)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def thresholds_from_train_rms(train_band_rms_pet, residual_scale=0.1, *, split='train'):
    """Fixed multiplier ONE times 8-band training PET-residual RMS / scale.

    All eight input bands must follow BANDS order. Returned four thresholds are
    detached copies in SELECTED order. No validation/test threshold estimation,
    silent floor, adaptive multiplier or per-case threshold is implemented.
    """
    _require(split == 'train', 'Threshold statistics must be train-only')
    rms = torch.as_tensor(train_band_rms_pet, dtype=torch.float64).detach().clone()
    scale = torch.as_tensor(residual_scale, dtype=torch.float64).detach().clone()
    _require(rms.shape == (8,), 'Expected eight training-band RMS values in BANDS order')
    _require(scale.numel() == 1, 'residual_scale must be scalar')
    _require(bool(torch.isfinite(rms).all()) and bool((rms > 0).all()),
             'Training RMS must be finite and strictly positive')
    _require(bool(torch.isfinite(scale).all()) and bool((scale > 0).all()),
             'residual_scale must be finite and strictly positive')
    result = rms[list(SELECTED)] / scale.to(device=rms.device).reshape(())
    _require(bool(torch.isfinite(result).all()) and bool((result > 0).all()),
             'Converted thresholds must be finite and strictly positive')
    return result.detach()


def _fixed_delta(delta, reference):
    value = torch.as_tensor(delta)
    _require(not value.requires_grad, 'Thresholds must be detached fixed parameters')
    value = value.detach().to(device=reference.device, dtype=reference.dtype)
    _require(bool(torch.isfinite(value).all()) and bool((value > 0).all()),
             'Thresholds must be finite and strictly positive')
    return value


def quadratic_matched_huber(error, delta):
    """rho=e^2 near zero; rho=2*delta*abs(e)-delta^2 outside (2*torch Huber)."""
    delta = _fixed_delta(delta, error)
    absolute = error.abs()
    # An explicit branch prevents large opposing linear gradients from
    # cancelling numerically inside the quadratic region. Avoid error.square()
    # in the linear branch so extreme residuals cannot overflow unused squares.
    # torch.minimum would assign half of the derivative at equality; where
    # selects the full quadratic derivative there, matching the linear side.
    bounded = torch.where(absolute <= delta, absolute, delta)
    quadratic = bounded.square()
    linear = delta * (2 * absolute - delta)
    return torch.where(absolute <= delta, quadratic, linear)


def haar3(error):
    """Orthogonal one-level Haar in wave_tools.py band/sign order; preserves dtype."""
    filters = []
    for band in BANDS:
        axes = [error.new_tensor([1., 1. if char == 'L' else -1.]) / (2 ** 0.5)
                for char in band]
        filters.append(axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :])
    return F.conv3d(error, torch.stack(filters)[:, None], stride=2)


def multih_huber_fm_loss(error, mri_support, thresholds, *, reduction='mean'):
    """Compute the proposed scalar loss (or per-scan vector with reduction='none').

    error: finite float [B,1,D,H,W], even positive spatial sizes.
    mri_support: same shape/device, fixed boolean or exact binary mask.
    thresholds: four positive detached values in LHH/HLH/HHL/HHH order.

    Empty MRI support in ANY scan is rejected instead of silently diluting a
    batch. All-partial support is valid and reduces exactly to ordinary masked
    MSE. Low-precision error is promoted to float32; float64 stays float64 for QA.
    """
    _require(isinstance(error, torch.Tensor) and error.is_floating_point(), 'error must be floating tensor')
    _require(error.ndim == 5 and error.shape[0] > 0 and error.shape[1] == 1 and
             all(size > 0 and size % 2 == 0 for size in error.shape[2:]),
             'Expected nonempty [B,1,D,H,W] with even spatial dimensions')
    _require(bool(torch.isfinite(error).all()), 'Non-finite error')
    _require(isinstance(mri_support, torch.Tensor) and mri_support.shape == error.shape and
             mri_support.device == error.device, 'MRI support shape/device mismatch')
    _require(not mri_support.requires_grad, 'MRI support must be detached')
    _require(bool(torch.isfinite(mri_support).all()) and
             bool(((mri_support == 0) | (mri_support == 1)).all()), 'MRI support must be finite binary')
    _require(reduction in ('mean', 'none'), 'Only per-scan none or equal-scan mean reduction is supported')
    dtype = torch.float64 if error.dtype == torch.float64 else torch.float32
    # Explicitly exclude ambient autocast from this loss's Parseval bookkeeping.
    with torch.autocast(device_type=error.device.type, enabled=False):
        value = error.to(dtype=dtype)
        support = mri_support.detach().to(dtype=dtype)
        delta = _fixed_delta(thresholds, value)
        _require(delta.shape == (4,), 'Expected four thresholds in SELECTED order')
        counts = support.flatten(1).sum(1)
        _require(bool((counts > 0).all()), 'Empty MRI support scan is not permitted')
        complete = F.avg_pool3d(support, 2, 2) == 1
        complete_voxels = complete.repeat_interleave(2, -3).repeat_interleave(2, -2).repeat_interleave(2, -1)
        partial_support = support * (~complete_voxels).to(dtype=dtype)
        partial_energy = (value.square() * partial_support).flatten(1).sum(1)
        waves = haar3(value)
        complete_weight = complete.to(dtype=dtype)
        ordinary_energy = (waves[:, UNCHANGED].square() * complete_weight).flatten(1).sum(1)
        robust = quadratic_matched_huber(waves[:, SELECTED], delta.view(1, 4, 1, 1, 1))
        robust_energy = (robust * complete_weight).flatten(1).sum(1)
        per_scan = (partial_energy + ordinary_energy + robust_energy) / counts
        _require(bool(torch.isfinite(per_scan).all()), 'Non-finite loss (possible numeric overflow)')
        return per_scan.mean() if reduction == 'mean' else per_scan


def selective_huber_fm_loss(error, mri_support, thresholds, *, selected, protect_boundary):
    """Four-arm ablation: same voxel normalization; only band/edge handling varies.

    Without special boundary handling, Haar is applied to MRI-masked error and
    every block uses the selected penalty. Pure background remains zero. This
    removes the mixed-block MSE fallback, NOT the model's MRI support projection.
    """
    selected = tuple(selected)
    _require(selected in (SELECTED, tuple(range(8))), 'Unsupported selected bands')
    _require(type(protect_boundary) is bool, 'Boundary flag must be boolean')
    _require(error.ndim == 5 and error.shape[1] == 1 and all(n % 2 == 0 for n in error.shape[2:]), 'Invalid error shape')
    _require(error.is_floating_point() and bool(torch.isfinite(error).all()), 'Invalid error')
    _require(mri_support.shape == error.shape and mri_support.device == error.device and
             not mri_support.requires_grad and bool(((mri_support == 0) | (mri_support == 1)).all()), 'Invalid support')
    dtype = torch.float64 if error.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=error.device.type, enabled=False):
        value, support = error.to(dtype), mri_support.detach().to(dtype)
        counts = support.flatten(1).sum(1)
        _require(bool((counts > 0).all()), 'Empty support')
        delta = _fixed_delta(thresholds, value)
        _require(delta.shape == (len(selected),), 'Threshold count mismatch')
        if protect_boundary:
            complete = F.avg_pool3d(support, 2, 2) == 1
            voxels = complete.repeat_interleave(2,-3).repeat_interleave(2,-2).repeat_interleave(2,-1)
            partial_energy = (value.square() * support * (~voxels)).flatten(1).sum(1)
            waves, weight = haar3(value), complete.to(dtype)
        else:
            waves = haar3(value * support)
            weight = torch.ones_like(waves[:, :1])
            partial_energy = torch.zeros_like(counts)
        unselected = tuple(i for i in range(8) if i not in selected)
        ordinary = (waves[:, unselected].square() * weight).flatten(1).sum(1) if unselected else torch.zeros_like(counts)
        robust = quadratic_matched_huber(waves[:, selected], delta.view(1,-1,1,1,1))
        result = (partial_energy + ordinary + (robust * weight).flatten(1).sum(1)) / counts
        _require(bool(torch.isfinite(result).all()), 'Nonfinite loss')
        return result.mean()
