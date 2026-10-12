"""Haar residual Flow with an optional, strictly MRI-only support projector.

The parameter names/shapes and initialization are identical for every arm.
No target or cached evaluation mask is accepted by the prediction interface.
"""
import torch
from residual_model import ResidualFlow


def mri_support(mri):
    if mri.ndim != 5 or mri.shape[1] != 1 or not torch.isfinite(mri).all():
        raise ValueError('Expected finite normalized MRI [B,1,D,H,W]')
    mask = (mri.detach().abs() > 1e-6).float()
    if not (mask.flatten(1).sum(1) > 0).all():
        raise ValueError('Empty MRI support')
    return mask


class SupportedFlow(ResidualFlow):
    def __init__(self, support_mode='none'):
        if support_mode not in ('none', 'mri'):
            raise ValueError('Only full Haar with none/mri support is supported')
        super().__init__('haar')  # Exactly one initialization; no new parameters.
        self.support_mode = support_mode

    def support(self, mri):
        return mri_support(mri) if self.support_mode == 'mri' else None

    def project_support(self, coefficients, support):
        if self.support_mode == 'none':
            if support is not None:
                raise ValueError('Unconstrained arm must receive support=None')
            return coefficients
        if support is None or support.requires_grad:
            raise ValueError('MRI arm requires a fixed support tensor')
        image = self.decode(coefficients)
        if image.shape != support.shape:
            raise ValueError('Support shape mismatch')
        return self.encode(image * support)

    def velocity(self, y, t, conditions, clinical, support):
        return self.project_support(super().forward(y, t, conditions, clinical), support)

    def integrate_supported(self, shape, conditions, clinical, support, steps=2):
        if steps != 2:
            raise ValueError('This protocol fixes both train and inference to two Euler steps')
        y = torch.zeros(shape, device=clinical.device, dtype=torch.float32)
        for i in range(steps):
            t = torch.full((shape[0],), i / steps, device=y.device)
            y = y + self.velocity(y, t, conditions, clinical, support) / steps
        return y

    def predict_image(self, mri, anchor, clinical, residual_scale, steps=2):
        """Deployable inputs only. Ground truth is deliberately absent."""
        support = self.support(mri)
        conditions = self.conditions(mri, anchor)
        coefficients = self.integrate_supported(
            self.encode(anchor).shape, conditions, clinical, support, steps)
        return anchor.float() + residual_scale * self.decode(coefficients)
