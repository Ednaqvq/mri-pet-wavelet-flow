from __future__ import annotations

import math
import torch
from torch import nn


class Diffusion(nn.Module):
    def __init__(self, steps=1000, prediction="epsilon", residual_scale=1.):
        super().__init__()
        if prediction not in ["epsilon", "v"] or residual_scale <= 0 or steps < 2:
            raise ValueError("Invalid diffusion configuration")
        self.steps, self.prediction, self.residual_scale = steps, prediction, residual_scale
        t = torch.linspace(0, steps, steps+1, dtype=torch.float64)/steps
        a = torch.cos((t+.008)/1.008*math.pi/2).square()
        beta = (1-a[1:]/a[:-1]).clamp(max=.999)
        self.register_buffer("alpha_bar", (1-beta).cumprod(0).float())

    def coefficients(self, t):
        a = self.alpha_bar[t].view(-1, 1, 1, 1, 1)
        return a.sqrt(), (1-a).sqrt()

    def q_sample(self, r0, t, noise):
        a, s = self.coefficients(t)
        return a*r0+s*noise

    def target(self, r0, t, noise):
        a, s = self.coefficients(t)
        return noise if self.prediction == "epsilon" else a*noise-s*r0

    def recover(self, rt, t, output):
        a, s = self.coefficients(t)
        if self.prediction == "epsilon":
            return (rt-s*output)/a, output
        return a*rt-s*output, s*rt+a*output

    def sampling_timesteps(self, steps, device, timestep_spacing="leading"):
        if not 1 <= steps <= self.steps:
            raise ValueError("Sampling steps must be within 1..training steps")
        if timestep_spacing == "leading":
            # Standard DDIM integer stride: 1000/50 -> 980, 960, ..., 0.
            # Avoid forcing epsilon prediction to divide by the almost-zero
            # signal coefficient at t=999 in the cosine schedule.
            return torch.arange(steps-1, -1, -1, device=device)*(self.steps//steps)
        if timestep_spacing == "linspace":
            return torch.linspace(self.steps-1, 0, steps, device=device).round().long()
        raise ValueError("timestep_spacing must be leading or linspace")

    def sample(self, denoiser, latents, steps=50, noise=None, clinical=None,
               timestep_spacing="leading"):
        """DDIM eta=0; spacing is explicit and solver arithmetic stays FP32.

        No clipping of residuals or non-finite replacement is performed.
        """
        zs, za, zc = latents
        times = self.sampling_timesteps(steps, zc.device, timestep_spacing)
        rt = torch.randn_like(zc, dtype=torch.float32) if noise is None else noise.float()
        conditions = denoiser.conditions(zs, za, zc)
        for i, scalar in enumerate(times):
            t = scalar.expand(rt.shape[0])
            pred = denoiser(rt, t, conditions, clinical)
            r0, eps = self.recover(rt.float(), t, pred.float())
            prev = self.alpha_bar[times[i+1]] if i+1 < len(times) else rt.new_tensor(1.)
            rt = prev.sqrt()*r0 + (1-prev).sqrt()*eps
            if not torch.isfinite(rt).all():
                raise FloatingPointError(f"Non-finite DDIM state at t={int(scalar)}, spacing={timestep_spacing}")
        return zc.float() + self.residual_scale*rt
