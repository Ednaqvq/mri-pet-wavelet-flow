"""Source-anchored flow matching and differentiable ODE integration."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .models import Autoencoder, CoarseMapper, ResidualUNet, timestep_embedding


class LatentRouter(nn.Module):
    def __init__(self, channels, time_dim, level):
        super().__init__()
        self.level = nn.Embedding(4, time_dim)
        self.level_index = level
        self.mlp = nn.Sequential(nn.Linear(channels+2*time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, 4))

    def forward(self, h, time, branches):
        level = self.level.weight[self.level_index].expand(h.shape[0], -1)
        query = torch.cat([h.mean((2, 3, 4)), time, level], -1)
        weights = self.mlp(query).float().softmax(-1).to(h.dtype)
        condition = sum(weights[:, i, None, None, None, None]*branch for i, branch in enumerate(branches))
        return condition, weights


class TransitionVelocityUNet(ResidualUNet):
    """Four independent condition pyramids; dynamic routing at every ResBlock."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        latent = kwargs['latent']
        channels, time_dim = kwargs['channels'], kwargs['time_dim']
        del self.condition, self.cond_down
        self.branch_inputs = nn.ModuleList(nn.Conv3d(n*latent, channels[0], 1) for n in (1, 1, 1, 2))
        self.branch_downs = nn.ModuleList(nn.ModuleList(nn.Conv3d(a, b, 3, 2, 1)
            for a, b in zip(channels, channels[1:])) for _ in range(4))
        self.routers = nn.ModuleList(LatentRouter(c, time_dim, i) for i, c in enumerate(channels))
        # Initial endpoint is the source anchor, rather than a random large displacement.
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def conditions(self, zs, za, zc):
        raw = (zs, za, zc, torch.cat([za-zs, zc-za], 1))
        pyramids = []
        for x, first, downs in zip(raw, self.branch_inputs, self.branch_downs):
            pyramid = [first(x)]
            for down in downs:
                pyramid.append(down(pyramid[-1]))
            pyramids.append(pyramid)
        return tuple(tuple(p[i] for p in pyramids) for i in range(4))

    def forward(self, z, t, conditions, clinical, return_routes=False):
        if clinical.shape != (z.shape[0], self.clinical_dim):
            raise ValueError(f'Expected clinical tensor [B,{self.clinical_dim}]')
        # Continuous [0,1] time; rescale only the Fourier frequencies.
        et = self.time(timestep_embedding(t*1000))
        e = et+self.clinical(clinical)
        h, skips, routes = self.input(z), [], []

        def routed(block, h, level):
            condition, weights = self.routers[level](h, et, conditions[level])
            if return_routes:
                routes.append(dict(level=level, weights=weights))
            return self.block(block, h, condition, e)

        for i, blocks in enumerate(self.enc):
            h = routed(blocks[0], h, i)
            if i == 3:
                shape = h.shape
                tokens = h.flatten(2).transpose(1, 2)
                q = self.attn_norm(tokens)
                h = (tokens+self.attn(q, q, q, need_weights=False)[0]).transpose(1, 2).reshape(shape).contiguous()
            h = routed(blocks[1], h, i)
            skips.append(h)
            if i < 3:
                h = self.down[i](h)
        for i in reversed(range(3)):
            h = self.up[i](F.interpolate(h, size=skips[i].shape[2:], mode='trilinear', align_corners=False))
            h = self.merge[i](torch.cat([h, skips[i]], 1))
            for block in self.dec[i]:
                h = routed(block, h, i)
        velocity = self.output(h)
        return (velocity, routes) if return_routes else velocity


class FlowGenerator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mri = Autoencoder(cfg['ae_base'], cfg['latent_channels'], cfg['checkpointing'])
        self.pet = Autoencoder(cfg['ae_base'], cfg['latent_channels'], cfg['checkpointing'])
        self.coarse = CoarseMapper(cfg['ae_base'], cfg['latent_channels'], cfg.get('structural_downsample','interpolate'))
        self.velocity = TransitionVelocityUNet(latent=cfg['latent_channels'], channels=cfg['unet_channels'],
            time_dim=cfg['time_dim'], heads=cfg['attention_heads'], dropout=cfg['dropout'],
            checkpointing=cfg['checkpointing'], clinical_dim=cfg.get('clinical_dim', 8))

    def source(self, mri):
        za = self.mri.encode(mri)
        zs, zc = self.coarse(mri, za)
        return zs, za, zc


def source_state(zc, sigma, noise=None):
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError('source_sigma must be finite and nonnegative')
    if sigma == 0:
        return zc.float()
    if noise is None:
        noise = torch.randn_like(zc, dtype=torch.float32)
    if noise.shape != zc.shape:
        raise ValueError('Source noise shape mismatch')
    return zc.float()+sigma*noise.float()


def flow_pair(z0, zp, t):
    t = t.reshape(-1, 1, 1, 1, 1).float()
    return (1-t)*z0.float()+t*zp.float(), zp.float()-z0.float()


def integrate(velocity, z0, conditions, clinical, steps=8, solver='heun', save_states=False):
    """No no_grad decorator: rollout endpoints can supervise the learned ODE.

    Euler uses N network calls; Heun uses 2N, including t=1 correction.
    Accumulation is FP32 even when velocity runs under mixed precision.
    """
    if not isinstance(steps, int) or steps < 1 or solver not in ('euler', 'heun'):
        raise ValueError('Expected positive integer steps and euler/heun solver')
    z, dt = z0.float(), 1./steps
    states = {0.: z} if save_states else {}
    for k in range(steps):
        t = z.new_full((z.shape[0],), k/steps)
        v = velocity(z, t, conditions, clinical).float()
        proposal = z+dt*v
        if solver == 'heun':
            v_next = velocity(proposal, t+dt, conditions, clinical).float()
            z = z+.5*dt*(v+v_next)
        else:
            z = proposal
        if not torch.isfinite(z).all():
            raise FloatingPointError(f'Non-finite flow state at step {k+1}/{steps}')
        if save_states:
            states[(k+1)/steps] = z
    return z, states


def path_order_loss(states, zp, za=None, margin=0., mri_weight=0.):
    """Optional ordering on predicted rollout states, never teacher interpolation.

    Fixed spatial pooling is phi; MRI distance is disabled unless explicitly requested.
    Latent axes of independently learned encoders have no guaranteed semantic alignment.
    """
    def distance(x, target):
        size = tuple(min(4, n) for n in x.shape[2:])
        return (F.adaptive_avg_pool3d(x.float(), size)-F.adaptive_avg_pool3d(target.float(), size)).abs().flatten(1).mean(1)
    times = sorted(states)
    terms = []
    for a, b in zip(times, times[1:]):
        terms.append(F.relu(distance(states[b], zp)-distance(states[a], zp)+margin*(b-a)).mean())
        if mri_weight:
            if za is None:
                raise ValueError('MRI ordering requires ZA')
            terms.append(mri_weight*F.relu(distance(states[a], za)-distance(states[b], za)+margin*(b-a)).mean())
    return torch.stack(terms).mean() if terms else zp.new_zeros(())
