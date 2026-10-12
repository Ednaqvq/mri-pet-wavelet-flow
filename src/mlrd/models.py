from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class SpatialGroupNorm(nn.GroupNorm):
    def forward(self, x):
        # Equivalent normalization over C/group × spatial elements. A 3D view
        # avoids ambiguous channels-last strides at a 1×1×1 CPU bottleneck.
        return super().forward(x.reshape(x.shape[0], x.shape[1], -1)).reshape_as(x)


def norm(channels):
    # Keep at least two channels per group, including tiny CPU test bottlenecks.
    groups = max(g for g in range(1, min(8, max(1, channels//2))+1) if channels % g == 0)
    return SpatialGroupNorm(groups, channels)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(norm(channels), nn.SiLU(), nn.Conv3d(channels, channels, 3, padding=1),
                                 norm(channels), nn.SiLU(), nn.Conv3d(channels, channels, 3, padding=1))

    def forward(self, x):
        return x + self.net(x)


class Encoder(nn.Module):
    def __init__(self, base, latent):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(1, base, 4, 2, 1), ResBlock(base), ResBlock(base),
                                 nn.Conv3d(base, latent, 4, 2, 1), ResBlock(latent), ResBlock(latent),
                                 nn.Conv3d(latent, latent, 3, padding=1))

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, base, latent):
        super().__init__()
        self.net = nn.Sequential(ResBlock(latent), ResBlock(latent),
                                 nn.ConvTranspose3d(latent, base, 4, 2, 1), ResBlock(base),
                                 nn.ConvTranspose3d(base, max(4, base//2), 4, 2, 1),
                                 nn.SiLU(), nn.Conv3d(max(4, base//2), 1, 3, padding=1))

    def forward(self, x):
        return self.net(x)


class Autoencoder(nn.Module):
    def __init__(self, base, latent, checkpointing=True):
        super().__init__()
        self.encoder, self.decoder = Encoder(base, latent), Decoder(base, latent)
        self.checkpointing = checkpointing

    def encode(self, x):
        if self.checkpointing and torch.is_grad_enabled() and any(p.requires_grad for p in self.encoder.parameters()):
            return checkpoint(self.encoder, x, use_reentrant=False)
        return self.encoder(x)

    def decode(self, z):
        if self.checkpointing and torch.is_grad_enabled():
            return checkpoint(self.decoder, z, use_reentrant=False)
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class LatentDiscriminator(nn.Module):
    """Patch discriminator regularizing the spatial MRI latent toward N(0,I)."""
    def __init__(self, latent):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(latent, latent, 3, padding=1), nn.LeakyReLU(.2),
                                 nn.Conv3d(latent, latent, 3, stride=2, padding=1), nn.LeakyReLU(.2),
                                 nn.Conv3d(latent, 1, 1))

    def forward(self, z):
        return self.net(z)


def strided_mri_stem(base, latent):
    """Learn both spatial reductions before fusing with the frozen MRI encoder."""
    return nn.Sequential(nn.Conv3d(1, base, 3, stride=2, padding=1), norm(base), nn.SiLU(),
                         nn.Conv3d(base, latent, 3, stride=2, padding=1), norm(latent), nn.SiLU())


class CoarseMapper(nn.Module):
    def __init__(self, base, latent, structural_downsample='interpolate'):
        super().__init__()
        if structural_downsample not in ('interpolate','strided_conv'):
            raise ValueError('Unknown structural_downsample mode')
        self.structural_downsample = structural_downsample
        self.structural = (strided_mri_stem(base, latent) if structural_downsample=='strided_conv' else
            nn.Sequential(nn.Conv3d(1, base, 3, padding=1), norm(base), nn.SiLU(),
                          nn.Conv3d(base, latent, 3, padding=1)))
        self.gate = nn.Conv3d(2*latent, latent, 1)
        self.refine = nn.Sequential(*[ResBlock(latent) for _ in range(3)])

    def forward(self, mri, za):
        x = mri if self.structural_downsample=='strided_conv' else F.interpolate(mri, size=za.shape[2:], mode="trilinear", align_corners=False)
        zs = self.structural(x)
        if zs.shape!=za.shape:
            raise ValueError('ZS/ZA shape mismatch: strided-convolution MRI inputs must have spatial dimensions divisible by 4')
        gate = torch.sigmoid(self.gate(torch.cat([zs, za], 1)))
        return zs, self.refine(gate*za + (1-gate)*zs)


def timestep_embedding(t, width=128):
    f = torch.exp(-math.log(10000)*torch.arange(width//2, device=t.device)/(width//2))
    phase = t.float()[:, None]*f[None]
    return torch.cat([phase.cos(), phase.sin()], -1)


class ConditionalBlock(nn.Module):
    def __init__(self, ch, time_dim, dropout):
        super().__init__()
        self.n1, self.n2 = norm(ch), norm(ch)
        self.c1, self.c2 = nn.Conv3d(ch, ch, 3, padding=1), nn.Conv3d(ch, ch, 3, padding=1)
        self.condition, self.time = nn.Conv3d(ch, 2*ch, 1), nn.Linear(time_dim, 2*ch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, condition, time):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = (self.condition(condition)+self.time(time)[:, :, None, None, None]).chunk(2, 1)
        h = (1+scale)*self.n2(h)+shift
        return x + self.c2(self.dropout(F.silu(h)))


class ResidualUNet(nn.Module):
    def __init__(self, latent=64, channels=(64, 128, 256, 384), time_dim=512,
                 heads=6, dropout=.1, checkpointing=True, clinical_dim=8):
        super().__init__()
        self.checkpointing = checkpointing
        self.time = nn.Sequential(nn.Linear(128, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.clinical_dim = clinical_dim
        if clinical_dim < 0:
            raise ValueError('clinical_dim must be nonnegative')
        self.clinical = (nn.Sequential(nn.Linear(clinical_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
                         if clinical_dim else None)
        self.input = nn.Conv3d(latent, channels[0], 3, padding=1)
        self.condition = nn.Conv3d(3*latent, channels[0], 1)
        self.cond_down = nn.ModuleList(nn.Conv3d(a, b, 3, 2, 1) for a, b in zip(channels, channels[1:]))
        self.down = nn.ModuleList(nn.Conv3d(a, b, 3, 2, 1) for a, b in zip(channels, channels[1:]))
        self.enc = nn.ModuleList(nn.ModuleList(ConditionalBlock(c, time_dim, dropout) for _ in range(2)) for c in channels)
        self.attn_norm = nn.LayerNorm(channels[-1])
        self.attn = nn.MultiheadAttention(channels[-1], heads, batch_first=True)
        self.up = nn.ModuleList(nn.Conv3d(channels[i+1], channels[i], 3, padding=1) for i in range(3))
        self.merge = nn.ModuleList(nn.Conv3d(2*c, c, 1) for c in channels[:3])
        self.dec = nn.ModuleList(nn.ModuleList(ConditionalBlock(c, time_dim, dropout) for _ in range(2)) for c in channels[:3])
        self.output = nn.Sequential(norm(channels[0]), nn.SiLU(), nn.Conv3d(channels[0], latent, 3, padding=1))

    def block(self, module, *args):
        if self.checkpointing and torch.is_grad_enabled():
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def conditions(self, zs, za, zc):
        cs = [self.condition(torch.cat([zs, za, zc], 1))]
        for layer in self.cond_down:
            cs.append(layer(cs[-1]))
        return cs

    def forward(self, residual, t, conditions, clinical):
        if clinical.shape != (residual.shape[0], self.clinical_dim):
            raise ValueError(f"Expected clinical tensor [B,{self.clinical_dim}]")
        e = self.time(timestep_embedding(t))
        if self.clinical is not None:
            e = e + self.clinical(clinical)
        h, skips = self.input(residual), []
        for i, blocks in enumerate(self.enc):
            h = self.block(blocks[0], h, conditions[i], e)
            if i == 3:
                shape = h.shape
                tokens = h.flatten(2).transpose(1, 2)
                q = self.attn_norm(tokens)
                tokens = tokens + self.attn(q, q, q, need_weights=False)[0]
                h = tokens.transpose(1, 2).reshape(shape).contiguous()
            h = self.block(blocks[1], h, conditions[i], e)
            skips.append(h)
            if i < 3:
                h = self.down[i](h)
        for i in reversed(range(3)):
            h = self.up[i](F.interpolate(h, size=skips[i].shape[2:], mode="trilinear", align_corners=False))
            h = self.merge[i](torch.cat([h, skips[i]], 1))
            for block in self.dec[i]:
                h = self.block(block, h, conditions[i], e)
        return self.output(h)


class Generator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mri = Autoencoder(cfg["ae_base"], cfg["latent_channels"], cfg["checkpointing"])
        self.pet = Autoencoder(cfg["ae_base"], cfg["latent_channels"], cfg["checkpointing"])
        self.coarse = CoarseMapper(cfg["ae_base"], cfg["latent_channels"], cfg.get('structural_downsample','interpolate'))
        self.denoiser = ResidualUNet(cfg["latent_channels"], cfg["unet_channels"], cfg["time_dim"],
                                    cfg["attention_heads"], cfg["dropout"], cfg["checkpointing"], cfg.get("clinical_dim", 8))

    def source(self, mri):
        za = self.mri.encode(mri)
        if self.coarse.training and self.denoiser.checkpointing and torch.is_grad_enabled():
            zs, zc = checkpoint(self.coarse, mri, za, use_reentrant=False)
        else:
            zs, zc = self.coarse(mri, za)
        return zs, za, zc
