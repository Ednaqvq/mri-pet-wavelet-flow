"""Independent generation-only Flow experiment; reuses data and metric contracts."""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .data import ClinicalTransform, select_rows
from .engine import Trainer, seed_all, fixed_noise, scaled_optimizer_step, safe_json
from .flow import FlowGenerator, source_state, flow_pair, integrate, path_order_loss
from .losses import reconstruction, masked_l1, ssim3d


class FlowTrainer:
    loader, move = Trainer.loader, Trainer.move
    generator_class = FlowGenerator

    def __init__(self, cfg, fold=0, device=None, inference_only=False):
        self.cfg, self.fold = cfg, fold
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        torch.set_num_threads(cfg.get('runtime', {}).get('cpu_threads', 2))
        seed_all(cfg['seed'])
        self.amp = cfg['training']['amp'] and self.device.type == 'cuda'
        precision = cfg.get('runtime', {}).get('precision', 'bf16')
        if precision not in ('bf16', 'fp16', 'fp32'):
            raise ValueError('precision must be bf16, fp16 or fp32')
        self.amp = self.amp and precision != 'fp32'
        if self.amp and precision == 'bf16' and not torch.cuda.is_bf16_supported():
            raise ValueError('GPU does not support bf16; explicitly configure fp16 or fp32')
        self.dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
        self.run = Path(cfg['output_dir'])/f'fold_{fold}'/'flow'
        self.run.mkdir(parents=True, exist_ok=True)
        self.metadata_hash, self.metadata, self.clinical = None, None, None
        if not inference_only:
            prepared = Path(cfg['data']['prepared_dir'])/'metadata.csv'
            self.metadata_hash = hashlib.sha256(prepared.read_bytes()).hexdigest()
            self.metadata = pd.read_csv(prepared)
            train, val = [select_rows(self.metadata, s, fold) for s in ('train', 'val')]
            if train.empty or val.empty or set(train.subject_id) & set(val.subject_id):
                raise ValueError('Empty split or subject leakage')
            self.clinical = ClinicalTransform.fit(train)
        self.gen = self.build_generator().to(self.device)
        for module in self.gen.modules():
            if hasattr(module, 'checkpointing'):
                module.checkpointing = cfg.get('runtime', {}).get('activation_checkpointing', True)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp and self.dtype == torch.float16)
        self.provenance = None
        self.configure()

    def build_generator(self):
        return self.generator_class(self.cfg['model'])

    def autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.amp)

    def configure(self, training=True):
        self.gen.eval().requires_grad_(False)
        self.gen.velocity.train(training).requires_grad_(training)
        return list(self.gen.velocity.parameters())

    def initialize(self, checkpoint):
        path = Path(checkpoint)
        s = torch.load(path, map_location='cpu', weights_only=False)
        if s.get('format_version') != 'generation-clinical4-v1' or s['stage'] != 'coarse':
            raise ValueError('Flow initialization requires a trusted coarse checkpoint')
        self.verify(s, flow=False)
        for name in ('mri', 'pet', 'coarse'):
            subset = {k[len(name)+1:]: v for k, v in s['generator'].items() if k.startswith(name+'.')}
            getattr(self.gen, name).load_state_dict(subset, strict=True)
        self.clinical = ClinicalTransform(s['clinical'])
        self.provenance = dict(checkpoint=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               coarse_epoch=s['epoch']+1, coarse_best_mae=s['best'])
        return s

    def verify(self, s, flow=True):
        if s['fold'] != self.fold or (self.metadata_hash is not None and s['metadata_hash'] != self.metadata_hash):
            raise ValueError('Checkpoint fold or metadata mismatch')
        if s['config']['model'] != self.cfg['model']:
            raise ValueError('Checkpoint model mismatch')
        if s['config']['data']['spatial_size'] != self.cfg['data']['spatial_size']:
            raise ValueError('Checkpoint geometry mismatch')
        if flow and s['config']['flow'] != self.cfg['flow']:
            raise ValueError('Checkpoint Flow definition mismatch')

    def load(self, checkpoint):
        s = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if s.get('format_version') != 'mlt-flow-clinical4-v1':
            raise ValueError('Not an MLT-Flow checkpoint')
        self.verify(s)
        self.gen.load_state_dict(s['generator'], strict=True)
        self.clinical = ClinicalTransform(s['clinical'])
        self.provenance = s['provenance']
        return s

    def loss(self, batch):
        f = self.cfg['flow']
        with torch.no_grad():
            zs, za, zc = self.gen.source(batch['input'])
            zp = self.gen.pet.encode(batch['target']).float()
        conditions = self.gen.velocity.conditions(zs, za, zc)
        z0 = source_state(zc, f['source_sigma'])
        t = torch.rand(z0.shape[0], device=self.device)
        zt, target = flow_pair(z0, zp, t)
        pred = self.gen.velocity(zt, t, conditions, batch['clinical']).float()
        fm = F.mse_loss(pred, target)
        end, pet, path = [fm.new_zeros(()) for _ in range(3)]
        if f['endpoint_weight'] or f['pet_weight'] or f['path_weight']:
            # A real source-only rollout, not a teacher-interpolated endpoint proxy.
            zf, states = integrate(self.gen.velocity, z0, conditions, batch['clinical'],
                f['train_steps'], f['train_solver'], save_states=bool(f['path_weight']))
            end = F.l1_loss(zf, zp)
            if f['pet_weight']:
                out = self.gen.pet.decode(zf)
                pet = reconstruction(out.float(), batch['target'].float(), batch['brain_mask'], self.cfg['loss'])
            if f['path_weight']:
                path = path_order_loss(states, zp, za, f.get('path_margin', 0.), f.get('mri_path_weight', 0.))
        total = fm+f['endpoint_weight']*end+f['pet_weight']*pet+f['path_weight']*path
        return total, {k: float(v.detach()) for k, v in dict(flow=fm, endpoint=end, pet=pet, path=path).items()}

    @torch.no_grad()
    def generate(self, mri, clinical, ids, trajectory=False):
        self.gen.eval()
        f = self.cfg['flow']
        with self.autocast():
            zs, za, zc = self.gen.source(mri)
            z0 = source_state(zc, f['source_sigma'], fixed_noise(zc, ids, self.cfg['seed']))
            conditions = self.gen.velocity.conditions(zs, za, zc)
            zf, states = integrate(self.gen.velocity, z0, conditions, clinical,
                f['sample_steps'], f['sample_solver'], save_states=trajectory)
            out = self.gen.pet.decode(zf).float()
            if not torch.isfinite(out).all():
                raise FloatingPointError('Non-finite Flow PET output')
            payload = dict(ZS=zs.float(), ZA=za.float(), ZC=zc.float(), Z0=z0, ZF=zf)
            routing, velocities = [], {}
            if trajectory:
                # Export actual ODE grid points, not an interpolation between endpoints.
                for t, z in states.items():
                    if any(abs(t-x) < 1e-7 for x in (0., .25, .5, .75, 1.)):
                        payload[f'Z_{t:.2f}'] = z
                        v, routes = self.gen.velocity(z, z.new_full((z.shape[0],), t), conditions, clinical, True)
                        velocities[f'V_{t:.2f}'] = v.float()
                        for block, rec in enumerate(routes):
                            routing.append(dict(t=t, block=block, level=rec['level'], weights=rec['weights'].float().cpu().tolist()))
            return out, payload, velocities, routing

    @torch.no_grad()
    def validate(self, loader):
        self.configure(False)
        scores, started = [], time.monotonic()
        for index, raw in enumerate(loader):
            b = self.move(raw)
            out, _, _, _ = self.generate(b['input'], b['clinical'], b['sample_id'])
            for i, sid in enumerate(b['sample_id']):
                x, y, mask = out[i:i+1], b['target'][i:i+1].float(), b['brain_mask'][i:i+1]
                mse = ((x-y).square()*mask).sum()/mask.sum().clamp_min(1)
                scores.append(dict(sample_id=sid, mae=float(masked_l1(x, y, mask)),
                    psnr=float(10*torch.log10(self.cfg['loss']['data_range']**2/mse.clamp_min(1e-12))),
                    ssim=float(ssim3d(x, y, self.cfg['loss']['data_range'], mask=mask)[0])))
            if index == 0 or (index+1) % 25 == 0 or index+1 == len(loader):
                print(json.dumps(dict(event='flow_validation', batch=index+1, batches=len(loader),
                    mean_mae=float(np.mean([s['mae'] for s in scores])), elapsed_seconds=round(time.monotonic()-started, 1))), flush=True)
        metrics = {k: float(np.mean([s[k] for s in scores])) for k in ('mae', 'psnr', 'ssim')}
        if not scores or not all(math.isfinite(x) for x in metrics.values()):
            raise FloatingPointError('Empty or non-finite Flow validation')
        return dict(selection_loss=metrics['mae'], samples=len(scores), generation=metrics), scores

    def checkpoint(self, epoch, best, stale, optimizer, scheduler):
        return dict(format_version='mlt-flow-clinical4-v1', stage='flow', epoch=epoch, best=best,
            stale_epochs=stale, config=self.cfg, fold=self.fold, metadata_hash=self.metadata_hash,
            generator=self.gen.state_dict(), clinical=self.clinical.state, provenance=self.provenance,
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=self.scaler.state_dict(),
            rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                     cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))

    def train(self, resume=False, max_samples=None, epochs_this_run=None):
        # Optional epoch-boundary yield for sequential single-GPU experiments.
        # The saved total budget and optimizer/scheduler/RNG state are unchanged.
        if epochs_this_run is not None and (not isinstance(epochs_this_run, int) or epochs_this_run < 1):
            raise ValueError('epochs_this_run must be a positive integer')
        tr = self.cfg['training']
        saved = None
        if resume:
            saved = self.load(self.run/'last.pt')
            if saved['config']['training'] != tr or saved['config'].get('runtime') != self.cfg.get('runtime'):
                raise ValueError('Resume requires unchanged training/runtime settings')
        elif (self.run/'last.pt').exists() or (self.run/'history.jsonl').exists():
            raise FileExistsError('Flow run exists: use --resume or a new output_dir')
        else:
            self.initialize(self.cfg['flow']['init_checkpoint'])
        parameters = self.configure()
        optimizer = torch.optim.AdamW(parameters, lr=tr['lr']['flow'], weight_decay=tr['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5,
            patience=tr.get('lr_patience', 5), threshold=tr.get('min_delta', 1e-4), threshold_mode='abs', min_lr=1e-6)
        start, best, stale, early_reference = 0, math.inf, 0, math.inf
        if saved:
            optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
            self.scaler.load_state_dict(saved['scaler'])
            start, best, stale = saved['epoch']+1, saved['best'], saved['stale_epochs']
            early_reference = saved.get('early_reference', best)
            random.setstate(saved['rng']['python']); np.random.set_state(saved['rng']['numpy'])
            torch.set_rng_state(saved['rng']['torch'])
            if self.device.type == 'cuda':
                torch.cuda.set_rng_state_all(saved['rng']['cuda'])
        (self.run/'config.json').write_text(json.dumps(self.cfg, indent=2), encoding='utf-8')
        (self.run/'clinical_transform.json').write_text(json.dumps(self.clinical.state, indent=2), encoding='utf-8')
        train, val = self.loader('flow', 'train', max_samples), self.loader('flow', 'val', max_samples)
        if tr['patience'] and stale >= tr['patience']:
            return self.run/'best.pt'
        consecutive = 0
        end_epoch = tr['epochs']['flow'] if epochs_this_run is None else min(tr['epochs']['flow'], start+epochs_this_run)
        for epoch in range(start, end_epoch):
            from common import check_stop
            check_stop()
            self.configure()
            started, total, n, updates, skipped = time.monotonic(), 0., 0, 0, 0
            parts_sum = dict(flow=0., endpoint=0., pet=0., path=0.)
            optimizer.zero_grad(set_to_none=True)
            print(json.dumps(dict(event='flow_epoch_start', epoch=epoch+1, batches=len(train), lr=optimizer.param_groups[0]['lr'])), flush=True)
            accumulation = tr['accumulation_steps']
            for index, raw in enumerate(train):
                b = self.move(raw)
                divisor = min(accumulation, len(train)-(index//accumulation)*accumulation)
                with self.autocast():
                    loss, parts = self.loss(b)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite Flow loss: {b["sample_id"]}')
                self.scaler.scale(loss/divisor).backward()
                if (index+1) % accumulation == 0 or index+1 == len(train):
                    if scaled_optimizer_step(parameters, optimizer, self.scaler, tr['grad_clip']):
                        updates += 1; consecutive = 0
                    else:
                        skipped += 1; consecutive += 1
                    if consecutive >= 8:
                        raise FloatingPointError('Eight consecutive Flow AMP overflows')
                count = len(b['sample_id'])
                total += float(loss.detach())*count; n += count
                for key in parts_sum:
                    parts_sum[key] += parts[key]*count
                if index == 0 or (index+1) % tr.get('log_every', 10) == 0 or index+1 == len(train):
                    print(json.dumps(dict(event='flow_train', epoch=epoch+1, batch=index+1, batches=len(train),
                        loss=total/n, components={k: v/n for k, v in parts_sum.items()},
                        elapsed_seconds=round(time.monotonic()-started, 1))), flush=True)
            if not updates:
                raise FloatingPointError('No successful Flow optimizer updates')
            metrics, _ = self.validate(val)
            score = metrics['selection_loss']
            improved = score < best
            if improved:
                best = score
            if score < early_reference-tr.get('min_delta', 1e-4):
                early_reference, stale = score, 0
            else:
                stale += 1
            scheduler.step(score)
            state = self.checkpoint(epoch, best, stale, optimizer, scheduler)
            state['early_reference'] = early_reference
            for name in (['last', 'best'] if improved else ['last']):
                torch.save(state, self.run/f'{name}.tmp')
                (self.run/f'{name}.tmp').replace(self.run/f'{name}.pt')
            report = dict(stage=getattr(self, 'stage_name', 'flow'), epoch=epoch+1, train_loss=total/n, validation=metrics,
                components={k: v/n for k, v in parts_sum.items()}, optimizer_updates=updates,
                amp_skipped_updates=skipped, next_lr=optimizer.param_groups[0]['lr'], stale_epochs=stale,
                elapsed_seconds=round(time.monotonic()-started, 1))
            with (self.run/'history.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(safe_json(report))+'\n')
            print(json.dumps(report), flush=True)
            if tr['patience'] and stale >= tr['patience']:
                break
        return self.run/'best.pt'
