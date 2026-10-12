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
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import ClinicalTransform, TrainingDataset, select_rows
from .diffusion import Diffusion
from .losses import masked_l1, reconstruction, ssim3d
from .models import Generator, LatentDiscriminator

STAGES = ["mri_aae", "pet_ae", "coarse", "diffusion"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def fixed_noise(z, ids, seed):
    result = []
    for sid in ids:
        value = int(hashlib.sha256(f"{seed}:{sid}".encode()).hexdigest()[:15], 16)
        generator = torch.Generator(device=z.device).manual_seed(value)
        result.append(torch.randn(z.shape[1:], generator=generator, device=z.device))
    return torch.stack(result)


def safe_json(obj):
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    if isinstance(obj, (float, np.floating)) and not math.isfinite(obj):
        return None
    return obj


def stage_scheduler(optimizer, epochs, settings=None):
    settings = settings or {"type": "cosine"}
    if settings["type"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if settings["type"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
            factor=settings.get("factor", .5), patience=settings.get("patience", 5),
            threshold=settings.get("threshold", 1e-4), threshold_mode="abs",
            min_lr=settings.get("min_lr", 1e-6))
    raise ValueError("Unknown stage learning-rate scheduler")


def scaled_optimizer_step(parameters, optimizer, scaler, max_norm):
    """Let GradScaler skip an overflow before gradient clipping can raise.

    Full-precision non-finite gradients and finite-gradient norm overflow remain
    hard errors. Never clip NaN/Inf gradients or pass them to optimizer.step().
    """
    parameters = list(parameters)
    scaler.unscale_(optimizer)
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        raise RuntimeError("Optimizer has no gradients")
    finite = bool(torch.stack([torch.isfinite(g).all() for g in grads]).all())
    if not finite and not scaler.is_enabled():
        raise FloatingPointError("Non-finite full-precision gradients")
    if finite:
        nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    # unscale_ already recorded found_inf. On overflow, step skips the update
    # and update lowers the scale for the next accumulated gradient group.
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return finite


class Trainer:
    def __init__(self, cfg, fold, device=None, inference_only=False):
        self.cfg, self.fold = cfg, fold
        # Performance-only options do not alter the checkpoint's model layout,
        # optimization schedule, batch size, or preprocessing values.
        runtime = cfg.get("runtime", {})
        if "cpu_threads" in runtime:
            threads = runtime["cpu_threads"]
            if not isinstance(threads, int) or threads < 1:
                raise ValueError("runtime.cpu_threads must be a positive integer")
            torch.set_num_threads(threads)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.amp = bool(cfg["training"]["amp"] and self.device.type == "cuda")
        precision = runtime.get('precision', 'fp16')
        if precision not in ('fp16', 'bf16', 'fp32'):
            raise ValueError('Unsupported precision')
        self.amp = self.amp and precision != 'fp32'
        self.dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
        if self.amp and precision == 'bf16' and not torch.cuda.is_bf16_supported():
            raise ValueError('BF16 unsupported on this GPU')
        seed_all(cfg["seed"])
        self.run = Path(cfg["output_dir"])/f"fold_{fold}"
        self.run.mkdir(parents=True, exist_ok=True)
        self.metadata_hash, self.metadata, self.clinical = None, None, None
        if not inference_only:
            prepared = Path(cfg["data"]["prepared_dir"])/"metadata.csv"
            self.metadata_hash = hashlib.sha256(prepared.read_bytes()).hexdigest()
            self.metadata = pd.read_csv(prepared)
            train = select_rows(self.metadata, "train", fold)
            val = select_rows(self.metadata, "val", fold)
            if train.empty or val.empty or set(train.subject_id) & set(val.subject_id):
                raise ValueError("Empty split or subject leakage")
            self.clinical = ClinicalTransform.fit(train)
        self.gen = Generator(cfg["model"]).to(self.device)
        if "activation_checkpointing" in runtime:
            enabled = runtime["activation_checkpointing"]
            if not isinstance(enabled, bool):
                raise ValueError("runtime.activation_checkpointing must be boolean")
            for module in self.gen.modules():
                if hasattr(module, "checkpointing"):
                    module.checkpointing = enabled
        self.diff = Diffusion(**cfg["diffusion"]).to(self.device)
        self.disc = LatentDiscriminator(cfg["model"]["latent_channels"]).to(self.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp and self.dtype == torch.float16)
        if runtime:
            print(json.dumps(dict(event="runtime_settings", runtime=runtime,
                                  cpu_threads=torch.get_num_threads())), flush=True)

    def loader(self, stage, split, max_samples=None):
        rows = select_rows(self.metadata, split, self.fold)
        if max_samples:
            rows = rows.head(max_samples)
        if rows.empty:
            raise ValueError(f"No eligible samples for {stage}/{split}")
        ds = TrainingDataset(rows, self.cfg["data"], self.clinical)
        return DataLoader(ds, batch_size=self.cfg["training"]["batch_size"], shuffle=split == "train",
                          num_workers=self.cfg["training"]["num_workers"], pin_memory=self.device.type == "cuda",
                          worker_init_fn=seed_worker)

    def configure_stage(self, stage, training=True):
        modules = [self.gen, self.disc]
        for module in modules:
            module.eval()
            module.requires_grad_(False)
        active = []
        if stage == "mri_aae":
            active = [self.gen.mri]
            self.disc.requires_grad_(training).train(training)
        elif stage == "pet_ae":
            active = [self.gen.pet]
        elif stage == "coarse":
            active = [self.gen.coarse]
        elif stage == "diffusion":
            active = [self.gen.denoiser]
        else:
            raise ValueError(stage)
        for module in active:
            module.requires_grad_(training).train(training)
        return [p for m in active for p in m.parameters() if p.requires_grad]

    def checkpoint(self, stage, epoch, best, optimizer, scheduler, disc_optimizer):
        return dict(format_version="generation-clinical4-v1", stage=stage, epoch=epoch, best=best, config=self.cfg, fold=self.fold,
                    metadata_hash=self.metadata_hash, generator=self.gen.state_dict(),
                    diffusion=self.diff.state_dict(), residual_scale=self.diff.residual_scale,
                    discriminator=self.disc.state_dict(), clinical=self.clinical.state,
                    optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=self.scaler.state_dict(),
                    disc_optimizer=disc_optimizer.state_dict() if disc_optimizer else None,
                    rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                             cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))

    def load(self, path):
        # Only load checkpoints created by this project and trusted by the user.
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("format_version") != "generation-clinical4-v1":
            raise ValueError("Checkpoint is incompatible with this four-feature generation model")
        if state["fold"] != self.fold or (self.metadata_hash is not None and state["metadata_hash"] != self.metadata_hash):
            raise ValueError("Checkpoint fold or metadata differs from this run")
        for key in ["model", "diffusion"]:
            if state["config"][key] != self.cfg[key]:
                raise ValueError(f"Checkpoint {key} configuration mismatch")
        if state["config"]["data"]["spatial_size"] != self.cfg["data"]["spatial_size"]:
            raise ValueError("Checkpoint image geometry mismatch")
        self.gen.load_state_dict(state["generator"])
        self.diff.load_state_dict(state["diffusion"])
        self.diff.residual_scale = state["residual_scale"]
        self.disc.load_state_dict(state["discriminator"])
        self.clinical = ClinicalTransform(state["clinical"])
        return state

    def move(self, batch):
        return {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

    def loss(self, stage, batch):
        mri, pet, mask = batch["input"], batch["target"], batch["brain_mask"]
        loss_cfg = self.cfg["loss"]
        if stage in ["mri_aae", "pet_ae"]:
            ae = self.gen.mri if stage == "mri_aae" else self.gen.pet
            target = mri if stage == "mri_aae" else pet
            out, z = ae(target)
            ae_loss_cfg = dict(loss_cfg, data_range=loss_cfg.get("mri_data_range", 2.)) if stage == "mri_aae" else loss_cfg
            value = reconstruction(out, target, mask, ae_loss_cfg)
            if stage == "mri_aae":
                value = value+loss_cfg["aae"]*F.softplus(-self.disc(z)).mean()
            return value
        zs, za, zc = self.gen.source(mri)
        with torch.no_grad():
            zp = self.gen.pet.encode(pet)
        coarse_loss = loss_cfg["latent"]*F.l1_loss(zc, zp)
        if stage == "coarse":
            return coarse_loss+reconstruction(self.gen.pet.decode(zc), pet, mask, loss_cfg)
        r0 = (zp-zc.detach())/self.diff.residual_scale
        t = torch.randint(self.diff.steps, (mri.shape[0],), device=self.device)
        noise = torch.randn_like(r0)
        rt = self.diff.q_sample(r0, t, noise)
        pred = self.gen.denoiser(rt, t, self.gen.denoiser.conditions(zs, za, zc), batch["clinical"])
        value = F.mse_loss(pred.float(), self.diff.target(r0, t, noise).detach().float())
        # For epsilon prediction, x0 reconstruction near pure noise divides by tiny alpha.
        # Auxiliary image/latent supervision is restricted to identifiable noise levels.
        valid = self.diff.alpha_bar[t] >= loss_cfg["aux_min_alpha"]
        if valid.any():
            estimate, _ = self.diff.recover(rt[valid].float(), t[valid], pred[valid].float())
            zf = zc[valid]+self.diff.residual_scale*estimate
            value = value+loss_cfg["latent"]*F.l1_loss(zf, zp[valid])
            value = value+loss_cfg["cosine"]*(1-F.cosine_similarity(zf, zp[valid], dim=1)).mean()
            value = value+reconstruction(self.gen.pet.decode(zf), pet[valid], mask[valid], loss_cfg)
        return value

    @torch.no_grad()
    def validate(self, stage, loader):
        self.gen.eval()
        self.disc.eval()
        total, n, image_scores = 0., 0, []
        validation_started = time.monotonic()
        for batch_index, raw in enumerate(loader):
            b = self.move(raw)
            with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.amp):
                target = b["input"] if stage == "mri_aae" else b["target"]
                if stage in ["mri_aae", "pet_ae"]:
                    out = (self.gen.mri if stage == "mri_aae" else self.gen.pet)(target)[0]
                else:
                    zs, za, zc = self.gen.source(b["input"])
                    zf = zc
                    if stage == "diffusion":
                        zf = self.diff.sample(self.gen.denoiser, (zs, za, zc), self.cfg["sampling"]["steps"],
                                              fixed_noise(zc, b["sample_id"], self.cfg["seed"]), clinical=b["clinical"],
                                              timestep_spacing=self.cfg["sampling"].get("timestep_spacing", "leading"))
                    out = self.gen.pet.decode(zf)
                if not torch.isfinite(out).all():
                    raise FloatingPointError(f"Non-finite {stage} validation output at {b['sample_id']}")
                value = masked_l1(out, target, b["brain_mask"])
                data_range = self.cfg["loss"].get("mri_data_range", 2.) if stage == "mri_aae" else self.cfg["loss"]["data_range"]
                for i in range(len(b["sample_id"])):
                    x, y, mask = out[i:i+1].float(), target[i:i+1], b["brain_mask"][i:i+1]
                    mse = ((x-y).square()*mask).sum()/mask.sum().clamp_min(1)
                    image_scores.append(dict(sample_id=b["sample_id"][i],
                                             mae=float(masked_l1(x, y, mask)),
                                             psnr=float(10*torch.log10(data_range**2/mse.clamp_min(1e-12))),
                                             ssim=float(ssim3d(x, y, data_range, mask=mask)[0])))
            total += float(value)*len(b["sample_id"])
            n += len(b["sample_id"])
            if stage == "diffusion" and (batch_index == 0 or (batch_index+1) % 25 == 0 or batch_index+1 == len(loader)):
                print(json.dumps(dict(event="validation_progress", stage=stage, batch=batch_index+1,
                                      batches=len(loader), samples=n, mean_mae=total/n,
                                      elapsed_seconds=round(time.monotonic()-validation_started, 1))), flush=True)
        metrics = dict(selection_loss=total/n, samples=n,
                       generation={k: float(np.mean([x[k] for x in image_scores])) for k in ["mae", "psnr", "ssim"]})
        return metrics, image_scores

    def train(self, stage, resume=False, max_samples=None):
        directory = self.run/stage
        directory.mkdir(parents=True, exist_ok=True)
        saved = None
        if resume:
            saved = self.load(directory/"last.pt")
            if saved["stage"] != stage:
                raise ValueError("Resume stage mismatch")
            if saved["config"]["training"] != self.cfg["training"]:
                raise ValueError("Resume requires identical training settings, including epoch counts")
        else:
            if (directory/"last.pt").exists():
                raise FileExistsError("Run already exists; use --resume or a new output_dir")
            index = STAGES.index(stage)
            if index:
                prerequisite = self.run/STAGES[index-1]/"best.pt"
                if not prerequisite.exists():
                    raise FileNotFoundError(f"Train prerequisite first: {prerequisite}")
                self.load(prerequisite)
        parameters = self.configure_stage(stage)
        optimizer = torch.optim.AdamW(parameters, lr=self.cfg["training"]["lr"][stage],
                                      weight_decay=self.cfg["training"]["weight_decay"])
        epochs = self.cfg["training"]["epochs"][stage]
        scheduler = stage_scheduler(optimizer, epochs, self.cfg["training"].get("schedulers", {}).get(stage))
        disc_optimizer = torch.optim.AdamW(self.disc.parameters(), lr=self.cfg["training"]["lr"][stage]) if stage == "mri_aae" else None
        start, best = 0, math.inf
        if saved:
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            self.scaler.load_state_dict(saved["scaler"])
            if disc_optimizer:
                disc_optimizer.load_state_dict(saved["disc_optimizer"])
            start, best = saved["epoch"]+1, saved["best"]
            random.setstate(saved["rng"]["python"])
            np.random.set_state(saved["rng"]["numpy"])
            torch.set_rng_state(saved["rng"]["torch"])
            if self.device.type == "cuda" and saved["rng"]["cuda"]:
                torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
        train, val = self.loader(stage, "train", max_samples), self.loader(stage, "val", max_samples)
        (directory/"config.json").write_text(json.dumps(self.cfg, indent=2), encoding="utf-8")
        (directory/"clinical_transform.json").write_text(json.dumps(self.clinical.state, indent=2), encoding="utf-8")
        accumulation = self.cfg["training"]["accumulation_steps"]
        stale = saved.get("stale_epochs", 0) if saved else 0
        consecutive_overflows = 0
        if self.cfg["training"]["patience"] and stale >= self.cfg["training"]["patience"]:
            return directory/"best.pt"
        for epoch in range(start, epochs):
            from common import check_stop
            check_stop()
            self.configure_stage(stage)
            epoch_started = time.monotonic()
            print(json.dumps(dict(event="epoch_start", stage=stage, epoch=epoch+1,
                                  epochs=epochs, train_batches=len(train), validation_batches=len(val))), flush=True)
            optimizer.zero_grad(set_to_none=True)
            total, count = 0., 0
            updates, skipped_updates = 0, 0
            for index, raw in enumerate(train):
                b = self.move(raw)
                if disc_optimizer:
                    self.disc.requires_grad_(True)
                    disc_optimizer.zero_grad(set_to_none=True)
                    with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.amp):
                        latent = self.gen.mri.encode(b["input"])
                    # Separate full-precision D update: never change the G scaler
                    # in the middle of an accumulated generator gradient group.
                    latent = latent.float()
                    dl = F.softplus(-self.disc(torch.randn_like(latent))).mean()+F.softplus(self.disc(latent)).mean()
                    dl.backward()
                    nn.utils.clip_grad_norm_(self.disc.parameters(), self.cfg["training"]["grad_clip"], error_if_nonfinite=True)
                    disc_optimizer.step()
                    self.disc.requires_grad_(False)
                group_start = (index//accumulation)*accumulation
                divisor = min(accumulation, len(train)-group_start)
                with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.amp):
                    loss = self.loss(stage, b)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite {stage} loss at {b['sample_id']}")
                self.scaler.scale(loss/divisor).backward()
                if (index+1) % accumulation == 0 or index+1 == len(train):
                    old_scale = self.scaler.get_scale()
                    updated = scaled_optimizer_step(parameters, optimizer, self.scaler,
                                                    self.cfg["training"]["grad_clip"])
                    if updated:
                        updates += 1
                        consecutive_overflows = 0
                    else:
                        skipped_updates += 1
                        consecutive_overflows += 1
                        print(json.dumps(dict(event="amp_overflow", stage=stage, epoch=epoch+1,
                                              batch=index+1, sample_ids=list(b["sample_id"]),
                                              scale_before=old_scale, scale_after=self.scaler.get_scale(),
                                              consecutive_overflows=consecutive_overflows)), flush=True)
                        if consecutive_overflows >= 8:
                            raise FloatingPointError("Eight consecutive AMP overflows; inspect model/data stability")
                total += float(loss.detach())*len(b["sample_id"])
                count += len(b["sample_id"])
                if index == 0 or (index+1) % self.cfg["training"].get("log_every", 10) == 0 or index+1 == len(train):
                    print(json.dumps(dict(event="train_progress", stage=stage, epoch=epoch+1,
                                          batch=index+1, batches=len(train), loss=float(loss.detach()),
                                          mean_loss=total/count, elapsed_seconds=round(time.monotonic()-epoch_started, 1))), flush=True)
            if not updates:
                raise FloatingPointError("Epoch had no successful optimizer updates")
            metrics, _ = self.validate(stage, val)
            if not math.isfinite(metrics["selection_loss"]):
                raise FloatingPointError(f"Non-finite {stage} validation loss; inspect sampling stability and intensity scales")
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metrics["selection_loss"])
            else:
                scheduler.step()
            improved = metrics["selection_loss"] < best
            if improved:
                best, stale = metrics["selection_loss"], 0
            else:
                stale += 1
            state = self.checkpoint(stage, epoch, best, optimizer, scheduler, disc_optimizer)
            state["stale_epochs"] = stale
            temporary = directory/"last.tmp"
            torch.save(state, temporary)
            temporary.replace(directory/"last.pt")
            if improved:
                torch.save(state, directory/"best.tmp")
                (directory/"best.tmp").replace(directory/"best.pt")
            report = dict(stage=stage, epoch=epoch+1, train_loss=total/count, validation=metrics,
                          optimizer_updates=updates, amp_skipped_updates=skipped_updates,
                          amp_scale=self.scaler.get_scale())
            with (directory/"history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(safe_json(report), ensure_ascii=False)+"\n")
            print(json.dumps(safe_json(report)), flush=True)
            if self.cfg["training"]["patience"] and stale >= self.cfg["training"]["patience"]:
                break
        return directory/"best.pt"
