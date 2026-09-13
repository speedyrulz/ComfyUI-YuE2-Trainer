"""Acoustic (NAR flow-matching) LoRA training on ComfyUI's YuE2 MODEL."""
from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn

from .constants import FRAMES_PER_SECOND, MODEL_KEY_PREFIX, CONTEXT
from .dataset import Dataset, Item
from .forward import nar_forward
from .lora import create_lora, select_target_modules, count_parameters
from .monitor import TrainMonitor
from .parallel import (Replica, clone_patcher_for_device, free_replicas, reduce_gradients, resolve_devices,
                       run_on_replicas, split_counts, sync_lora_weights)
from .prefix import PrefixCache, build_acoustic_prefix, load_clip_for_prefill, music_prefix_ids, resolve_mode


@dataclass
class AcousticConfig:
    steps: int = 200
    batch_size: int = 1                 # items per optimizer step (accumulated one at a time)
    grad_accumulation: int = 1
    learning_rate: float = 1e-4
    rank: int = 16
    alpha: float = 16.0
    targets: str = "attention+mlp"
    train_acoustic_head: bool = False   # also adapt vae2llm / llm2vae / time embedder
    segment_seconds: float = 30.0       # random crop length (0 = whole song / chunk)
    mode: str = "auto"                  # auto|full|melody|off  (CoT instruction used for the prefix)
    use_semantic_tokens: bool = True    # when an item carries semantic tokens, condition on them
    timestep_sampling: str = "uniform"  # uniform | logit_normal
    shift: float = 1.0                  # sigma = shift*u / (1 + (shift-1)*u)
    logit_mean: float = 0.0
    logit_std: float = 1.0
    weight_decay: float = 0.01
    warmup_steps: int = 10
    max_grad_norm: float = 1.0
    seed: int = 0
    lora_dtype: str = "fp32"
    gradient_checkpointing: bool = True
    optimizer: str = "AdamW"
    devices: str = "auto"               # auto | cuda:N | all | cuda:0,cuda:1
    log_every: int = 1                  # console line every N steps
    tensorboard_dir: str = ""           # "" = off; parent folder for TensorBoard runs
    run_name: str = ""                  # TensorBoard run name (timestamp appended)
    existing_lora: Optional[dict] = None
    save_every: int = 0
    save_callback: Optional[Callable[[dict, int], None]] = None


@dataclass
class TrainResult:
    lora_sd: dict
    losses: list = field(default_factory=list)
    steps: int = 0
    seconds: float = 0.0
    info: dict = field(default_factory=dict)


def _sample_sigma(cfg: AcousticConfig, rng: torch.Generator) -> float:
    if cfg.timestep_sampling == "logit_normal":
        u = torch.sigmoid(torch.randn((), generator=rng) * cfg.logit_std + cfg.logit_mean).item()
    else:
        u = torch.rand((), generator=rng).item()
    u = min(max(u, 1e-4), 1.0 - 1e-4)
    if cfg.shift != 1.0:
        u = cfg.shift * u / (1.0 + (cfg.shift - 1.0) * u)
    return float(u)


def _make_optimizer(name: str, params, lr: float, weight_decay: float):
    if name == "AdamW":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "SGD":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "Adafactor":
        try:
            from transformers.optimization import Adafactor
            return Adafactor(params, lr=lr, scale_parameter=False, relative_step=False, warmup_init=False)
        except ImportError:  # pragma: no cover
            logging.warning("transformers not available, falling back to AdamW")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _lr_at(step: int, total: int, warmup: int, base: float) -> float:
    if warmup > 0 and step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))


def _chunk_bounds(item: Item, prefix_len: int) -> list[tuple[int, int]]:
    """Original inference chunking (comfy.text_encoders.yue2.chunk_ranges)."""
    frames = item.frames
    size = (CONTEXT - prefix_len - 3) // 2
    return [(s, min(s + size, frames)) for s in range(0, frames, size)]


@dataclass
class _Sample:
    item: Item
    prefix: PrefixCache
    chunk: tuple[int, int]


def prepare_samples(clip, dataset: Dataset, cfg: AcousticConfig, progress=None) -> list[_Sample]:
    items = dataset.with_latents()
    if not items:
        raise ValueError("Dataset has no encoded latents; run the Encode Dataset node first")
    device = load_clip_for_prefill(clip)
    samples = []
    for index, item in enumerate(items):
        cot = resolve_mode(cfg.mode, item.abc, item.has_chords())
        abc = item.abc if cot != "off" else None
        semantic = item.semantic if (cfg.use_semantic_tokens and item.semantic) else None
        if semantic is not None and abs(len(semantic) - item.frames) > 2:
            logging.warning("YuE2 trainer: %s has %d semantic tokens but %d latent frames; using text-only conditioning",
                            item.id, len(semantic), item.frames)
            semantic = None
        if semantic is not None:
            n = min(len(semantic), item.frames)
            prefix_ids, _ = music_prefix_ids(clip, item.style, item.lyrics, abc, cot)
            for chunk in _chunk_bounds(item, len(prefix_ids)):
                chunk = (chunk[0], min(chunk[1], n))
                if chunk[1] <= chunk[0]:
                    continue
                prefix = build_acoustic_prefix(clip, item.style, item.lyrics, abc, cot, semantic=semantic[:n],
                                               chunk=chunk, device=device)
                samples.append(_Sample(item, prefix, chunk))
        else:
            prefix = build_acoustic_prefix(clip, item.style, item.lyrics, abc, cot, total_frames=item.frames, device=device)
            samples.append(_Sample(item, prefix, (0, item.frames)))
        if progress is not None:
            progress(index + 1, len(items))
    return samples


def _load_patchers(model_patcher, devices: list[torch.device]):
    """One ModelPatcher per device, loaded fully through ComfyUI's model management.

    Replicas beyond the first are deep copies made while the weights sit on the offload
    device, so each GPU ends up with its own complete copy of the frozen base model.
    """
    import comfy.model_management
    patchers = []
    for index, device in enumerate(devices):
        mp = clone_patcher_for_device(model_patcher, device, fresh=index > 0)
        mp.set_model_compute_dtype(torch.bfloat16)
        patchers.append(mp)
    for mp in patchers:
        comfy.model_management.load_models_gpu([mp], memory_required=1e20, force_full_load=True)
        mp.model.diffusion_model.requires_grad_(False)
    return patchers


def _setup_replicas(patchers, devices: list[torch.device], cfg: AcousticConfig, lora_dtype) -> list[Replica]:
    replicas = []
    for index, (mp, device) in enumerate(zip(patchers, devices)):
        root = mp.model
        targets = select_target_modules(root, cfg.targets, include_acoustic_extra=cfg.train_acoustic_head,
                                        name_prefix=MODEL_KEY_PREFIX)
        lora = create_lora(root, targets, cfg.rank, cfg.alpha, lora_dtype=lora_dtype,
                           existing=cfg.existing_lora, save_prefix="")
        lora.inject(mp, root)
        for adapter in lora.adapters:  # bypass hooks move adapters to ComfyUI's default device; pin them to ours
            adapter.to(device)
        gen = torch.Generator().manual_seed(cfg.seed + 7919 * index)
        replicas.append(Replica(index=index, device=device, root=root, lora=lora, generator=gen,
                                extra={"py_rng": random.Random(cfg.seed + 104729 * index), "patcher": mp}))
    sync_lora_weights(replicas)
    return replicas


def train_acoustic_lora(model_patcher, clip, dataset: Dataset, cfg: AcousticConfig,
                        progress: Optional[Callable[[int, int, float], None]] = None,
                        interrupt_check: Optional[Callable[[], None]] = None) -> TrainResult:
    import comfy.model_management

    start_time = time.perf_counter()
    torch.manual_seed(cfg.seed)
    devices = resolve_devices(cfg.devices)

    # 1. AR prefix caches via CLIP (then free it to make room for the acoustic model).
    samples = prepare_samples(clip, dataset, cfg)
    logging.info("YuE2 trainer: %d training chunks from %d items", len(samples), len(dataset.with_latents()))
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()

    # 2. Load the acoustic model (one replica per device) and attach LoRA.
    patchers = _load_patchers(model_patcher, devices)
    lora_dtype = torch.float32 if cfg.lora_dtype == "fp32" else torch.bfloat16
    comfy.model_management.in_training = True
    replicas = _setup_replicas(patchers, devices, cfg, lora_dtype)
    primary = replicas[0]
    optimizer = _make_optimizer(cfg.optimizer, primary.lora.trainable, cfg.learning_rate, cfg.weight_decay)
    logging.info("YuE2 trainer: %.2fM trainable LoRA parameters on %s", count_parameters(primary.lora.trainable) / 1e6,
                 ", ".join(str(d) for d in devices))

    micro_steps = cfg.batch_size * cfg.grad_accumulation
    if micro_steps < len(devices):
        logging.warning("YuE2 trainer: batch_size x grad_accumulation (%d) is smaller than the number of GPUs (%d); "
                        "raise it to keep every GPU busy", micro_steps, len(devices))
    counts = split_counts(micro_steps, len(replicas))
    seg_frames = int(round(cfg.segment_seconds * FRAMES_PER_SECOND)) if cfg.segment_seconds > 0 else 0
    losses = []
    monitor = TrainMonitor("acoustic", cfg.steps, cfg.log_every, cfg.tensorboard_dir or None, cfg.run_name,
                           config={k: v for k, v in vars(cfg).items() if k not in ("existing_lora", "save_callback")})

    def work(replica: Replica, n_micro: int) -> torch.Tensor:
        device, dm = replica.device, replica.root.diffusion_model
        rng, py_rng = replica.generator, replica.extra["py_rng"]
        total = torch.zeros((), device=device, dtype=torch.float32)
        for _ in range(n_micro):
            sample = py_rng.choice(samples)
            c0, c1 = sample.chunk
            latents = sample.item.latents[:, c0:c1]
            T = latents.shape[-1]
            if seg_frames and T > seg_frames:
                offset = py_rng.randint(0, T - seg_frames)
                latents = latents[:, offset: offset + seg_frames]
            else:
                offset = 0
            x0 = latents.to(device=device, dtype=torch.float32)[None].clone()
            noise = torch.randn(x0.shape, generator=rng, dtype=torch.float32).to(device)
            sigma = torch.tensor([_sample_sigma(cfg, rng)], device=device)
            x_t = (1.0 - sigma.view(-1, 1, 1)) * x0 + sigma.view(-1, 1, 1) * noise
            target = noise - x0
            prefix_kv = sample.prefix.kv.to(device, non_blocking=True).clone()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred = nar_forward(dm, x_t.to(torch.bfloat16), sigma, prefix_kv, sample.prefix.ar_length,
                                   frame_offset=offset, checkpointing=cfg.gradient_checkpointing)
            loss = torch.nn.functional.mse_loss(pred.float(), target)
            (loss / micro_steps).backward()
            total += loss.detach()
            del pred, loss, prefix_kv, x_t, x0, noise
        return total

    try:
        for step in range(cfg.steps):
            if interrupt_check is not None:
                interrupt_check()
            for group in optimizer.param_groups:
                group["lr"] = _lr_at(step, cfg.steps, cfg.warmup_steps, cfg.learning_rate)
            optimizer.zero_grad(set_to_none=True)
            loss_sum = run_on_replicas(replicas, work, counts)
            reduce_gradients(replicas)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(primary.lora.trainable, cfg.max_grad_norm or float("inf")))
            optimizer.step()
            sync_lora_weights(replicas)
            step_loss = loss_sum / micro_steps
            losses.append(step_loss)
            monitor.step(step + 1, step_loss, optimizer.param_groups[0]["lr"], grad_norm)
            if progress is not None:
                progress(step + 1, cfg.steps, step_loss)
            if cfg.save_every and cfg.save_callback and (step + 1) % cfg.save_every == 0 and step + 1 < cfg.steps:
                cfg.save_callback(primary.lora.export(), step + 1)
    finally:
        comfy.model_management.in_training = False
        monitor.close()
        for replica in replicas:
            replica.lora.eject(replica.extra["patcher"])
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        free_replicas(replicas)
        if len(patchers) > 1:  # drop the extra full-model copies from VRAM
            del patchers[1:]
            comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

    exported = primary.lora.export()
    for adapter in primary.lora.adapters:
        adapter.requires_grad_(False)
    info = {"kind": "acoustic", "rank": cfg.rank, "alpha": cfg.alpha, "targets": cfg.targets,
            "train_acoustic_head": cfg.train_acoustic_head, "steps": cfg.steps, "items": len(dataset.with_latents()),
            "chunks": len(samples), "segment_seconds": cfg.segment_seconds, "timestep_sampling": cfg.timestep_sampling,
            "shift": cfg.shift, "learning_rate": cfg.learning_rate, "mode": cfg.mode,
            "devices": [str(d) for d in devices], "micro_steps": micro_steps,
            "tensorboard": str(monitor.log_dir) if monitor.log_dir else None,
            "semantic_conditioned_chunks": sum(1 for s in samples if s.prefix.ar_length == len(s.prefix.ids))}
    return TrainResult(lora_sd=exported, losses=losses, steps=cfg.steps,
                       seconds=time.perf_counter() - start_time, info=info)


__all__ = ["AcousticConfig", "TrainResult", "train_acoustic_lora", "prepare_samples"]
