"""Planner / semantic (AR next-token) LoRA training on ComfyUI's YuE2 CLIP (the language model)."""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from .acoustic import TrainResult, _check_trainable_weights, _make_optimizer, _lr_at
from .constants import CLIP_KEY_PREFIX, CODEC_OFFSET, CONTEXT, MUSIC_END
from .dataset import Dataset, Item
from .forward import ar_hidden, chunked_cross_entropy
from .lora import create_lora, select_target_modules, count_parameters
from .monitor import TrainMonitor
from .parallel import (Replica, clone_patcher_for_device, free_replicas, reduce_gradients, resolve_devices,
                       run_on_replicas, split_counts, sync_lora_weights)
from .prefix import abc_sequence, music_prefix_ids, resolve_mode, load_clip_for_prefill


@dataclass
class PlannerConfig:
    steps: int = 200
    batch_size: int = 1
    grad_accumulation: int = 1
    learning_rate: float = 1e-4
    rank: int = 16
    alpha: float = 16.0
    targets: str = "attention+mlp"
    train_abc: bool = True             # style+lyrics -> ABC score
    train_semantic: bool = False       # style+lyrics(+ABC) -> semantic codec tokens (items with tokens only)
    abc_mode: str = "full"             # full|melody|auto
    max_tokens: int = 4096             # crop of the trained span (ABC or codec tokens)
    weight_decay: float = 0.01
    lr_schedule: str = "cosine"         # cosine | constant | linear (warmup applies to all)
    warmup_steps: int = 10
    max_grad_norm: float = 1.0
    seed: int = 0
    lora_dtype: str = "fp32"
    gradient_checkpointing: bool = True
    optimizer: str = "AdamW"
    devices: str = "auto"              # auto | cuda:N | all | cuda:0,cuda:1
    log_every: int = 1                  # console line every N steps
    tensorboard_dir: str = ""           # "" = off; parent folder for TensorBoard runs
    run_name: str = ""                  # TensorBoard run name (timestamp appended)
    existing_lora: Optional[dict] = None
    save_every: int = 0
    save_callback: Optional[Callable[[dict, int], None]] = None


@dataclass
class _Sequence:
    item: Item
    kind: str
    ids: list
    loss_start: int


def build_sequences(clip, dataset: Dataset, cfg: PlannerConfig) -> list[_Sequence]:
    sequences = []
    for item in dataset.items:
        if cfg.train_abc and item.abc and item.abc.strip():
            cot = resolve_mode(cfg.abc_mode, item.abc, item.has_chords())
            ids, loss_start = abc_sequence(clip, item.style, item.lyrics, item.abc, cot)
            sequences.append(_Sequence(item, "abc", ids, loss_start))
        if cfg.train_semantic and item.semantic:
            cot = resolve_mode("auto", item.abc, item.has_chords())
            prefix, _ = music_prefix_ids(clip, item.style, item.lyrics, item.abc if cot != "off" else None, cot)
            ids = prefix + [int(t) + CODEC_OFFSET for t in item.semantic] + [MUSIC_END]
            sequences.append(_Sequence(item, "semantic", ids, len(prefix)))
    if not sequences:
        raise ValueError("No planner training sequences: items need an ABC score (train_abc) "
                         "or semantic tokens (train_semantic)")
    return sequences


def _crop(seq: _Sequence, max_tokens: int, rng: random.Random) -> tuple[list, int]:
    """Keep the prefix; crop the trained span to ``max_tokens`` at a random offset."""
    span = len(seq.ids) - seq.loss_start
    limit = min(max_tokens, CONTEXT - seq.loss_start)
    if span <= limit:
        return seq.ids, seq.loss_start
    start = rng.randint(0, span - limit)
    ids = seq.ids[:seq.loss_start] + seq.ids[seq.loss_start + start: seq.loss_start + start + limit]
    return ids, seq.loss_start


def _load_clips(clip, devices: list[torch.device]):
    """One CLIP (with its own text-encoder copy for replicas beyond the first) per device, loaded via ComfyUI."""
    import comfy.model_management
    clips = []
    for index, device in enumerate(devices):
        work = clip.clone()
        work.patcher = clone_patcher_for_device(clip.patcher, device, fresh=index > 0)
        work.cond_stage_model = work.patcher.model
        work.patcher.set_model_compute_dtype(torch.bfloat16)
        clips.append(work)
    for work, device in zip(clips, devices):
        comfy.model_management.load_models_gpu([work.patcher], force_full_load=True)
        work.cond_stage_model.set_clip_options({"execution_device": device})
        work.cond_stage_model.model.requires_grad_(False)
        _check_trainable_weights(work.cond_stage_model.model)
    return clips


def _setup_replicas(clips, devices: list[torch.device], cfg: PlannerConfig, lora_dtype) -> list[Replica]:
    replicas = []
    for index, (work, device) in enumerate(zip(clips, devices)):
        root = work.cond_stage_model
        targets = select_target_modules(root, cfg.targets, include_acoustic_extra=False, name_prefix="")
        lora = create_lora(root, targets, cfg.rank, cfg.alpha, lora_dtype=lora_dtype,
                           existing=cfg.existing_lora, save_prefix=CLIP_KEY_PREFIX)
        lora.inject(work.patcher, root)
        for adapter in lora.adapters:
            adapter.to(device)
        gen = torch.Generator().manual_seed(cfg.seed + 7919 * index)
        replicas.append(Replica(index=index, device=device, root=root, lora=lora, generator=gen,
                                extra={"py_rng": random.Random(cfg.seed + 104729 * index), "clip": work}))
    sync_lora_weights(replicas)
    return replicas


def train_planner_lora(clip, dataset: Dataset, cfg: PlannerConfig,
                       progress: Optional[Callable[[int, int, float], None]] = None,
                       interrupt_check: Optional[Callable[[], None]] = None) -> TrainResult:
    import comfy.model_management

    start_time = time.perf_counter()
    torch.manual_seed(cfg.seed)
    devices = resolve_devices(cfg.devices)
    sequences = build_sequences(clip, dataset, cfg)
    logging.info("YuE2 trainer: %d planner sequences (%d abc, %d semantic)", len(sequences),
                 sum(s.kind == "abc" for s in sequences), sum(s.kind == "semantic" for s in sequences))

    comfy.model_management.unload_all_models()
    clips = _load_clips(clip, devices)
    lora_dtype = torch.float32 if cfg.lora_dtype == "fp32" else torch.bfloat16
    comfy.model_management.in_training = True
    replicas = _setup_replicas(clips, devices, cfg, lora_dtype)
    primary = replicas[0]
    optimizer = _make_optimizer(cfg.optimizer, primary.lora.trainable, cfg.learning_rate, cfg.weight_decay)
    logging.info("YuE2 trainer: %.2fM trainable LoRA parameters on %s", count_parameters(primary.lora.trainable) / 1e6,
                 ", ".join(str(d) for d in devices))

    micro_steps = cfg.batch_size * cfg.grad_accumulation
    if micro_steps < len(devices):
        logging.warning("YuE2 trainer: batch_size x grad_accumulation (%d) is smaller than the number of GPUs (%d); "
                        "raise it to keep every GPU busy", micro_steps, len(devices))
    counts = split_counts(micro_steps, len(replicas))
    losses = []
    info = {"kind": "planner", "rank": cfg.rank, "alpha": cfg.alpha, "targets": cfg.targets, "steps": cfg.steps,
            "sequences": len(sequences), "train_abc": cfg.train_abc, "train_semantic": cfg.train_semantic,
            "max_tokens": cfg.max_tokens, "learning_rate": cfg.learning_rate, "lr_schedule": cfg.lr_schedule,
            "devices": [str(d) for d in devices], "micro_steps": micro_steps,
            }
    monitor = TrainMonitor("planner", cfg.steps, cfg.log_every, cfg.tensorboard_dir or None, cfg.run_name,
                           config={k: v for k, v in vars(cfg).items() if k not in ("existing_lora", "save_callback")})

    def work_fn(replica: Replica, n_micro: int) -> torch.Tensor:
        device, llm = replica.device, replica.root.model
        py_rng = replica.extra["py_rng"]
        total = torch.zeros((), device=device, dtype=torch.float32)
        for _ in range(n_micro):
            seq = py_rng.choice(sequences)
            ids, loss_start = _crop(seq, cfg.max_tokens, py_rng)
            tokens = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                hidden = ar_hidden(llm, tokens, torch.bfloat16, checkpointing=cfg.gradient_checkpointing)
                hidden = hidden[0, loss_start - 1: -1]
                targets_t = tokens[0, loss_start:]
                loss = chunked_cross_entropy(llm.lm_head, hidden, targets_t)
            (loss / micro_steps).backward()
            total += loss.detach()
            del hidden, loss, tokens
        return total

    try:
        for step in range(cfg.steps):
            if interrupt_check is not None:
                interrupt_check()
            for group in optimizer.param_groups:
                group["lr"] = _lr_at(step, cfg.steps, cfg.warmup_steps, cfg.learning_rate, cfg.lr_schedule)
            optimizer.zero_grad(set_to_none=True)
            loss_sum = run_on_replicas(replicas, work_fn, counts)
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
                cfg.save_callback(primary.lora.export(), step + 1,
                                  {**info, "steps": step + 1, "partial": True, "loss": step_loss})
    finally:
        comfy.model_management.in_training = False
        monitor.close()
        for replica in replicas:
            replica.lora.eject(replica.extra["clip"].patcher)
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        free_replicas(replicas)
        if len(clips) > 1:  # drop the extra full-model copies from VRAM
            del clips[1:]
            comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

    exported = primary.lora.export()
    for adapter in primary.lora.adapters:
        adapter.requires_grad_(False)
    info["tensorboard"] = str(monitor.log_dir) if monitor.log_dir else None
    return TrainResult(lora_sd=exported, losses=losses, steps=cfg.steps,
                       seconds=time.perf_counter() - start_time, info=info)


__all__ = ["PlannerConfig", "train_planner_lora", "build_sequences"]
