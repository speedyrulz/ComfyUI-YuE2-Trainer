"""Planner / semantic (AR next-token) LoRA training on ComfyUI's YuE2 CLIP (the language model)."""
from __future__ import annotations

import contextlib
import logging
import random
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from .acoustic import TrainResult, _check_trainable_weights, _make_optimizer, _lr_at, split_holdout
from .constants import CLIP_KEY_PREFIX, CODEC_OFFSET, CONTEXT, MUSIC_END
from .dataset import Dataset, Item
from .forward import ar_hidden, chunked_cross_entropy
from .lora import create_lora, select_target_modules, count_parameters
from .monitor import TrainMonitor
from .parallel import (Replica, clone_patcher_for_device, free_replicas, reduce_gradients, resolve_devices,
                       run_on_replicas, split_counts, sync_lora_weights)
from .prefix import abc_sequence, music_prefix_ids, resolve_mode, load_clip_for_prefill
from .resume import capture_state, restore_state

PROBE_SAMPLING = {"temperature": 0.7, "top_p": 0.9, "top_k": 30, "repetition_penalty": 1.005}   # YuE2GenerateABC defaults


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
    eval_every: int = 50                # fixed-crop validation loss every N steps (0 = off)
    eval_samples: int = 8               # size of the fixed evaluation set
    eval_holdout: int = 1               # songs kept out of training and used for the evaluation set (0 = score training crops)
    regularization_fraction: float = 0.5   # share of micro-steps drawn from the regularization scores (when given)
    probe_every: int = 0                # generate a score with the current LoRA every N steps (0 = off)
    probe_style: str = ""               # blank = style prompt of the first training item
    probe_lyrics: str = ""              # blank = lyrics of the first training item
    probe_mode: str = ""                # blank = abc_mode (full when abc_mode is auto)
    probe_max_tokens: int = 8192        # token budget of a probe; a probe that uses it all did not end its score
    probe_seed: int = 0
    probe_callback: Optional[Callable[[int, str, dict], Optional[str]]] = None   # (step, abc, meta) -> saved path
    tensorboard_dir: str = ""           # "" = off; parent folder for TensorBoard runs
    run_name: str = ""                  # TensorBoard run name (timestamp appended)
    existing_lora: Optional[dict] = None
    resume_state: Optional[dict] = None   # optimizer / RNG / step state saved next to existing_lora
    save_every: int = 0
    save_callback: Optional[Callable] = None   # (lora_sd, step, info, state)


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


CROP_CONTEXT = 256   # unsupervised context tokens at the start of a window that does not begin at the score's start


def _crop(seq: _Sequence, max_tokens: int, rng: random.Random) -> tuple[list, int]:
    """Keep the prefix; crop the trained span to ``max_tokens`` without teaching false starts or endless scores.

    A span longer than the limit is trained through one of three windows per draw: its head (so the
    opening of a score is learned), its tail (so the closing token is learned and the model keeps ending
    its scores), or a random middle window. Windows that do not start at the beginning keep their first
    tokens as unsupervised context, so the model is never taught to begin a score mid-way. The old
    uniformly random window almost never contained the closing token for long songs, and a planner LoRA
    trained that way stops ending its scores after a few dozen steps.
    """
    span = len(seq.ids) - seq.loss_start
    limit = min(max_tokens, CONTEXT - seq.loss_start)
    if span <= limit:
        return seq.ids, seq.loss_start
    draw = rng.random()
    if draw < 1.0 / 3.0:
        start = 0
    elif draw < 2.0 / 3.0:
        start = span - limit
    else:
        start = rng.randint(0, span - limit)
    ids = seq.ids[:seq.loss_start] + seq.ids[seq.loss_start + start: seq.loss_start + start + limit]
    context = min(CROP_CONTEXT, limit // 4) if start > 0 else 0
    return ids, seq.loss_start + context


def pick_sequence(rng: random.Random, train: list, regularization: list, fraction: float):
    """One training sequence per micro-step: a regularization score with probability ``fraction``, else a song."""
    if regularization and fraction > 0 and rng.random() < fraction:
        return rng.choice(regularization)
    return rng.choice(train)


def build_eval_set(sequences: list[_Sequence], cfg: PlannerConfig, seed: int) -> list[tuple[list, int]]:
    """Fixed crops (ids, loss_start) for the validation loss; the same crops are scored at every evaluation."""
    n = int(cfg.eval_samples) if cfg.eval_every > 0 else 0
    if n <= 0 or not sequences:
        return []
    rng = random.Random(seed + 12345)
    order = list(range(len(sequences)))
    rng.shuffle(order)
    return [_crop(sequences[order[i % len(order)]], cfg.max_tokens, rng) for i in range(n)]


def _evaluate(replica: "Replica", eval_set: list[tuple[list, int]]) -> float:
    """Mean cross-entropy over the fixed evaluation crops with the current LoRA (no gradients)."""
    device, llm = replica.device, replica.root.model
    total = 0.0
    with torch.no_grad():
        for ids, loss_start in eval_set:
            tokens = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                hidden = ar_hidden(llm, tokens, torch.bfloat16, checkpointing=False)
                loss = chunked_cross_entropy(llm.lm_head, hidden[0, loss_start - 1: -1], tokens[0, loss_start:])
            total += loss.item()
            del hidden, loss, tokens
    return total / len(eval_set)


def generate_abc(clip, style: str, lyrics: str, mode: str, seed: int, max_tokens: int,
                 sampling: Optional[dict] = None) -> tuple[str, int, bool]:
    """Write one ABC score through ComfyUI's own YuE2 sampler (the LoRA hooks on ``clip`` apply).

    Returns (abc_text, generated_tokens, ended): ``ended`` is False when the whole ``max_tokens`` budget was
    used without producing the closing token, which is what an over-trained planner LoRA does.
    """
    sampling = {**PROBE_SAMPLING, **(sampling or {})}
    tokens = clip.tokenize(style, lyrics=lyrics, cot=mode, seed=seed, max_tokens=max_tokens, penalty_window=100)
    with torch.no_grad():
        ids = clip.generate(tokens, max_length=max_tokens, seed=seed, **sampling)
    ids = [int(t) for t in ids]
    return clip.decode(ids), len(ids), len(ids) < max_tokens


@contextlib.contextmanager
def _adapter_weights_as(lora, dtype):
    """Temporarily run the LoRA adapters in ``dtype`` (ComfyUI's sampler feeds the bypass hooks bf16 activations
    and does not autocast, while the trainable weights are fp32). The fp32 tensors are put back afterwards."""
    stash = [(param, param.data) for adapter in lora.adapters for param in adapter.parameters()]
    try:
        for param, data in stash:
            param.data = data.to(dtype)
        yield
    finally:
        for param, data in stash:
            param.data = data


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
                       interrupt_check: Optional[Callable[[], None]] = None,
                       regularization: Optional[Dataset] = None) -> TrainResult:
    import comfy.model_management

    start_time = time.perf_counter()
    torch.manual_seed(cfg.seed)
    devices = resolve_devices(cfg.devices)
    sequences = build_sequences(clip, dataset, cfg)
    held = split_holdout([s.item.id for s in sequences], cfg.eval_holdout, cfg.seed) if cfg.eval_every > 0 else set()
    eval_pool = [s for s in sequences if s.item.id in held] if held else sequences
    if held:
        sequences = [s for s in sequences if s.item.id not in held]
        logging.info("YuE2 trainer: held out for evaluation (not trained on): %s", ", ".join(sorted(held)))
    elif cfg.eval_holdout > 0 and cfg.eval_every > 0:
        logging.warning("YuE2 trainer: too few items to hold one out for evaluation; scoring training crops instead")
    logging.info("YuE2 trainer: %d planner sequences (%d abc, %d semantic)", len(sequences),
                 sum(s.kind == "abc" for s in sequences), sum(s.kind == "semantic" for s in sequences))
    reg_sequences: list[_Sequence] = []
    if regularization is not None and regularization.items and cfg.regularization_fraction > 0:
        try:
            reg_sequences = build_sequences(clip, regularization, cfg)
        except ValueError:
            logging.warning("YuE2 trainer: the regularization dataset has nothing for the enabled targets (train_abc=%s, "
                            "train_semantic=%s: ABC scores need train_abc, semantic tokens need train_semantic); ignoring it",
                            cfg.train_abc, cfg.train_semantic)
        else:
            logging.info("YuE2 trainer: %d regularization sequences (%d abc, %d semantic), drawn for %.0f%% of the micro-steps",
                         len(reg_sequences), sum(q.kind == "abc" for q in reg_sequences),
                         sum(q.kind == "semantic" for q in reg_sequences), cfg.regularization_fraction * 100.0)
    probe_style = cfg.probe_style.strip() or sequences[0].item.style
    probe_lyrics = cfg.probe_lyrics if cfg.probe_lyrics.strip() else sequences[0].item.lyrics
    probe_mode = cfg.probe_mode or (cfg.abc_mode if cfg.abc_mode in ("full", "melody") else "full")

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
    eval_set = build_eval_set(eval_pool, cfg, cfg.seed)
    drift_set = build_eval_set(reg_sequences, cfg, cfg.seed + 777) if reg_sequences else []
    losses: list[float] = []
    evals: list[list] = []
    drift: list[list] = []
    probes: list[dict] = []
    start_step = 0
    restored = restore_state(cfg.resume_state, "planner", cfg, optimizer, replicas) if cfg.resume_state else None
    if restored:
        start_step = int(restored["step"])
        losses, evals = list(restored["losses"]), [list(e) for e in restored["evals"]]
        drift, probes = [list(e) for e in restored.get("drift", [])], list(restored.get("probes", []))
        if start_step >= cfg.steps:
            raise ValueError(f"The resumed run is already at step {start_step}; set steps above it to continue "
                             "(or turn resume_state off to start a new run from the LoRA weights)")
        logging.info("YuE2 trainer: resuming at step %d of %d (optimizer and random state restored)", start_step, cfg.steps)
    info = {"kind": "planner", "rank": cfg.rank, "alpha": cfg.alpha, "targets": cfg.targets, "steps": cfg.steps,
            "eval_every": cfg.eval_every if eval_set else 0, "eval_samples": len(eval_set), "eval": evals,
            "eval_holdout": sorted(held), "drift": drift, "probe": probes,
            "regularization_sequences": len(reg_sequences),
            "regularization_fraction": cfg.regularization_fraction if reg_sequences else 0.0,
            "probe_every": cfg.probe_every, "resumed_from": start_step,
            "sequences": len(sequences), "train_abc": cfg.train_abc, "train_semantic": cfg.train_semantic,
            "max_tokens": cfg.max_tokens, "learning_rate": cfg.learning_rate, "lr_schedule": cfg.lr_schedule,
            "devices": [str(d) for d in devices], "micro_steps": micro_steps,
            }
    monitor = TrainMonitor("planner", cfg.steps, cfg.log_every, cfg.tensorboard_dir or None, cfg.run_name,
                           config={k: v for k, v in vars(cfg).items()
                                   if k not in ("existing_lora", "save_callback", "probe_callback", "resume_state")},
                           eval_label="held-out" if held else "fixed-set", start_step=start_step)
    monitor.evals = [tuple(e) for e in evals]
    monitor.losses = list(losses)
    monitor.drift = [tuple(e) for e in drift]
    monitor.probes = list(probes)

    def work_fn(replica: Replica, n_micro: int) -> torch.Tensor:
        device, llm = replica.device, replica.root.model
        py_rng = replica.extra["py_rng"]
        total = torch.zeros((), device=device, dtype=torch.float32)
        for _ in range(n_micro):
            seq = pick_sequence(py_rng, sequences, reg_sequences, cfg.regularization_fraction)
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

    def run_eval(index: int):
        value = _evaluate(primary, eval_set)
        evals.append([index, value])
        moved = _evaluate(primary, drift_set) if drift_set else None
        if moved is not None:
            drift.append([index, moved])
        monitor.eval(index, value, moved)

    def run_probe(index: int):
        started = time.perf_counter()
        with _adapter_weights_as(primary.lora, torch.bfloat16):
            text, count, ended = generate_abc(primary.extra["clip"], probe_style, probe_lyrics, probe_mode,
                                              cfg.probe_seed, cfg.probe_max_tokens)
        seconds = time.perf_counter() - started
        meta = {"step": index, "tokens": count, "ended": ended, "seconds": seconds, "style": probe_style,
                "lyrics": probe_lyrics, "mode": probe_mode, "seed": cfg.probe_seed, "max_tokens": cfg.probe_max_tokens}
        path = cfg.probe_callback(index, text, meta) if cfg.probe_callback else None
        entry = {"step": index, "tokens": count, "ended": ended, "seconds": round(seconds, 1)}
        if path:
            entry["path"] = str(path)
        probes.append(entry)
        monitor.probe(index, count, ended, seconds, path)

    def state_at(step: int) -> dict:
        return capture_state("planner", cfg, step, optimizer, replicas, losses, evals,
                             extra={"drift": drift, "probes": probes})

    def due(step: int, every: int) -> bool:
        return every > 0 and (step % every == 0 or step == cfg.steps)

    final_state = None
    try:
        if eval_set and not (evals and evals[-1][0] == start_step):
            run_eval(start_step)
        if cfg.probe_every > 0 and not (probes and probes[-1]["step"] == start_step):
            run_probe(start_step)
        for step in range(start_step, cfg.steps):
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
            if eval_set and due(step + 1, cfg.eval_every):
                run_eval(step + 1)
            if due(step + 1, cfg.probe_every):
                run_probe(step + 1)
            if progress is not None:
                progress(step + 1, cfg.steps, step_loss)
            if cfg.save_every and cfg.save_callback and (step + 1) % cfg.save_every == 0 and step + 1 < cfg.steps:
                cfg.save_callback(primary.lora.export(), step + 1,
                                  {**info, "steps": step + 1, "partial": True, "loss": step_loss}, state_at(step + 1))
        final_state = state_at(cfg.steps)
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
    if evals:
        info["eval_loss_start"], info["eval_loss_final"] = evals[0][1], evals[-1][1]
    return TrainResult(lora_sd=exported, losses=losses, steps=cfg.steps,
                       seconds=time.perf_counter() - start_time, info=info, evals=evals, state=final_state)


__all__ = ["PlannerConfig", "train_planner_lora", "build_sequences", "build_eval_set", "CROP_CONTEXT",
           "pick_sequence", "generate_abc", "PROBE_SAMPLING"]
