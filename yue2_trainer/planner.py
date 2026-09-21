"""Planner / semantic (AR next-token) LoRA training on ComfyUI's YuE2 CLIP (the language model)."""
from __future__ import annotations

import contextlib
import logging
import random
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from .acoustic import (TrainResult, _average_state, _check_trainable_weights, _keep, _make_optimizer, _lr_at,
                       _restore_average, split_holdout)
from .constants import CLIP_KEY_PREFIX, CODEC_OFFSET, CODEC_SIZE, CONTEXT, FRAMES_PER_SECOND, MUSIC_END
from .dataset import CHORD_RE, Dataset, Item
from .ema import WeightAverage, averaged
from .forward import ar_hidden, chunked_cross_entropy, chunked_losses
from .lora import adapter_weights_as, create_lora, select_target_modules, count_parameters
from .monitor import TrainMonitor
from .parallel import (Replica, clone_patcher_for_device, free_replicas, reduce_gradients, resolve_devices,
                       run_on_replicas, split_counts, sync_lora_weights)
from .prefix import abc_sequence, music_prefix_ids, negative_prefix_ids, resolve_mode, load_clip_for_prefill
from .render import conditioning_from_tokens
from .resume import capture_state, restore_state

PROBE_SAMPLING = {"temperature": 0.7, "top_p": 0.9, "top_k": 30, "repetition_penalty": 1.005}   # YuE2GenerateABC defaults
MUSIC_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 100, "repetition_penalty": 1.2}    # YuE2GenerateMusic defaults


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
    keep: str = "final"                # final | best_eval: which weights the trainer returns
    ema_decay: float = 0.0             # EMA of the LoRA weights, warm-started (0 = off); see ema.py
    regularization_fraction: float = 0.5   # share of micro-steps drawn from the regularization scores (when given)
    kl_weight: float = 0.0              # trust region: weight of KL(base || LoRA) on the trained positions (0 = off)
    abc_dropout: float = 0.0            # share of semantic draws trained with the no-sheet (cot off) prompt
    probe_every: int = 0                # generate a score with the current LoRA every N steps (0 = off)
    probe_style: str = ""               # blank = style prompt of the first training item
    probe_lyrics: str = ""              # blank = lyrics of the first training item
    probe_mode: str = ""                # blank = abc_mode (full when abc_mode is auto)
    probe_max_tokens: int = 8192        # token budget of a probe; a probe that uses it all did not end its score
    probe_seed: int = 0
    probe_abc: str = ""                 # fixed score for the samples (blank = the score each probe writes)
    sample_every: int = 0               # every N steps (and before step 1) write a song clip with the current LoRA: score,
                                        # sample_seconds of music tokens, audio through render_callback (0 = off)
    sample_seconds: float = 30.0
    sample_seed: int = 0                # seed of a sample's music tokens and of its rendering
    probe_callback: Optional[Callable[..., Optional[str]]] = None   # (step, abc, meta, music_tokens | None) -> saved path
    render_callback: Optional[Callable[..., Optional[str]]] = None  # (step, conditioning, frames, meta) -> audio path
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
    alt_ids: Optional[list] = None      # semantic: the same tokens behind the no-sheet (cot off) prompt
    alt_loss_start: int = 0


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
            ended = bool(item.extra.get("semantic_ended", True))   # a stream cut off at a budget must not teach a false ending
            span = [int(t) + CODEC_OFFSET for t in item.semantic] + ([MUSIC_END] if ended else [])
            alt_ids, alt_start = None, 0
            if cot != "off":   # the sheet-less variant for abc_dropout (what Comfy runs with an empty ABC input)
                alt_prefix, _ = music_prefix_ids(clip, item.style, item.lyrics, None, "off")
                alt_ids, alt_start = alt_prefix + span, len(alt_prefix)
            sequences.append(_Sequence(item, "semantic", prefix + span, len(prefix), alt_ids, alt_start))
    if not sequences:
        raise ValueError("No planner training sequences: items need an ABC score (train_abc) "
                         "or semantic tokens (train_semantic)")
    return sequences


CROP_CONTEXT = 256   # unsupervised context tokens at the start of a window that does not begin at the score's start


def _variant(seq: _Sequence, rng: random.Random, abc_dropout: float) -> tuple[list, int]:
    """(ids, loss_start) of a draw: a semantic sequence trains behind the no-sheet prompt with probability
    ``abc_dropout`` so one LoRA also serves generation without a score."""
    if seq.alt_ids is not None and abc_dropout > 0 and rng.random() < abc_dropout:
        return seq.alt_ids, seq.alt_loss_start
    return seq.ids, seq.loss_start


def _crop(seq: _Sequence, max_tokens: int, rng: random.Random) -> tuple[list, int]:
    return _crop_ids(seq.ids, seq.loss_start, max_tokens, rng)


def _crop_ids(ids: list, loss_start: int, max_tokens: int, rng: random.Random) -> tuple[list, int]:
    """Keep the prefix; crop the trained span to ``max_tokens`` without teaching false starts or endless scores.

    A span longer than the limit is trained through one of three windows per draw: its head (so the
    opening of a score is learned), its tail (so the closing token is learned and the model keeps ending
    its scores), or a random middle window. Windows that do not start at the beginning keep their first
    tokens as unsupervised context, so the model is never taught to begin a score mid-way. The old
    uniformly random window almost never contained the closing token for long songs, and a planner LoRA
    trained that way stops ending its scores after a few dozen steps.
    """
    span = len(ids) - loss_start
    limit = min(max_tokens, CONTEXT - loss_start)
    if span <= limit:
        return ids, loss_start
    draw = rng.random()
    if draw < 1.0 / 3.0:
        start = 0
    elif draw < 2.0 / 3.0:
        start = span - limit
    else:
        start = rng.randint(0, span - limit)
    out = ids[:loss_start] + ids[loss_start + start: loss_start + start + limit]
    context = min(CROP_CONTEXT, limit // 4) if start > 0 else 0
    return out, loss_start + context


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


def generate_music(clip, style: str, lyrics: str, abc: Optional[str], mode: str, seed: int, max_seconds: float,
                   sampling: Optional[dict] = None, cfg_scale: Optional[float] = None) -> tuple[list[int], bool]:
    """Write the music (semantic codec) token stream for a prompt through ComfyUI's own YuE2 sampler.

    ``mode`` is the planning mode the score was written in (full / melody); without a score the model runs in
    its ``off`` mode (music straight from style and lyrics), exactly as ``YuE2GenerateMusic`` does. Returns
    (codebook indices 0..32767, ended): ``ended`` is False when the stream ran to the ``max_seconds`` budget
    without the closing token. The LoRA hooks on ``clip`` apply; the sampling defaults are the music node's.
    """
    import comfy.model_management
    import comfy.ops
    sampling = {**MUSIC_SAMPLING, **(sampling or {})}
    cot = mode if abc and abc.strip() and mode in ("full", "melody") else "off"
    score = abc if cot != "off" else None
    prefix, abc_ids = music_prefix_ids(clip, style, lyrics, score, cot)
    negative = negative_prefix_ids(clip, abc_ids, cot)
    max_tokens = min(max(1, round(max_seconds * FRAMES_PER_SECOND)), CONTEXT - max(len(prefix), len(negative)))
    if max_tokens < 1:
        raise ValueError("The prompt and score fill the model context; there is no room for music tokens")
    tokens = clip.tokenize(style, lyrics=lyrics, cot=cot, abc=score or "", seed=seed, max_tokens=max_tokens, **sampling)
    if cfg_scale is not None:
        tokens["cfg_scale"] = cfg_scale
    clip.load_model(tokens)
    device = clip.patcher.load_device
    te = clip.cond_stage_model
    te.set_clip_options({"execution_device": device})
    dtype = torch.bfloat16 if comfy.model_management.should_use_bf16(device) else torch.float32
    device_context = getattr(comfy.model_management, "cuda_device_context", None)
    quantized = getattr(comfy.ops, "use_quantized_matmul", None)
    with contextlib.ExitStack() as stack:
        stack.enter_context(torch.no_grad())
        if device_context is not None:
            stack.enter_context(device_context(device))
        if quantized is not None:
            stack.enter_context(quantized(te, device))
        ids, truncated = te._generate(prefix, seed, max_tokens, "semantic", dtype, negative=negative,
                                      cfg_scale=tokens["cfg_scale"], legacy_off=cot == "off", penalty_window=50,
                                      min_tokens=min(200, max_tokens), **sampling)
    codes = [int(t) - CODEC_OFFSET for t in ids]
    if codes and not (0 <= min(codes) and max(codes) < CODEC_SIZE):
        raise RuntimeError("YuE2 music sampling returned tokens outside the codec vocabulary")
    return codes, not truncated


@contextlib.contextmanager
def _lora_off(lora):
    """Run the base model: ComfyUI's bypass hooks scale every adapter by its ``multiplier`` at call time."""
    stash = [(adapter, getattr(adapter, "multiplier", 1.0)) for adapter in lora.adapters]
    try:
        for adapter, _ in stash:
            adapter.multiplier = 0.0
        yield
    finally:
        for adapter, value in stash:
            adapter.multiplier = value


_adapter_weights_as = adapter_weights_as   # ComfyUI's sampler feeds the bypass hooks bf16 activations


def _release_prefetch():
    """What ComfyUI's executor does between nodes: drop the dynamic loader's prefetch queues and CUDA malloc
    graphs. A training node runs thousands of forwards inside one node, so they would otherwise pile up."""
    try:
        import comfy.model_prefetch
        comfy.model_prefetch.cleanup_prefetch_queues()
    except Exception as exc:  # noqa: BLE001 - older ComfyUI without the module
        logging.debug("YuE2 trainer: no prefetch queues to release (%s)", exc)


def restage(patcher):
    """Make a model loaded by ComfyUI's dynamic VRAM loader fast again for generation after a training step.

    During a training step the loader clamps how much of the model may stay resident on the GPU (a watermark
    limit on the model's address space, set from the free memory at that moment) and never raises it again.
    Token-by-token generation with a long context then no longer fits under the clamp, and the loader copies
    the trimmed weights from pinned RAM for every generated token: 96% of the GPU time in host-to-device
    copies, probes 8-20x slower (7 tokens/s instead of 60). Resetting the limits and prioritising the model,
    as a fresh load does, restores full speed. Harmless for models that are not dynamically loaded."""
    import comfy.model_management
    _release_prefetch()
    comfy.model_management.soft_empty_cache()
    vbar = getattr(patcher, "_vbar_get", lambda: None)()
    if vbar is None:
        if not getattr(patcher, "is_dynamic", lambda: False)():
            comfy.model_management.load_models_gpu([patcher], force_full_load=True)
        return
    try:
        import comfy_aimdo.control
        comfy_aimdo.control.lib.vbars_reset_watermark_limits(vbar._devctx)
        vbar.set_watermark_limit(vbar.max_size)
        vbar.prioritize()
    except Exception as exc:  # noqa: BLE001 - a comfy-aimdo without these calls
        logging.debug("YuE2 trainer: vbar watermark reset not available (%s)", exc)


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
    probe_item = sequences[0].item
    default_probe_prompt = not cfg.probe_style.strip() and not cfg.probe_lyrics.strip()

    def probe_score(text: str, ended: bool) -> tuple[Optional[str], str]:
        """The score a music probe is conditioned on, and where it came from."""
        if cfg.probe_abc.strip():
            return cfg.probe_abc, "probe_abc"
        if text and ended:
            return text, "probe"
        if default_probe_prompt and probe_item.abc:
            return probe_item.abc, "song"
        return None, "none"

    comfy.model_management.unload_all_models()
    clips = _load_clips(clip, devices)
    lora_dtype = torch.float32 if cfg.lora_dtype == "fp32" else torch.bfloat16
    comfy.model_management.in_training = True
    replicas = _setup_replicas(clips, devices, cfg, lora_dtype)
    primary = replicas[0]
    optimizer = _make_optimizer(cfg.optimizer, primary.lora.trainable, cfg.learning_rate, cfg.weight_decay)
    logging.info("YuE2 trainer: %.2fM trainable LoRA parameters on %s", count_parameters(primary.lora.trainable) / 1e6,
                 ", ".join(str(d) for d in devices))
    average = WeightAverage(primary.lora.trainable, cfg.ema_decay) if cfg.ema_decay > 0 else None
    if average is not None:
        logging.info("YuE2 trainer: averaging the LoRA weights (EMA, decay %.4g, warm-started); evaluation, probes, "
                     "samples, checkpoints and the result use the average", cfg.ema_decay)

    micro_steps = cfg.batch_size * cfg.grad_accumulation
    if micro_steps < len(devices):
        logging.warning("YuE2 trainer: batch_size x grad_accumulation (%d) is smaller than the number of GPUs (%d); "
                        "raise it to keep every GPU busy", micro_steps, len(devices))
    counts = split_counts(micro_steps, len(replicas))
    eval_set = build_eval_set(eval_pool, cfg, cfg.seed)
    drift_set = build_eval_set(reg_sequences, cfg, cfg.seed + 777) if reg_sequences else []
    losses: list[float] = []
    kls: list[float] = []
    evals: list[list] = []
    drift: list[list] = []
    probes: list[dict] = []
    start_step = 0
    restored = restore_state(cfg.resume_state, "planner", cfg, optimizer, replicas) if cfg.resume_state else None
    if restored:
        start_step = int(restored["step"])
        losses, evals = list(restored["losses"]), [list(e) for e in restored["evals"]]
        drift, probes = [list(e) for e in restored.get("drift", [])], list(restored.get("probes", []))
        kls = list(restored.get("kl", []))
        _restore_average(average, restored, replicas)
        if start_step >= cfg.steps:
            raise ValueError(f"The resumed run is already at step {start_step}; set steps above it to continue "
                             "(or turn resume_state off to start a new run from the LoRA weights)")
        logging.info("YuE2 trainer: resuming at step %d of %d (optimizer and random state restored)", start_step, cfg.steps)
    info = {"kind": "planner", "rank": cfg.rank, "alpha": cfg.alpha, "targets": cfg.targets, "steps": cfg.steps,
            "eval_every": cfg.eval_every if eval_set else 0, "eval_samples": len(eval_set), "eval": evals,
            "eval_holdout": sorted(held), "drift": drift, "probe": probes,
            "regularization_sequences": len(reg_sequences),
            "regularization_fraction": cfg.regularization_fraction if reg_sequences else 0.0,
            "kl_weight": cfg.kl_weight, "kl": kls, "abc_dropout": cfg.abc_dropout,
            "probe_every": cfg.probe_every, "sample_every": cfg.sample_every, "sample_seconds": cfg.sample_seconds,
            "ema_decay": cfg.ema_decay, "resumed_from": start_step,
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
        total = torch.zeros((2,), device=device, dtype=torch.float32)   # [cross-entropy, KL to base]
        for _ in range(n_micro):
            seq = pick_sequence(py_rng, sequences, reg_sequences, cfg.regularization_fraction)
            ids, loss_start = _crop_ids(*_variant(seq, py_rng, cfg.abc_dropout), cfg.max_tokens, py_rng)
            tokens = torch.tensor([ids], dtype=torch.long, device=device)
            base_hidden = None
            if cfg.kl_weight > 0:
                with torch.no_grad(), _lora_off(replica.lora), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    base_hidden = ar_hidden(llm, tokens, torch.bfloat16, checkpointing=False)[0, loss_start - 1: -1]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                hidden = ar_hidden(llm, tokens, torch.bfloat16, checkpointing=cfg.gradient_checkpointing)
                hidden = hidden[0, loss_start - 1: -1]
                targets_t = tokens[0, loss_start:]
                ce, kl = chunked_losses(llm.lm_head, hidden, targets_t, base_hidden)
            loss = ce if kl is None else ce + cfg.kl_weight * kl
            (loss / micro_steps).backward()
            total[0] += ce.detach()
            if kl is not None:
                total[1] += kl.detach()
            del hidden, base_hidden, loss, ce, kl, tokens
        return total

    best: dict = {}
    render_failed: list = []

    def run_eval(index: int):
        with averaged(average):
            value = _evaluate(primary, eval_set)
            evals.append([index, value])
            moved = _evaluate(primary, drift_set) if drift_set else None
            if moved is not None:
                drift.append([index, moved])
            monitor.eval(index, value, moved)
            if cfg.keep == "best_eval" and index > 0 and (not best or value < best["eval"]):
                best.update(step=index, eval=value, lora_sd=primary.lora.export(), state=state_at(index))

    def run_probe(index: int, with_music: bool):
        """A probe score; with ``with_music`` also the sample: music tokens for it, rendered through render_callback."""
        # Free the training step's cached VRAM first: with ComfyUI's dynamic VRAM loader, memory the caching
        # allocator holds looks occupied, and the loader then streams the language model's weights from RAM for
        # every generated token (8x slower probes).
        comfy.model_management.soft_empty_cache()
        restage(primary.extra["clip"].patcher)
        with averaged(average):
            _run_probe(index, with_music)
        _release_prefetch()

    def _run_probe(index: int, with_music: bool):
        started = time.perf_counter()
        with _adapter_weights_as(primary.lora, torch.bfloat16):
            text, count, ended = generate_abc(primary.extra["clip"], probe_style, probe_lyrics, probe_mode,
                                              cfg.probe_seed, cfg.probe_max_tokens)
        seconds = time.perf_counter() - started
        meta = {"step": index, "tokens": count, "ended": ended, "seconds": seconds, "style": probe_style,
                "lyrics": probe_lyrics, "mode": probe_mode, "seed": cfg.probe_seed, "max_tokens": cfg.probe_max_tokens}
        entry = {"step": index, "tokens": count, "ended": ended, "seconds": round(seconds, 1)}
        codes = music = None
        if with_music:
            abc, source = probe_score(text, ended)
            mode = resolve_mode("auto", abc, bool(abc and CHORD_RE.search(abc)))
            started = time.perf_counter()
            with _adapter_weights_as(primary.lora, torch.bfloat16):
                codes, music_ended = generate_music(primary.extra["clip"], probe_style, probe_lyrics, abc, mode,
                                                    cfg.sample_seed, cfg.sample_seconds)
            music = {"tokens": len(codes), "seconds": round(len(codes) / FRAMES_PER_SECOND, 1), "ended": music_ended,
                     "distinct": round(len(set(codes)) / max(1, len(codes)), 3), "abc_source": source, "mode": mode,
                     "budget_seconds": cfg.sample_seconds,
                     "generation_seconds": round(time.perf_counter() - started, 1)}
            meta["music"] = {**music, "abc": abc or "", "seed": cfg.sample_seed}
            entry["music"] = music
        if cfg.render_callback and codes and not render_failed:
            started = time.perf_counter()
            try:
                with _adapter_weights_as(primary.lora, torch.bfloat16):
                    conditioning = conditioning_from_tokens(primary.extra["clip"], probe_style, probe_lyrics, abc, mode, codes)
                audio_path = cfg.render_callback(index, conditioning, len(codes), meta)
                del conditioning
            except Exception as exc:  # noqa: BLE001 - rendering is auxiliary; the tokens are still written
                if type(exc).__name__ == "InterruptProcessingException":
                    raise
                render_failed.append(str(exc))
                logging.warning("YuE2 trainer: rendering the sample failed (%s); later samples stay token files "
                                "(render them with YuE2 Conditioning From Tokens)", exc)
                audio_path = None
            if audio_path:
                music["audio"] = meta["music"]["audio"] = str(audio_path)
                music["render_seconds"] = meta["music"]["render_seconds"] = round(time.perf_counter() - started, 1)
        path = cfg.probe_callback(index, text, meta, codes) if cfg.probe_callback else None
        if path:
            entry["path"] = str(path)
        probes.append(entry)
        monitor.probe(index, count, ended, seconds, path, music=music)

    def state_at(step: int) -> dict:
        return capture_state("planner", cfg, step, optimizer, replicas, losses, evals,
                             extra={"drift": drift, "probes": probes, "kl": kls, **_average_state(average)})

    def due(step: int, every: int) -> bool:
        return every > 0 and (step % every == 0 or step == cfg.steps)

    final_state = None
    try:
        if eval_set and not (evals and evals[-1][0] == start_step):
            run_eval(start_step)
        if (cfg.probe_every > 0 or cfg.sample_every > 0) and not (probes and probes[-1]["step"] == start_step):
            run_probe(start_step, cfg.sample_every > 0)
        for step in range(start_step, cfg.steps):
            if interrupt_check is not None:
                interrupt_check()
            for group in optimizer.param_groups:
                group["lr"] = _lr_at(step, cfg.steps, cfg.warmup_steps, cfg.learning_rate, cfg.lr_schedule)
            optimizer.zero_grad(set_to_none=True)
            sums = run_on_replicas(replicas, work_fn, counts)
            reduce_gradients(replicas)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(primary.lora.trainable, cfg.max_grad_norm or float("inf")))
            optimizer.step()
            sync_lora_weights(replicas)
            if average is not None:
                average.update()
            step_loss = float(sums[0]) / micro_steps
            losses.append(step_loss)
            extra = None
            if cfg.kl_weight > 0:
                kls.append(float(sums[1]) / micro_steps)
                extra = {"loss/kl": kls[-1]}
            monitor.step(step + 1, step_loss, optimizer.param_groups[0]["lr"], grad_norm, extra)
            if eval_set and due(step + 1, cfg.eval_every):
                run_eval(step + 1)
            if due(step + 1, cfg.probe_every) or due(step + 1, cfg.sample_every):
                run_probe(step + 1, due(step + 1, cfg.sample_every))
            if progress is not None:
                progress(step + 1, cfg.steps, step_loss)
            if cfg.save_every and cfg.save_callback and (step + 1) % cfg.save_every == 0 and step + 1 < cfg.steps:
                with averaged(average):
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

    raw = primary.lora.export() if average is not None else None
    with averaged(average):
        exported = primary.lora.export()
    if average is not None:
        info["ema_updates"] = average.updates
    for adapter in primary.lora.adapters:
        adapter.requires_grad_(False)
    info["tensorboard"] = str(monitor.log_dir) if monitor.log_dir else None
    if evals:
        info["eval_loss_start"], info["eval_loss_final"] = evals[0][1], evals[-1][1]
    exported, final_state = _keep(cfg, best, exported, final_state, info, "planner")
    return TrainResult(lora_sd=exported, losses=losses, steps=cfg.steps,
                       seconds=time.perf_counter() - start_time, info=info, evals=evals, state=final_state,
                       raw_lora_sd=raw)


__all__ = ["PlannerConfig", "train_planner_lora", "build_sequences", "build_eval_set", "CROP_CONTEXT",
           "pick_sequence", "generate_abc", "generate_music", "conditioning_from_tokens", "PROBE_SAMPLING",
           "MUSIC_SAMPLING"]
