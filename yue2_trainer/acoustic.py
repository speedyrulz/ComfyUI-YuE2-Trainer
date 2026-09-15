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

from .constants import FRAMES_PER_SECOND, LATENT_CHANNELS, MODEL_KEY_PREFIX, CONTEXT
from .dataset import Dataset, Item
from .ema import WeightAverage, averaged
from .forward import nar_forward
from .lora import adapter_weights_as, create_lora, select_target_modules, count_parameters
from .monitor import TrainMonitor
from .parallel import (Replica, clone_patcher_for_device, free_replicas, reduce_gradients, resolve_devices,
                       run_on_replicas, split_counts, sync_lora_weights)
from .constants import MUSIC_END
from .prefix import (PrefixCache, build_acoustic_prefix, compute_prefix_kv, load_clip_for_prefill, music_prefix_ids,
                     negative_prefix_ids, resolve_mode)
from .render import conditioning_from_tokens, decode_audio, sample_latents
from .resume import capture_state, restore_state


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
    mode: str = "full"                  # full|melody|off|auto  (CoT instruction used for the prefix)
    conditioning: str = "compact"       # compact | inference_like  (see README: acoustic conditioning regimes)
    use_semantic_tokens: bool = False   # condition on semantic tokens for items that carry them
    caption_dropout: float = 0.1        # probability of training a step on the unconditional (instruction-only) prefix
    timestep_sampling: str = "uniform"  # uniform | logit_normal
    shift: float = 1.0                  # sigma = shift*u / (1 + (shift-1)*u)
    logit_mean: float = 0.0
    logit_std: float = 1.0
    weight_decay: float = 0.01
    lr_schedule: str = "cosine"          # cosine | constant | linear (warmup applies to all)
    warmup_steps: int = 10
    max_grad_norm: float = 1.0
    seed: int = 0
    lora_dtype: str = "fp32"
    gradient_checkpointing: bool = True
    optimizer: str = "AdamW"
    devices: str = "auto"               # auto | cuda:N | all | cuda:0,cuda:1
    log_every: int = 1                  # console line every N steps
    eval_every: int = 50                # fixed-noise validation loss every N steps (0 = off)
    eval_samples: int = 8               # size of the fixed evaluation set
    eval_holdout: int = 1               # songs kept out of training and used for the evaluation set (0 = score training crops)
    keep: str = "final"                # final | best_eval: which weights the trainer returns
    ema_decay: float = 0.0             # EMA of the LoRA weights, warm-started (0 = off); see ema.py
    tensorboard_dir: str = ""           # "" = off; parent folder for TensorBoard runs
    run_name: str = ""                  # TensorBoard run name (timestamp appended)
    existing_lora: Optional[dict] = None
    resume_state: Optional[dict] = None   # optimizer / RNG / step state saved next to existing_lora
    save_every: int = 0
    save_callback: Optional[Callable] = None   # (lora_sd, step, info, state)
    sample_every: int = 0               # render sample_codes with the current LoRA every N steps (0 = off; needs a VAE)
    sample_seconds: float = 30.0
    sample_seed: int = 0
    sample_codes: Optional[list] = None      # music tokens (codebook indices) of the rendered stream
    sample_prompt: Optional[dict] = None     # style / lyrics / abc / mode the stream was written under
    sample_callback: Optional[Callable] = None   # (step, audio [N, C] float32, rate) -> saved path


@dataclass
class TrainResult:
    lora_sd: dict
    losses: list = field(default_factory=list)
    steps: int = 0
    seconds: float = 0.0
    info: dict = field(default_factory=dict)
    evals: list = field(default_factory=list)   # [[step, fixed-noise loss], ...]
    state: Optional[dict] = None                # resumable optimizer / RNG / step state at the end of the run
    raw_lora_sd: Optional[dict] = None          # with ema_decay: the last step's live weights (lora_sd is the average)


def _shift_sigma(cfg: AcousticConfig, u: float) -> float:
    u = min(max(u, 1e-4), 1.0 - 1e-4)
    if cfg.shift != 1.0:
        u = cfg.shift * u / (1.0 + (cfg.shift - 1.0) * u)
    return float(u)


def _sample_sigma(cfg: AcousticConfig, rng: torch.Generator) -> float:
    if cfg.timestep_sampling == "logit_normal":
        u = torch.sigmoid(torch.randn((), generator=rng) * cfg.logit_std + cfg.logit_mean).item()
    else:
        u = torch.rand((), generator=rng).item()
    return _shift_sigma(cfg, u)


def _sigma_quantile(cfg: AcousticConfig, q: float) -> float:
    """Sigma at quantile ``q`` of the training sigma distribution (used to stratify the evaluation set)."""
    q = min(max(q, 1e-4), 1.0 - 1e-4)
    if cfg.timestep_sampling == "logit_normal":
        u = torch.sigmoid(torch.special.ndtri(torch.tensor(q, dtype=torch.float64)) * cfg.logit_std + cfg.logit_mean).item()
    else:
        u = q
    return _shift_sigma(cfg, u)


def split_holdout(item_ids: list, holdout: int, seed: int) -> set:
    """Deterministic set of item ids to keep out of training for the evaluation set.

    Never holds out more than ``items - 3`` so that a small dataset keeps enough songs to train on;
    with three items or fewer nothing is held out.
    """
    unique = sorted(set(item_ids))
    n = min(int(holdout), max(0, len(unique) - 3)) if holdout > 0 else 0
    if n <= 0:
        return set()
    rng = random.Random(seed + 4242)
    return set(rng.sample(unique, n))


@dataclass
class _EvalEntry:
    index: int            # training sample (chunk) index
    offset: int           # frame offset inside the chunk
    length: int           # frames
    sigma: float
    noise: torch.Tensor   # [1, 64, length] on CPU


def build_eval_set(chunks: list[tuple[int, int]], cfg: AcousticConfig, seg_frames: int, seed: int) -> list[_EvalEntry]:
    """Fixed crops, sigmas and noise for the validation loss.

    The same entries are evaluated every time, so the resulting curve moves only when the model does;
    sigmas are stratified over the training distribution instead of sampled, which removes most of the
    variance a random draw would add.
    """
    n = int(cfg.eval_samples) if cfg.eval_every > 0 else 0
    if n <= 0 or not chunks:
        return []
    rng = random.Random(seed + 12345)
    gen = torch.Generator().manual_seed(seed + 12345)
    order = list(range(len(chunks)))
    rng.shuffle(order)
    entries = []
    for i in range(n):
        index = order[i % len(order)]
        c0, c1 = chunks[index]
        frames = c1 - c0
        length = seg_frames if seg_frames and frames > seg_frames else frames
        offset = rng.randint(0, frames - length) if frames > length else 0
        sigma = _sigma_quantile(cfg, (i + 0.5) / n)
        noise = torch.randn((1, LATENT_CHANNELS, length), generator=gen)
        entries.append(_EvalEntry(index, offset, length, sigma, noise))
    return entries


def _evaluate(replica: "Replica", samples: list, eval_set: list[_EvalEntry], cfg: AcousticConfig) -> float:
    """Mean flow-matching loss over the fixed evaluation set with the current LoRA (no gradients)."""
    device, dm = replica.device, replica.root.diffusion_model
    total = 0.0
    with torch.no_grad():
        for entry in eval_set:
            sample = samples[entry.index]
            c0, _ = sample.chunk
            x0 = sample.item.latents[:, c0 + entry.offset: c0 + entry.offset + entry.length]
            x0 = x0.to(device=device, dtype=torch.float32)[None]
            noise = entry.noise.to(device)
            sigma = torch.tensor([entry.sigma], device=device)
            x_t = (1.0 - sigma.view(-1, 1, 1)) * x0 + sigma.view(-1, 1, 1) * noise
            prefix_kv = sample.prefix.kv.to(device).clone()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred = nar_forward(dm, x_t.to(torch.bfloat16), sigma, prefix_kv, sample.prefix.ar_length,
                                   frame_offset=0 if sample.compact else entry.offset, checkpointing=False)
            total += torch.nn.functional.mse_loss(pred.float(), noise - x0).item()
            del pred, prefix_kv, x_t, x0, noise
    return total / len(eval_set)


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


LR_SCHEDULES = ["cosine", "constant", "linear"]


def _lr_at(step: int, total: int, warmup: int, base: float, schedule: str = "cosine") -> float:
    """Learning rate for 0-based ``step``. Warmup ramps linearly for ``warmup`` steps in every schedule.

    cosine:   decays from base to 10% of base with a half-cosine
    constant: stays at base
    linear:   decays from base to 10% of base in a straight line
    """
    if warmup > 0 and step < warmup:
        return base * (step + 1) / warmup
    if schedule == "constant":
        return base
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    if schedule == "linear":
        return base * (1.0 - 0.9 * progress)
    return base * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


@dataclass
class _Sample:
    item: Item
    prefix: PrefixCache
    chunk: tuple[int, int]
    uncond: Optional[PrefixCache] = None   # instruction-only prefix for caption dropout
    compact: bool = False                  # positions restart at the segment (compact regime)


def _chunk_ranges(frames: int, prefix_len: int) -> list[tuple[int, int]]:
    """Original inference chunking (comfy.text_encoders.yue2.chunk_ranges): keeps every position inside the context."""
    size = max(1, (CONTEXT - prefix_len - 3) // 2)
    return [(s, min(s + size, frames)) for s in range(0, frames, size)]


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
        if semantic is not None and len(semantic) < item.frames - 2:
            logging.warning("YuE2 trainer: %s has %d semantic tokens but %d latent frames; using text-only conditioning",
                            item.id, len(semantic), item.frames)
            semantic = None
        elif semantic is not None and len(semantic) > item.frames + 2:
            logging.info("YuE2 trainer: %s has %d semantic tokens for %d latent frames; using the first %d "
                         "(the audio was cropped from the start)", item.id, len(semantic), item.frames, item.frames)
        if semantic is not None:
            n = min(len(semantic), item.frames)
            prefix_ids, _ = music_prefix_ids(clip, item.style, item.lyrics, abc, cot)
            for chunk in _chunk_ranges(item.frames, len(prefix_ids)):
                chunk = (chunk[0], min(chunk[1], n))
                if chunk[1] <= chunk[0]:
                    continue
                prefix = build_acoustic_prefix(clip, item.style, item.lyrics, abc, cot, semantic=semantic[:n],
                                               chunk=chunk, device=device)
                uncond = None
                if cfg.caption_dropout > 0:
                    uncond = build_acoustic_prefix(clip, item.style, item.lyrics, abc, cot, semantic=semantic[:n],
                                                   chunk=chunk, device=device, unconditional=True)
                samples.append(_Sample(item, prefix, chunk, uncond))
        elif cfg.conditioning == "compact":
            # The regime of the standalone trainers: cot=off, style only, MUSIC_END right after MUSIC_START,
            # the NAR tokens immediately after the prefix and latent positions restarting at every segment.
            ids = music_prefix_ids(clip, item.style, "", None, "off")[0] + [MUSIC_END]
            prefix = PrefixCache(ids=ids, kv=compute_prefix_kv(clip, ids, device=device), ar_length=len(ids))
            uncond = None
            if cfg.caption_dropout > 0:
                uids = negative_prefix_ids(clip, [], "off") + [MUSIC_END]
                uncond = PrefixCache(ids=uids, kv=compute_prefix_kv(clip, uids, device=device), ar_length=len(uids))
            samples.append(_Sample(item, prefix, (0, item.frames), uncond, compact=True))
        else:
            # Text-only (codec-dropout) regime with inference geometry: the song is split into the same chunks
            # inference would use, and the NAR tokens keep the positions they would have after that chunk's
            # codec tokens. One prefix K/V cache is shared by all chunks of the item.
            prefix_ids, abc_ids = music_prefix_ids(clip, item.style, item.lyrics, abc, cot)
            kv = compute_prefix_kv(clip, prefix_ids, device=device)
            ukv = None
            if cfg.caption_dropout > 0:
                uids = negative_prefix_ids(clip, abc_ids, cot)
                ukv = (uids, compute_prefix_kv(clip, uids, device=device))
            for c0, c1 in _chunk_ranges(item.frames, len(prefix_ids)):
                ar_length = len(prefix_ids) + (c1 - c0) + 1
                prefix = PrefixCache(ids=prefix_ids, kv=kv, ar_length=ar_length)
                uncond = PrefixCache(ids=ukv[0], kv=ukv[1], ar_length=len(ukv[0]) + (c1 - c0) + 1) if ukv else None
                samples.append(_Sample(item, prefix, (c0, c1), uncond))
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
        _check_trainable_weights(mp.model.diffusion_model)
    return patchers


def _check_trainable_weights(module):
    """The base weights stay frozen; a quantized checkpoint (yue2_3b_int8_convrot) is allowed: ComfyUI's quantized
    linears back-propagate to their inputs, which is all the bypass LoRA needs."""
    weight = next(iter(p for n, p in module.named_parameters() if n.endswith("qkv_proj.weight")), None)
    if weight is None:
        return
    if weight.requires_grad:
        raise ValueError("The base model's weights must be frozen before LoRA training")
    quantized = type(weight).__name__ != "Parameter" or weight.dtype not in (torch.bfloat16, torch.float16, torch.float32)
    if quantized:
        logging.info("YuE2 trainer: training on a quantized checkpoint (%s); the LoRA trains in %s through ComfyUI's "
                     "quantized layers (less VRAM, slower steps)", type(weight).__name__, "bf16")


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
                        interrupt_check: Optional[Callable[[], None]] = None, vae=None) -> TrainResult:
    import comfy.model_management

    start_time = time.perf_counter()
    torch.manual_seed(cfg.seed)
    devices = resolve_devices(cfg.devices)

    # 1. AR prefix caches via CLIP (then free it to make room for the acoustic model).
    samples = prepare_samples(clip, dataset, cfg)
    sample_cond = None
    if cfg.sample_every > 0 and cfg.sample_codes and vae is not None:
        prompt = cfg.sample_prompt or {}
        with torch.no_grad():
            cond = conditioning_from_tokens(clip, prompt.get("style", ""), prompt.get("lyrics", ""), prompt.get("abc"),
                                            prompt.get("mode", "off"), list(cfg.sample_codes))
        sample_cond = [[cond[0][0].to("cpu"), cond[0][1]]]
        del cond
        logging.info("YuE2 trainer: a %.1f-s stream (%s) will be rendered with the LoRA every %d steps",
                     len(cfg.sample_codes) / FRAMES_PER_SECOND, prompt.get("name") or "sample", cfg.sample_every)
    elif cfg.sample_every > 0:
        logging.warning("YuE2 trainer: sample_every is set but %s; no samples will be rendered",
                        "no VAE was given" if vae is None else "there is no token stream to render")
    held = split_holdout([s.item.id for s in samples], cfg.eval_holdout, cfg.seed) if cfg.eval_every > 0 else set()
    eval_pool = [s for s in samples if s.item.id in held] if held else samples
    if held:
        samples = [s for s in samples if s.item.id not in held]
        logging.info("YuE2 trainer: held out for evaluation (not trained on): %s", ", ".join(sorted(held)))
    elif cfg.eval_holdout > 0 and cfg.eval_every > 0:
        logging.warning("YuE2 trainer: too few items to hold one out for evaluation; scoring training crops instead")
    logging.info("YuE2 trainer: %d training chunks from %d items", len(samples), len({s.item.id for s in samples}))
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
    average = WeightAverage(primary.lora.trainable, cfg.ema_decay) if cfg.ema_decay > 0 else None
    if average is not None:
        logging.info("YuE2 trainer: averaging the LoRA weights (EMA, decay %.4g, warm-started); evaluation, samples, "
                     "checkpoints and the result use the average", cfg.ema_decay)

    micro_steps = cfg.batch_size * cfg.grad_accumulation
    if micro_steps < len(devices):
        logging.warning("YuE2 trainer: batch_size x grad_accumulation (%d) is smaller than the number of GPUs (%d); "
                        "raise it to keep every GPU busy", micro_steps, len(devices))
    counts = split_counts(micro_steps, len(replicas))
    seg_frames = int(round(cfg.segment_seconds * FRAMES_PER_SECOND)) if cfg.segment_seconds > 0 else 0
    eval_set = build_eval_set([s.chunk for s in eval_pool], cfg, seg_frames, cfg.seed)
    losses: list[float] = []
    samples_log: list[dict] = []
    evals: list[list] = []
    start_step = 0
    restored = restore_state(cfg.resume_state, "acoustic", cfg, optimizer, replicas) if cfg.resume_state else None
    if restored:
        start_step = int(restored["step"])
        losses, evals = list(restored["losses"]), [list(e) for e in restored["evals"]]
        samples_log = list(restored.get("samples", []))
        _restore_average(average, restored, replicas)
        if start_step >= cfg.steps:
            raise ValueError(f"The resumed run is already at step {start_step}; set steps above it to continue "
                             "(or turn resume_state off to start a new run from the LoRA weights)")
        logging.info("YuE2 trainer: resuming at step %d of %d (optimizer and random state restored)", start_step, cfg.steps)
    info = {"kind": "acoustic", "rank": cfg.rank, "alpha": cfg.alpha, "targets": cfg.targets,
            "eval_every": cfg.eval_every if eval_set else 0, "eval_samples": len(eval_set), "eval": evals,
            "eval_holdout": sorted(held), "resumed_from": start_step,
            "train_acoustic_head": cfg.train_acoustic_head, "steps": cfg.steps, "items": len({s.item.id for s in samples}),
            "chunks": len(samples), "segment_seconds": cfg.segment_seconds, "timestep_sampling": cfg.timestep_sampling,
            "shift": cfg.shift, "learning_rate": cfg.learning_rate, "lr_schedule": cfg.lr_schedule, "mode": cfg.mode,
            "caption_dropout": cfg.caption_dropout, "conditioning": cfg.conditioning,
            "devices": [str(d) for d in devices], "micro_steps": micro_steps,
            "semantic_conditioned_chunks": sum(1 for s in samples if s.prefix.ar_length == len(s.prefix.ids)),
            "sample_every": cfg.sample_every if sample_cond is not None else 0, "samples": samples_log,
            "ema_decay": cfg.ema_decay}

    monitor = TrainMonitor("acoustic", cfg.steps, cfg.log_every, cfg.tensorboard_dir or None, cfg.run_name,
                           config={k: v for k, v in vars(cfg).items()
                                   if k not in ("existing_lora", "save_callback", "resume_state")},
                           eval_label="held-out" if held else "fixed-set", start_step=start_step)
    monitor.evals = [tuple(e) for e in evals]
    monitor.losses = list(losses)

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
            prefix = sample.prefix
            if sample.uncond is not None and py_rng.random() < cfg.caption_dropout:
                prefix = sample.uncond
            prefix_kv = prefix.kv.to(device, non_blocking=True).clone()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred = nar_forward(dm, x_t.to(torch.bfloat16), sigma, prefix_kv, prefix.ar_length,
                                   frame_offset=0 if sample.compact else offset,
                                   checkpointing=cfg.gradient_checkpointing)
            loss = torch.nn.functional.mse_loss(pred.float(), target)
            (loss / micro_steps).backward()
            total += loss.detach()
            del pred, loss, prefix_kv, x_t, x0, noise
        return total

    best: dict = {}
    sample_failed: list = []

    def run_eval(index: int):
        with averaged(average):
            value = _evaluate(primary, eval_pool, eval_set, cfg)
            evals.append([index, value])
            monitor.eval(index, value)
            if cfg.keep == "best_eval" and index > 0 and (not best or value < best["eval"]):
                best.update(step=index, eval=value, lora_sd=primary.lora.export(), state=state_at(index))

    def state_at(step: int) -> dict:
        return capture_state("acoustic", cfg, step, optimizer, replicas, losses, evals,
                             extra={"samples": samples_log, **_average_state(average)})

    def run_sample(index: int):
        if sample_failed:
            return
        try:
            with averaged(average):
                _run_sample(index)
        except Exception as exc:  # noqa: BLE001 - rendering is auxiliary to the run
            if type(exc).__name__ == "InterruptProcessingException":
                raise
            sample_failed.append(str(exc))
            logging.warning("YuE2 trainer: rendering the sample failed (%s); no further samples this run", exc)
            comfy.model_management.soft_empty_cache()

    def _run_sample(index: int):
        started = time.perf_counter()
        frames = len(cfg.sample_codes)
        with adapter_weights_as(primary.lora, torch.bfloat16):
            latents = sample_latents(primary.extra["patcher"], sample_cond, frames, cfg.sample_seed)
        audio, rate = decode_audio(vae, latents, primary.device)
        del latents
        path = cfg.sample_callback(index, audio, rate) if cfg.sample_callback else None
        samples_log.append({"step": index, "seconds": round(frames / FRAMES_PER_SECOND, 1), "path": str(path) if path else None})
        monitor.sample(index, frames / FRAMES_PER_SECOND, path, time.perf_counter() - started)
        comfy.model_management.soft_empty_cache()

    final_state = None
    try:
        if eval_set and not (evals and evals[-1][0] == start_step):
            run_eval(start_step)
        if sample_cond is not None and not (samples_log and samples_log[-1]["step"] == start_step):
            run_sample(start_step)
        for step in range(start_step, cfg.steps):
            if interrupt_check is not None:
                interrupt_check()
            for group in optimizer.param_groups:
                group["lr"] = _lr_at(step, cfg.steps, cfg.warmup_steps, cfg.learning_rate, cfg.lr_schedule)
            optimizer.zero_grad(set_to_none=True)
            loss_sum = run_on_replicas(replicas, work, counts)
            reduce_gradients(replicas)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(primary.lora.trainable, cfg.max_grad_norm or float("inf")))
            optimizer.step()
            sync_lora_weights(replicas)
            if average is not None:
                average.update()
            step_loss = loss_sum / micro_steps
            losses.append(step_loss)
            monitor.step(step + 1, step_loss, optimizer.param_groups[0]["lr"], grad_norm)
            if eval_set and ((step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps):
                run_eval(step + 1)
            if sample_cond is not None and ((step + 1) % cfg.sample_every == 0 or step + 1 == cfg.steps):
                run_sample(step + 1)
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
            replica.lora.eject(replica.extra["patcher"])
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        free_replicas(replicas)
        if len(patchers) > 1:  # drop the extra full-model copies from VRAM
            del patchers[1:]
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
    exported, final_state = _keep(cfg, best, exported, final_state, info, "acoustic")
    return TrainResult(lora_sd=exported, losses=losses, steps=cfg.steps,
                       seconds=time.perf_counter() - start_time, info=info, evals=evals, state=final_state,
                       raw_lora_sd=raw)


def _keep(cfg, best: dict, exported: dict, final_state, info: dict, kind: str):
    """Apply ``cfg.keep``: return the weights (and resume state) of the best evaluation instead of the last step."""
    info["kept_step"] = cfg.steps
    if cfg.keep != "best_eval":
        return exported, final_state
    if not best:
        logging.warning("YuE2 trainer: keep=best_eval but no evaluation ran after step 0 (eval_every off?); keeping the final step")
        return exported, final_state
    info["kept_step"], info["kept_eval"] = best["step"], best["eval"]
    if best["step"] == cfg.steps:
        logging.info("YuE2 %s: the final step had the best evaluation loss (%.4f)", kind, best["eval"])
        return exported, final_state
    logging.info("YuE2 %s: keeping the weights from step %d (evaluation loss %.4f) instead of the final step %d",
                 kind, best["step"], best["eval"], cfg.steps)
    return best["lora_sd"], best["state"]


def _average_state(average) -> dict:
    """The weight average's resume state, when there is one."""
    return {"ema": average.state()} if average is not None else {}


def _restore_average(average, restored: dict, replicas: list):
    """Continue the weight average (and the live weights) from the resume state, or restart it."""
    if average is None:
        return
    if average.load(restored.get("ema")):
        sync_lora_weights(replicas)
        logging.info("YuE2 trainer: weight average restored (%d updates)", average.updates)
    else:
        logging.warning("YuE2 trainer: the resume state has no matching weight average; averaging restarts from the "
                        "checkpoint's weights")


__all__ = ["AcousticConfig", "TrainResult", "train_acoustic_lora", "prepare_samples", "build_eval_set", "split_holdout"]
