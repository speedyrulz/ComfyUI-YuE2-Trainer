"""Rendering music-token streams to audio: conditioning from saved tokens, sampling through ComfyUI's own
acoustic stage, decoding with the checkpoint's VAE, and the small file helpers the probe writers use.

Used by the planner trainer (rendering its music probes on a second GPU), by the acoustic trainer (rendering
a fixed stream with the LoRA under training every N steps) and by the YuE2 Conditioning From Tokens node.
"""
from __future__ import annotations

import contextlib
import json
import logging
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .constants import CODEC_OFFSET, CODEC_SIZE, FRAMES_PER_SECOND, LATENT_CHANNELS, SAMPLE_RATE
from .dataset import CHORD_RE
from .parallel import available_cuda_devices, clone_patcher_for_device
from .prefix import music_prefix_ids, resolve_mode

LOG = logging.getLogger("yue2_trainer")


def conditioning_from_tokens(clip, style: str, lyrics: str, abc: Optional[str], mode: str, codes: list) -> list:
    """Acoustic-stage conditioning for an existing music-token stream (codebook indices 0..32767), exactly what
    ``YuE2GenerateMusic`` would output had it sampled these tokens: the AR key/value cache over
    prefix + tokens + MUSIC_END, chunked to the context. Returns a ComfyUI CONDITIONING list."""
    import comfy.model_management
    if not codes:
        raise ValueError("No music tokens to condition on")
    if min(codes) < 0 or max(codes) >= CODEC_SIZE:
        raise ValueError(f"music tokens must be codebook indices 0..{CODEC_SIZE - 1}")
    cot = mode if abc and abc.strip() and mode in ("full", "melody") else "off"
    score = abc if cot != "off" else None
    prefix, abc_ids = music_prefix_ids(clip, style, lyrics, score, cot)
    tokens = clip.tokenize(style, lyrics=lyrics, cot=cot, abc=score or "", max_tokens=len(codes))
    clip.load_model(tokens)
    device = clip.patcher.load_device
    te = clip.cond_stage_model
    te.set_clip_options({"execution_device": device})
    dtype = torch.bfloat16 if comfy.model_management.should_use_bf16(device) else torch.float32
    device_context = getattr(comfy.model_management, "cuda_device_context", None)
    with contextlib.ExitStack() as stack:
        stack.enter_context(torch.no_grad())
        if device_context is not None:
            stack.enter_context(device_context(device))
        cond, chunks = te._acoustic_conditioning(prefix, [int(c) + CODEC_OFFSET for c in codes], dtype)
    return [[cond, {"pooled_output": None, "yue2_chunks": chunks, "yue2_abc_ids": abc_ids,
                    "yue2_frames": len(codes), "yue2_truncated": False}]]


def _text(base: Path, *suffixes: str) -> str:
    for suffix in suffixes:
        p = Path(str(base) + suffix)
        if p.is_file():
            return p.read_text(encoding="utf-8")
    return ""


def tokens_and_prompt(path, style: str = "", lyrics: str = "", abc: str = "", mode: str = "auto",
                      max_seconds: float = 0.0) -> dict:
    """Load a ``.semantic.npy`` and the prompt it belongs to.

    A probe file (``step_000030.semantic.npy``) takes style, lyrics, the score and the mode from the probe's
    ``.json``; a dataset sidecar (``<song>.semantic.npy``) from the song's ``.json`` / ``.style.txt`` /
    ``.lyrics.txt`` / ``.abc`` sidecars. Explicit ``style`` / ``lyrics`` / ``abc`` / ``mode`` win. Raw vocabulary
    ids are accepted and converted. Returns dict(codes, style, lyrics, abc, mode, name)."""
    path = Path(path)
    codes = [int(c) for c in np.load(path, allow_pickle=False).reshape(-1).tolist()]
    if codes and max(codes) >= CODEC_SIZE:
        codes = [c - CODEC_OFFSET for c in codes]
    name = path.name[:-len(".semantic.npy")] if path.name.endswith(".semantic.npy") else path.stem
    base = path.parent / name
    meta: dict = {}
    meta_path = Path(str(base) + ".json")
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    music = meta.get("music") or {}
    style = style.strip() or (meta.get("style") or "") or _text(base, ".style.txt", ".tags.txt").strip()
    lyrics = lyrics if lyrics.strip() else ((meta.get("lyrics") or "") or _text(base, ".lyrics.txt", ".txt"))
    if not abc.strip():
        if "abc" in music:
            abc = music.get("abc") or ""
        elif isinstance(meta.get("abc"), str) and "\n" in meta["abc"]:
            abc = meta["abc"]
        else:
            abc = _text(base, ".abc")
    if mode == "auto":
        mode = music.get("mode") or resolve_mode("auto", abc, bool(abc and CHORD_RE.search(abc)))
    if max_seconds > 0:
        codes = codes[: max(1, round(max_seconds * FRAMES_PER_SECOND))]
    return {"codes": codes, "style": style, "lyrics": lyrics, "abc": abc, "mode": mode, "name": name}


def sample_stream(items, path: Optional[str], max_seconds: float = 0.0) -> Optional[dict]:
    """The stream the acoustic trainer renders: ``path`` (a probe's or a song's ``.semantic.npy``) or, without
    one, the first dataset item that carries semantic tokens, with its own prompt. None when there is nothing."""
    if path:
        return tokens_and_prompt(path, max_seconds=max_seconds)
    for item in items:
        if item.semantic:
            codes = list(item.semantic)
            if max_seconds > 0:
                codes = codes[: max(1, round(max_seconds * FRAMES_PER_SECOND))]
            abc = item.abc or ""
            return {"codes": codes, "style": item.style, "lyrics": item.lyrics, "abc": abc,
                    "mode": resolve_mode("auto", abc, item.has_chords()), "name": item.id}
    return None


def pick_render_device(spec: str, training: list) -> Optional[torch.device]:
    """``auto``: a CUDA device that training does not use (None when there is none); ``cuda:N``: that device;
    ``off``: None."""
    spec = (spec or "auto").strip()
    if spec in ("off", "none", ""):
        return None
    used = {torch.device(d) for d in training}
    if spec == "auto":
        for name in available_cuda_devices():
            if torch.device(name) not in used:
                return torch.device(name)
        return None
    return torch.device(spec)


def sample_latents(patcher, conditioning: list, frames: int, seed: int, steps: int = 32, cfg: float = 1.0,
                   sampler: str = "euler", scheduler: str = "simple") -> torch.Tensor:
    """Run ComfyUI's acoustic stage (the KSampler path) on ``patcher`` for ``frames`` latent frames."""
    import comfy.sample
    cond, extra = conditioning[0]
    positive = [[cond, dict(extra)]]
    negative = [[torch.zeros_like(cond), dict(extra)]]
    latent = torch.zeros((1, LATENT_CHANNELS, frames), dtype=torch.float32)
    noise = comfy.sample.prepare_noise(latent, seed)
    with torch.no_grad():
        return comfy.sample.sample(patcher, noise, steps, cfg, sampler, scheduler, positive, negative, latent, seed=seed)


def decode_audio(vae, latents: torch.Tensor, device) -> tuple[np.ndarray, int]:
    """VAE-decode [1, 64, T] latents on ``device`` without going through ComfyUI's model loader (which could
    evict a model under training); the VAE weights go back where they were. Returns (float32 [N, C], rate)."""
    module = vae.first_stage_model
    origin = next(module.parameters()).device
    dtype = getattr(vae, "vae_dtype", torch.float32)
    process = getattr(vae, "process_output", None) or (lambda audio: audio)

    def run(target):
        module.to(target)
        with torch.no_grad():
            audio = process(module.decode(latents.to(target, dtype)).float())
        if audio.shape[-1] <= 8 < audio.shape[1]:   # the raw module gives [B, C, N]; accept [B, N, C] too
            audio = audio.movedim(-1, 1)
        return audio
    try:
        try:
            audio = run(torch.device(device))
        except torch.cuda.OutOfMemoryError:
            LOG.warning("YuE2 trainer: not enough VRAM on %s to decode %d frames; decoding on the CPU", device, latents.shape[-1])
            torch.cuda.empty_cache()
            audio = run(torch.device("cpu"))
    finally:
        module.to(origin)
    std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    audio = (audio / std)[0].clamp(-1.0, 1.0).cpu().numpy().T
    rate = int(getattr(vae, "audio_sample_rate_output", getattr(vae, "audio_sample_rate", SAMPLE_RATE)))
    return np.ascontiguousarray(audio, dtype=np.float32), rate


def save_wav(path, audio: np.ndarray, rate: int) -> str:
    """16-bit PCM WAV of ``audio`` [N, C] in -1..1 (no optional dependencies)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2")
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(pcm.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(int(rate))
        handle.writeframes(np.ascontiguousarray(pcm).tobytes())
    return str(path)


class Renderer:
    """The acoustic model and VAE on one device, for rendering probe streams while the planner trains elsewhere.

    The MODEL patcher is cloned for ``device`` and loaded through ComfyUI on first use; the VAE is moved over
    for each decode and put back afterwards."""

    def __init__(self, model_patcher, vae, device, steps: int = 32, cfg: float = 1.0, sampler: str = "euler",
                 scheduler: str = "simple"):
        self.model_patcher = model_patcher
        self.vae = vae
        self.device = torch.device(device)
        self.steps, self.cfg, self.sampler, self.scheduler = steps, cfg, sampler, scheduler
        self.patcher = None

    def load(self):
        import comfy.model_management
        if self.patcher is None:
            self.patcher = clone_patcher_for_device(self.model_patcher, self.device, fresh=False)
            comfy.model_management.load_models_gpu([self.patcher], force_full_load=True)
            LOG.info("YuE2 trainer: acoustic model loaded on %s for rendering probes", self.device)
        return self.patcher

    def render(self, conditioning: list, frames: int, seed: int) -> tuple[np.ndarray, int]:
        latents = sample_latents(self.load(), conditioning, frames, seed, self.steps, self.cfg, self.sampler, self.scheduler)
        return decode_audio(self.vae, latents, self.device)


__all__ = ["conditioning_from_tokens", "tokens_and_prompt", "sample_stream", "pick_render_device", "sample_latents",
           "decode_audio", "save_wav", "Renderer"]
