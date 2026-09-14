"""Semantic tokens for real recordings through the community tokenizer head (Mothersuperior v4).

YuE2's own audio-to-semantic tokenizer is not public. Mothersuperior's *realaudio tokenizer v4* is a small
transformer head that maps MERT-v2-FullSong layer-20 features (24 kHz mono, 25 Hz) to YuE2 codebook indices
(0..32767). It was fitted on YuE2's own generations, so it is an approximation: exact top-1 agreement with
the true codes is under 10-16%, but near-miss codes render almost the same and a round trip through the
acoustic model keeps the rhythm and most of the harmony of a real recording (see README, *Semantic tokens for
your own recordings*). The preprocessing below follows the head author's recipe as implemented in
filliptm/ComfyUI-FL-YuE2 (``training/prepare.py``): 30-second MERT chunks, per-song instance normalisation,
512-frame windows with 50% overlap and a quarter-window edge trim.

Files (CC BY-NC 4.0, download them yourself):
    models/audio_encoders/tokenizer_head_joint_v4.pt   https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4
    m-a-p/MERT-v2-FullSong                             https://huggingface.co/m-a-p/MERT-v2-FullSong (folder or HF id)

Note: YuE2's sampler and ``_acoustic_conditioning`` work in raw vocabulary ids (code + CODEC_OFFSET); this
module, ``Item.semantic`` and the ``.semantic.npy`` sidecars hold codebook indices 0..32767.
"""
from __future__ import annotations

import contextlib
import logging
import math
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio import load_audio, to_stereo_48k
from .constants import CODEC_SIZE, FRAMES_PER_SECOND, SAMPLE_RATE
from .dataset import Dataset, Item, cache_key

HEAD_FILENAME = "tokenizer_head_joint_v4.pt"
DEFAULT_MERT = "m-a-p/MERT-v2-FullSong"
MERT_RATE = 24000
MERT_LAYER = 20
MERT_CHUNK = 30 * MERT_RATE
SOURCE_TAG = "mothersuperior_v4"


class TokenHead(nn.Module):
    """The v4 head: linear in, learned positions, 8 pre-norm transformer layers, linear out over the codebook."""

    def __init__(self, width=512, layers=8, heads=8, window=512, input_dim=1024, vocab=CODEC_SIZE):
        super().__init__()
        self.inp = nn.Linear(input_dim, width)
        self.pos = nn.Parameter(torch.empty(1, window, width))
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.1, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab)

    @property
    def window(self) -> int:
        return int(self.pos.shape[1])

    def forward(self, x):
        return self.head(self.norm(self.enc(self.inp(x) + self.pos[:, :x.shape[1]])))


def load_head(path, device) -> TokenHead:
    path = Path(path)
    if path.suffix == ".safetensors":
        import safetensors.torch
        state = safetensors.torch.load_file(str(path))
    else:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        state = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
    head = TokenHead()
    head.load_state_dict(state, strict=True)
    return head.to(device).eval().requires_grad_(False)


def normalize(features: np.ndarray) -> np.ndarray:
    """Per-song instance normalisation of every feature dimension (the head's ``instnorm`` setting)."""
    features = features.astype(np.float32)
    return (features - features.mean(0)) / (features.std(0) + 1e-5)


def windows(total: int, window: int) -> list[tuple[int, int, int]]:
    """(start, lo, hi) per window: predictions are taken from [lo, hi) only, trimming a quarter window at
    every edge that is not the start or the end of the song, so each frame comes from a window it sits well inside."""
    starts = list(range(0, max(1, total - window + 1), window // 2))
    if starts[-1] + window < total:
        starts.append(max(0, total - window))
    out = []
    for start in starts:
        count = min(window, total - start)
        lo = start + (0 if start == 0 else window // 4)
        hi = start + count - (0 if start + count >= total else window // 4)
        out.append((start, lo, hi))
    return out


@torch.no_grad()
def predict(head: TokenHead, features: np.ndarray) -> np.ndarray:
    """Codebook indices (int32, one per 25 Hz frame) for normalised-or-raw MERT features [T, 1024]."""
    features = normalize(features)
    total, window = len(features), head.window
    out = np.zeros(total, dtype=np.int32)
    device = next(head.parameters()).device
    for start, lo, hi in windows(total, window):
        value = features[start:start + window]
        count = len(value)
        value = np.pad(value, ((0, window - count), (0, 0)))
        x = torch.tensor(value[None], device=device)
        with _device_context(device), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = head(x)[0, :count].float()
        if not torch.isfinite(logits).all():
            with _device_context(device):
                logits = head(x)[0, :count].float()
        ids = logits.argmax(-1).cpu().numpy()
        out[lo:hi] = ids[lo - start:hi - start]
    return out


def resample(audio: np.ndarray, source: int, target: int) -> np.ndarray:
    from scipy.signal import resample_poly
    if source == target:
        return audio.astype(np.float32)
    divisor = math.gcd(source, target)
    return resample_poly(audio, target // divisor, source // divisor, axis=0).astype(np.float32)


def _device_context(device: torch.device):
    return torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext()


@torch.no_grad()
def mert_features(model, processor, mono24k: np.ndarray, interrupt: Optional[Callable[[], None]] = None) -> np.ndarray:
    """MERT-v2-FullSong layer-20 features at 25 Hz, float16 [T, 1024]; 30-second chunks like the head's training data."""
    device = next(model.parameters()).device
    chunks = [mono24k[start:start + MERT_CHUNK] for start in range(0, len(mono24k), MERT_CHUNK)]
    chunks = [chunk for chunk in chunks if len(chunk) >= MERT_RATE]
    if not chunks:
        raise ValueError("audio must be at least one second long")
    features = []
    fallbacks = 0
    for chunk in chunks:
        if interrupt is not None:
            interrupt()
        inputs = {k: v.to(device) for k, v in processor([chunk], sampling_rate=MERT_RATE, return_tensors="pt").items()}
        with _device_context(device), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            value = model(**inputs, output_hidden_states=True).hidden_states[MERT_LAYER][0].float()
        if not torch.isfinite(value).all():
            # bf16 attention on a GPU that is not the process's current CUDA device can return NaN (seen with
            # torch 2.13 on a second GPU); fp32 never does. Recompute this chunk in fp32.
            fallbacks += 1
            with _device_context(device):
                value = model(**inputs, output_hidden_states=True).hidden_states[MERT_LAYER][0].float()
            if not torch.isfinite(value).all():
                raise RuntimeError("MERT features are not finite even in fp32")
        features.append(value.cpu())
    if fallbacks:
        logging.warning("YuE2 trainer: %d/%d MERT chunks produced NaN in bf16 and were recomputed in fp32", fallbacks, len(chunks))
    joined = torch.cat(features)
    frames = round(len(mono24k) / MERT_RATE * FRAMES_PER_SECOND)
    return F.interpolate(joined.T[None], size=frames, mode="linear", align_corners=False)[0].T.half().numpy()


class SemanticTokenizer:
    """MERT-v2-FullSong + the v4 head on one device. Use as a context manager or call ``close``."""

    def __init__(self, head_path, mert_path: str = DEFAULT_MERT, device="cuda"):
        from transformers import AutoFeatureExtractor, AutoModel
        self.device = torch.device(device)
        if self.device.type == "cuda":
            # Create the process's default CUDA context before using another GPU: with torch 2.13 the first
            # bf16 matmuls on a second GPU return NaN when the current device was never initialised.
            torch.empty(1, device=torch.device("cuda", torch.cuda.current_device()))
            torch.empty(1, device=self.device)
        t0 = time.perf_counter()
        source = str(mert_path)
        local = Path(source).expanduser()
        kwargs = {"local_files_only": True} if local.is_dir() else {}
        if local.is_dir():
            source = str(local)
        self.processor = AutoFeatureExtractor.from_pretrained(source, **kwargs)
        self.mert = AutoModel.from_pretrained(source, trust_remote_code=True, **kwargs).to(self.device).eval().requires_grad_(False)
        self.head = load_head(head_path, self.device)
        logging.info("YuE2 trainer: loaded MERT (%s) and tokenizer head (%s) on %s in %.1fs", source, Path(head_path).name,
                     self.device, time.perf_counter() - t0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.mert = self.head = self.processor = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def tokenize_wave(self, stereo48k: torch.Tensor, interrupt: Optional[Callable[[], None]] = None) -> np.ndarray:
        mono = resample(stereo48k.float().mean(0).cpu().numpy(), SAMPLE_RATE, MERT_RATE)
        return predict(self.head, mert_features(self.mert, self.processor, mono, interrupt))

    def tokenize_file(self, path, interrupt: Optional[Callable[[], None]] = None) -> np.ndarray:
        wave, sr = load_audio(str(path))
        return self.tokenize_wave(to_stereo_48k(wave, sr), interrupt)


def sidecar_path(item: Item) -> Optional[Path]:
    if not item.audio_path:
        return None
    audio = Path(item.audio_path)
    return audio.with_name(audio.stem + ".semantic.npy")


def tokenize_dataset(dataset: Dataset, tokenizer: SemanticTokenizer, force: bool = False, write_sidecars: bool = True,
                     cache_dir=None, progress: Optional[Callable[[int, int], None]] = None,
                     interrupt: Optional[Callable[[], None]] = None) -> dict:
    """Fill ``item.semantic`` for every audio item (in place) and write ``<stem>.semantic.npy`` sidecars.

    Items from YuE2 output folders already carry the exact tokens and are left alone. Existing sidecars are
    reused unless ``force``. Returns a summary dict.
    """
    done, reused, skipped, failed = [], [], [], []
    cache = Path(cache_dir) if cache_dir else None
    todo = [item for item in dataset.items if item.audio_path]
    for index, item in enumerate(todo):
        if interrupt is not None:
            interrupt()
        if item.source == "yue2_output" and item.semantic:
            skipped.append(item.id)
        elif item.semantic and not force and item.extra.get("semantic_source", SOURCE_TAG) == SOURCE_TAG:
            reused.append(item.id)
        else:
            target = sidecar_path(item)
            cached = cache / f"{cache_key(item, 'semantic|' + SOURCE_TAG)}.npy" if cache else None
            tokens = None
            if not force:
                for candidate in (target, cached):
                    if candidate is not None and candidate.is_file():
                        tokens = np.load(candidate, allow_pickle=False).astype(np.int64)
                        reused.append(item.id)
                        break
            if tokens is None:
                try:
                    t0 = time.perf_counter()
                    tokens = tokenizer.tokenize_file(item.audio_path, interrupt).astype(np.int64)
                    logging.info("YuE2 trainer: %s -> %d semantic tokens (%.1f s of audio) in %.0fs", item.id, len(tokens),
                                 len(tokens) / FRAMES_PER_SECOND, time.perf_counter() - t0)
                    wrote = None
                    if write_sidecars and target is not None:
                        try:
                            np.save(target, tokens.astype(np.int32))
                            wrote = target
                        except OSError as exc:
                            logging.warning("YuE2 trainer: cannot write %s (%s)", target, exc)
                    if wrote is None and cached is not None:
                        cached.parent.mkdir(parents=True, exist_ok=True)
                        np.save(cached, tokens.astype(np.int32))
                    done.append(item.id)
                except Exception as exc:  # noqa: BLE001 - one bad file should not kill the run
                    logging.warning("YuE2 trainer: semantic tokens failed for %s (%s)", item.id, exc)
                    failed.append(item.id)
                    tokens = None
            if tokens is not None:
                item.semantic = [int(t) for t in tokens]
                item.extra["semantic_source"] = SOURCE_TAG
                if item.frames is not None and abs(len(tokens) - item.frames) > 2 and len(tokens) < item.frames:
                    logging.warning("YuE2 trainer: %s has %d semantic tokens but %d latent frames", item.id, len(tokens), item.frames)
        if progress is not None:
            progress(index + 1, len(todo))
    return {"tokenized": done, "reused": reused, "skipped_true_tokens": skipped, "failed": failed, "source": SOURCE_TAG}


def summarize(summary: dict) -> str:
    return (f"semantic tokens ({summary['source']}): {len(summary['tokenized'])} tokenized, {len(summary['reused'])} reused, "
            f"{len(summary['skipped_true_tokens'])} with exact YuE2 tokens, {len(summary['failed'])} failed"
            + (": " + ", ".join(summary["failed"]) if summary["failed"] else ""))


__all__ = ["TokenHead", "load_head", "normalize", "windows", "predict", "mert_features", "SemanticTokenizer",
           "tokenize_dataset", "summarize", "sidecar_path", "HEAD_FILENAME", "DEFAULT_MERT", "SOURCE_TAG"]
