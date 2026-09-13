"""Prompt construction and AR-prefix key/value caching through ComfyUI's YuE2 CLIP."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from .constants import ABC_END, ABC_START, CODEC_OFFSET, EOD, INSTRUCTIONS, MUSIC_END, MUSIC_START, CONTEXT


def encode_text(clip, text: str) -> list[int]:
    return clip.tokenizer.tokenizer.encode(text).ids


def prompt_ids(clip, style: str, lyrics: str, cot: str) -> list[int]:
    """[EOD] + instruction/style/lyrics text (identical to the ComfyUI tokenizer's ``prefix`` minus ABC_START)."""
    prompt = f"{INSTRUCTIONS[cot]}\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n"
    return [EOD] + encode_text(clip, prompt)


def resolve_mode(mode: str, abc: Optional[str], has_chords: bool) -> str:
    """Pick the CoT mode for an item. ``auto`` follows the score: chords -> full, else melody, none -> off."""
    if not abc or not abc.strip():
        return "off"
    if mode == "auto":
        return "full" if has_chords else "melody"
    return mode


def music_prefix_ids(clip, style: str, lyrics: str, abc: Optional[str], cot: str) -> tuple[list[int], list[int]]:
    """Return (full prefix ending in MUSIC_START, abc_ids). Matches YuE2TEModel.encode_token_weights."""
    base = prompt_ids(clip, style, lyrics, cot)
    if cot == "off" or not abc:
        return base + [ABC_START, ABC_END, MUSIC_START], []
    abc_ids = encode_text(clip, abc)
    return base + [ABC_START] + abc_ids + [ABC_END, MUSIC_START], abc_ids


def abc_sequence(clip, style: str, lyrics: str, abc: str, cot: str) -> tuple[list[int], int]:
    """Planner target: prefix + ABC_START, then ABC tokens + ABC_END. Returns (ids, loss_start)."""
    base = prompt_ids(clip, style, lyrics, cot) + [ABC_START]
    return base + encode_text(clip, abc) + [ABC_END], len(base)


@dataclass
class PrefixCache:
    ids: list
    kv: torch.Tensor          # [layers, 2, 1, kv_heads, L, head_dim] on CPU
    ar_length: int            # RoPE position of the first NAR token

    @property
    def length(self) -> int:
        return int(self.kv.shape[-2])


def load_clip_for_prefill(clip):
    import comfy.model_management
    comfy.model_management.load_models_gpu([clip.patcher], force_full_load=True)
    device = clip.patcher.load_device
    clip.cond_stage_model.set_clip_options({"execution_device": device})
    return device


@torch.no_grad()
def compute_prefix_kv(clip, ids: list[int], device=None) -> torch.Tensor:
    """Run the AR prefill once and return every layer's rotary-encoded K/V as [L, 2, 1, kvh, S, hd] (CPU)."""
    import comfy.model_management
    from comfy.text_encoders.llama import FixedKV
    te = clip.cond_stage_model
    if device is None:
        device = load_clip_for_prefill(clip)
    dtype = torch.bfloat16 if comfy.model_management.should_use_bf16(device) else torch.float32
    if len(ids) > te.config.max_position_embeddings:
        raise ValueError(f"Prefix of {len(ids)} tokens exceeds the model context")
    _, cache, _ = te._prefill([list(ids)], len(ids), dtype)
    layers = []
    for entry in cache:
        if isinstance(entry, FixedKV):
            key, value = entry.key[:, :len(ids)].transpose(1, 2), entry.value[:, :len(ids)].transpose(1, 2)
        else:
            key, value, _ = entry
            key, value = key[:, :, :len(ids)], value[:, :, :len(ids)]
        layers.append(torch.stack((key, value), dim=0))
    out = torch.stack(layers, dim=0).to("cpu", dtype=torch.bfloat16).contiguous()
    del cache
    return out


def build_acoustic_prefix(clip, style: str, lyrics: str, abc: Optional[str], cot: str,
                          semantic: Optional[list[int]] = None, chunk: Optional[tuple[int, int]] = None,
                          total_frames: Optional[int] = None, device=None) -> PrefixCache:
    """Cache the AR prefix for the acoustic model.

    Without semantic tokens the model is conditioned in its codec-dropout ("text-only") mode:
    only the text/ABC prefix is visible, but the NAR tokens keep the RoPE positions they would
    have after ``total_frames`` codec tokens, exactly as the reference implementation masks them.
    With semantic tokens the prefix is the full inference prefix for ``chunk`` = (start, end).
    """
    prefix, _ = music_prefix_ids(clip, style, lyrics, abc, cot)
    if semantic is not None:
        start, end = chunk if chunk is not None else (0, len(semantic))
        ids = prefix + [int(t) + CODEC_OFFSET for t in semantic[start:end]] + [MUSIC_END]
        ar_length = len(ids)
    else:
        ids = prefix
        frames = total_frames if total_frames is not None else 0
        ar_length = len(prefix) + frames + 1
    if ar_length + 2 > CONTEXT:
        logging.warning("YuE2 trainer: prefix positions exceed the context (%d); long songs are clipped", ar_length)
    kv = compute_prefix_kv(clip, ids, device=device)
    return PrefixCache(ids=ids, kv=kv, ar_length=ar_length)


__all__ = ["encode_text", "prompt_ids", "music_prefix_ids", "abc_sequence", "resolve_mode", "PrefixCache",
           "compute_prefix_kv", "build_acoustic_prefix", "load_clip_for_prefill"]
