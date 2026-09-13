"""Training items: audio + style + lyrics (+ optional ABC score / semantic tokens / latents).

A dataset folder is scanned for audio files. Per-file metadata comes from sidecars that
share the audio file's stem:

    song.mp3
    song.json          {"style": ..., "lyrics": ..., "abc": ..., "semantic": [...]}   (all optional)
    song.style.txt     style / tags prompt
    song.lyrics.txt    lyrics with [Verse]/[Chorus] tags   (song.txt is accepted too)
    song.abc           ABC score (melody or full)
    song.semantic.npy  YuE2 semantic codec tokens (int, 25 per second)

Directories written by the native ``yue2`` runtime (``save_artifacts``) are also accepted:
they contain request.json, audio.flac, score.abc, semantic.npy and latent.npy, so no VAE
pass is needed and the exact semantic tokens are available.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .constants import AUDIO_EXTENSIONS, CODEC_SIZE, FRAMES_PER_SECOND

CHORD_RE = re.compile(r'"[A-G][^"]*"')


@dataclass
class Item:
    id: str
    audio_path: Optional[str]
    style: str
    lyrics: str
    abc: Optional[str] = None
    semantic: Optional[list] = None
    latents: Optional[torch.Tensor] = None   # [64, T] float16/float32 on CPU
    seconds: Optional[float] = None
    source: str = "file"
    extra: dict = field(default_factory=dict)

    @property
    def frames(self) -> Optional[int]:
        return None if self.latents is None else int(self.latents.shape[-1])

    def has_chords(self) -> bool:
        return bool(self.abc) and CHORD_RE.search(self.abc) is not None

    def summary(self) -> str:
        parts = [self.id]
        if self.seconds is not None:
            parts.append(f"{self.seconds:.1f}s")
        if self.latents is not None:
            parts.append(f"latents={self.frames}f")
        parts.append("abc" if self.abc else "no-abc")
        if self.semantic is not None:
            parts.append(f"semantic={len(self.semantic)}")
        return " | ".join(parts)


@dataclass
class Dataset:
    items: list
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.items)

    def with_latents(self):
        return [item for item in self.items if item.latents is not None]

    def describe(self) -> str:
        lines = [f"{len(self.items)} items"]
        lines += ["  " + item.summary() for item in self.items]
        return "\n".join(lines)


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def _semantic_from(value, base: Path) -> Optional[list]:
    if value is None:
        return None
    if isinstance(value, str):
        p = Path(value)
        if not p.is_absolute():
            p = base / p
        value = np.load(p, allow_pickle=False)
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return None
    tokens = [int(t) for t in arr.tolist()]
    if min(tokens) < 0 or max(tokens) >= CODEC_SIZE:
        raise ValueError(f"semantic tokens outside 0..{CODEC_SIZE - 1}")
    return tokens


def _item_from_audio(audio: Path, default_style: str, default_lyrics: str) -> Item:
    stem = audio.with_suffix("")
    style = default_style
    lyrics = default_lyrics
    abc = None
    semantic = None
    extra = {}
    meta_path = Path(str(stem) + ".json")
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        style = meta.get("style", meta.get("tags", style)) or style
        lyrics = meta.get("lyrics", lyrics) or lyrics
        abc = meta.get("abc", abc)
        if isinstance(abc, str) and abc.endswith(".abc") and "\n" not in abc:
            abc = _read_text(audio.parent / abc)
        semantic = _semantic_from(meta.get("semantic"), audio.parent)
        extra = {k: v for k, v in meta.items() if k not in {"style", "tags", "lyrics", "abc", "semantic"}}
    for name in (".style.txt", ".tags.txt"):
        text = _read_text(Path(str(stem) + name))
        if text:
            style = text
            break
    for name in (".lyrics.txt", ".txt"):
        text = _read_text(Path(str(stem) + name))
        if text is not None:
            lyrics = text
            break
    text = _read_text(Path(str(stem) + ".abc"))
    if text:
        abc = text
    sem_path = Path(str(stem) + ".semantic.npy")
    if sem_path.is_file():
        semantic = _semantic_from(str(sem_path), audio.parent)
    return Item(id=audio.stem, audio_path=str(audio), style=style, lyrics=lyrics, abc=abc,
                semantic=semantic, extra=extra)


def _item_from_yue2_output(directory: Path) -> Optional[Item]:
    request = directory / "request.json"
    audio = directory / "audio.flac"
    if not request.is_file():
        return None
    meta = json.loads(request.read_text(encoding="utf-8"))
    abc = _read_text(directory / "score.abc")
    semantic = None
    if (directory / "semantic.npy").is_file():
        semantic = _semantic_from(str(directory / "semantic.npy"), directory)
    latents = None
    if (directory / "latent.npy").is_file():
        arr = np.load(directory / "latent.npy", allow_pickle=False)
        if arr.ndim == 2 and arr.shape[1] == 64:
            arr = arr.T
        latents = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float16)
    item = Item(id=meta.get("id") or directory.name, audio_path=str(audio) if audio.is_file() else None,
                style=meta.get("style", ""), lyrics=meta.get("lyrics", ""), abc=abc, semantic=semantic,
                latents=latents, source="yue2_output")
    if latents is not None:
        item.seconds = latents.shape[-1] / FRAMES_PER_SECOND
    return item


def scan_folder(folder, default_style: str = "", default_lyrics: str = "", recursive: bool = True) -> Dataset:
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")
    items = []
    seen_dirs = set()
    walker = folder.rglob("*") if recursive else folder.glob("*")
    for path in sorted(walker):
        if path.is_dir():
            if (path / "request.json").is_file() and path not in seen_dirs:
                item = _item_from_yue2_output(path)
                if item is not None:
                    items.append(item)
                    seen_dirs.add(path)
            continue
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        if path.parent in seen_dirs or (path.name == "audio.flac" and (path.parent / "request.json").is_file()):
            continue
        items.append(_item_from_audio(path, default_style, default_lyrics))
    if (folder / "request.json").is_file() and folder not in seen_dirs:
        item = _item_from_yue2_output(folder)
        if item is not None:
            items.append(item)
    if not items:
        raise ValueError(f"No audio files or YuE2 output directories found in {folder}")
    for item in items:
        if not item.style:
            logging.warning("YuE2 trainer: item %s has an empty style prompt", item.id)
    return Dataset(items=items, meta={"folder": str(folder), "default_style": default_style,
                                      "default_lyrics": default_lyrics})


def cache_key(item: Item, tag: str) -> str:
    h = hashlib.sha1()
    h.update(tag.encode())
    if item.audio_path:
        st = os.stat(item.audio_path)
        h.update(f"{item.audio_path}|{st.st_size}|{int(st.st_mtime)}".encode("utf-8"))
    else:
        h.update(item.id.encode("utf-8"))
    return h.hexdigest()[:24]


def save_cache(cache_dir, key: str, payload: dict):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / f"{key}.pt.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, cache_dir / f"{key}.pt")


def load_cache(cache_dir, key: str) -> Optional[dict]:
    path = Path(cache_dir) / f"{key}.pt"
    if not path.is_file():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001
        logging.warning("YuE2 trainer: ignoring unreadable cache %s (%s)", path, e)
        return None


def clone_dataset(dataset: Dataset) -> Dataset:
    """Shallow copy: items are new objects, tensors are shared."""
    return Dataset(items=[Item(**dict(item.__dict__)) for item in dataset.items], meta=dict(dataset.meta))


__all__ = ["Item", "Dataset", "scan_folder", "cache_key", "save_cache", "load_cache", "clone_dataset", "CHORD_RE"]
