"""Resumable training state: optimizer, random generators, step count and the loss history.

A LoRA checkpoint written by ``save_every`` (or by the save node at the end) gets a sibling
``<name>.resume`` file. Training again from that LoRA with ``resume_state`` restores the
optimizer moments, every replica's random generators and the step count, so the run
continues exactly where it stopped (``steps`` is then the total length of the run).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

STATE_SUFFIX = ".resume"       # not a ComfyUI checkpoint extension, so it stays out of the LoRA dropdown
STATE_VERSION = 1
_MATCH_KEYS = ("rank", "alpha", "targets", "optimizer")


def state_path(lora_path) -> Path:
    """``models/loras/name_000025.safetensors`` -> ``models/loras/name_000025.resume``."""
    path = Path(lora_path)
    return path.with_name(path.name[: -len(path.suffix)] + STATE_SUFFIX if path.suffix else path.name + STATE_SUFFIX)


def capture_state(kind: str, cfg, step: int, optimizer, replicas, losses: list, evals: list,
                  extra: Optional[dict] = None) -> dict:
    """Everything needed to continue a run after ``step`` optimizer steps (CPU tensors only)."""
    optim = optimizer.state_dict()
    state = {
        "version": STATE_VERSION,
        "kind": kind,
        "step": int(step),
        "config": {key: getattr(cfg, key) for key in _MATCH_KEYS},
        "optimizer": _to_cpu(optim),
        "generators": [r.generator.get_state().clone() for r in replicas],
        "py_rng": [r.extra["py_rng"].getstate() for r in replicas],
        "losses": [float(v) for v in losses],
        "evals": [[int(s), float(v)] for s, v in evals],
    }
    for key, value in (extra or {}).items():
        state[key] = value
    return state


def restore_state(state: dict, kind: str, cfg, optimizer, replicas) -> Optional[dict]:
    """Load ``state`` into ``optimizer`` and the replicas' generators.

    Returns the state (so the caller can pick up ``step``, ``losses`` and ``evals``) or ``None``
    when it does not belong to this trainer/configuration, in which case training starts fresh
    from the LoRA weights alone.
    """
    if not state:
        return None
    if state.get("kind") != kind:
        logging.warning("YuE2 trainer: resume state is for the %s trainer, not %s; starting fresh", state.get("kind"), kind)
        return None
    mismatch = {k: (state.get("config", {}).get(k), getattr(cfg, k)) for k in _MATCH_KEYS
                if state.get("config", {}).get(k) != getattr(cfg, k)}
    if mismatch:
        logging.warning("YuE2 trainer: resume state was trained with %s; starting fresh from the LoRA weights",
                        ", ".join(f"{k}={a!r} (now {b!r})" for k, (a, b) in mismatch.items()))
        return None
    try:
        optimizer.load_state_dict(state["optimizer"])
    except (ValueError, KeyError, RuntimeError) as exc:
        logging.warning("YuE2 trainer: optimizer state does not match the LoRA (%s); starting fresh", exc)
        return None
    for index, replica in enumerate(replicas):
        if index < len(state.get("generators", [])):
            replica.generator.set_state(state["generators"][index].to("cpu"))
        if index < len(state.get("py_rng", [])):
            replica.extra["py_rng"].setstate(_rng_tuple(state["py_rng"][index]))
    return state


def _rng_tuple(value):
    """random.Random.getstate() is (version, tuple_of_ints, gauss_next); torch.save round-trips tuples as lists."""
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return (value[0], tuple(int(v) for v in value[1]), value[2])
    return value


def _to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().to("cpu")
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(v) for v in value)
    return value


def save_state(state: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    return path


def load_state(path) -> Optional[dict]:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        logging.warning("YuE2 trainer: cannot read resume state %s (%s); starting fresh", path, exc)
        return None


__all__ = ["STATE_SUFFIX", "state_path", "capture_state", "restore_state", "save_state", "load_state"]
