"""Device selection and simple data-parallel LoRA training across several GPUs.

Each GPU holds a full replica of the (frozen) base model (a deep-copied ModelPatcher loaded
through ComfyUI's model management with its own load device) plus its own copy of the LoRA
adapters. Every optimizer step, each replica processes a share of the micro-batches in its
own thread, LoRA gradients are summed onto the primary replica, the optimizer steps there,
and the updated LoRA weights are broadcast back. Only the LoRA parameters (a few million
values) cross the PCIe bus, so two GPUs give close to double throughput.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn


def available_cuda_devices() -> list[str]:
    if not torch.cuda.is_available():
        return []
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]


def device_choices() -> list[str]:
    """Options for a node combo: auto, each GPU, and 'all' (data parallel on every GPU)."""
    cuda = available_cuda_devices()
    return ["auto"] + cuda + (["all"] if len(cuda) > 1 else [])


def resolve_devices(spec: str) -> list[torch.device]:
    """'auto' -> ComfyUI's default device; 'cuda:1' -> that GPU; 'all' -> every GPU; 'cuda:0,cuda:1' -> those."""
    import comfy.model_management
    spec = (spec or "auto").strip().lower()
    if spec in ("", "auto", "default"):
        return [torch.device(comfy.model_management.get_torch_device())]
    if spec == "all":
        cuda = available_cuda_devices()
        if not cuda:
            return [torch.device(comfy.model_management.get_torch_device())]
        return [torch.device(d) for d in cuda]
    devices = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if part.isdigit():
            part = f"cuda:{part}"
        dev = torch.device(part)
        if dev.type == "cuda" and (dev.index or 0) >= torch.cuda.device_count():
            raise ValueError(f"Device {part} does not exist ({torch.cuda.device_count()} CUDA devices available)")
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        devices.append(dev)
    if not devices:
        raise ValueError(f"No devices in '{spec}'")
    # de-duplicate, keep order
    seen, out = set(), []
    for dev in devices:
        if dev not in seen:
            seen.add(dev)
            out.append(dev)
    return out


@dataclass
class Replica:
    index: int
    device: torch.device
    root: nn.Module                 # module the LoRA target names are relative to
    lora: object = None             # yue2_trainer.lora.LoraSetup
    generator: torch.Generator = None
    extra: dict = field(default_factory=dict)


def sync_lora_weights(replicas: list[Replica]):
    """Copy the primary replica's LoRA weights to the others."""
    primary = replicas[0]
    with torch.no_grad():
        for other in replicas[1:]:
            for p0, pk in zip(primary.lora.trainable, other.lora.trainable):
                pk.data.copy_(p0.data.to(pk.device, non_blocking=True))
    for other in replicas[1:]:
        if other.device.type == "cuda":
            torch.cuda.synchronize(other.device)


def reduce_gradients(replicas: list[Replica]):
    """Sum LoRA gradients of all replicas into the primary's parameters (losses are pre-scaled)."""
    primary = replicas[0]
    with torch.no_grad():
        for other in replicas[1:]:
            for p0, pk in zip(primary.lora.trainable, other.lora.trainable):
                if pk.grad is None:
                    continue
                g = pk.grad.to(p0.device, non_blocking=True)
                if p0.grad is None:
                    p0.grad = g.clone()
                else:
                    p0.grad.add_(g)
                pk.grad = None
    if primary.device.type == "cuda":
        torch.cuda.synchronize(primary.device)


def split_counts(total: int, parts: int) -> list[int]:
    base, rem = divmod(total, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def run_on_replicas(replicas: list[Replica], work: Callable[[Replica, int], torch.Tensor], counts: list[int]) -> float:
    """Run ``work(replica, n_micro)`` on every replica concurrently (one thread per GPU); return summed loss."""
    jobs = [(r, n) for r, n in zip(replicas, counts) if n > 0]
    if len(jobs) == 1:
        replica, n = jobs[0]
        return float(work(replica, n).item())

    errors = []
    results = [None] * len(jobs)

    def runner(slot, replica, n):
        try:
            if replica.device.type == "cuda":
                torch.cuda.set_device(replica.device)
            results[slot] = work(replica, n)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=len(jobs), thread_name_prefix="yue2-train") as pool:
        futures = [pool.submit(runner, slot, r, n) for slot, (r, n) in enumerate(jobs)]
        for future in futures:
            future.result()
    if errors:
        raise errors[0]
    return float(sum(r.float().cpu() for r in results if r is not None).item())


def free_replicas(replicas: list[Replica]):
    for replica in replicas[1:]:
        replica.root = None
        replica.lora = None
    for dev in {r.device for r in replicas}:
        if dev.type == "cuda":
            with torch.cuda.device(dev):
                torch.cuda.empty_cache()


__all__ = ["available_cuda_devices", "device_choices", "resolve_devices", "Replica",
           "sync_lora_weights", "reduce_gradients", "split_counts", "run_on_replicas", "free_replicas"]
