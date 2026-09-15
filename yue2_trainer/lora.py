"""LoRA adapter setup on ComfyUI modules, using ComfyUI's own weight adapters and bypass hooks.

The resulting state dict uses the *generic* key format that ``comfy/lora.py`` maps directly
onto module names, so the files load with the stock ``LoraLoader`` / ``LoraLoaderModelOnly``:

    diffusion_model.model.layers.0.self_attn.qkv_proj.lora_up.weight   (MODEL / acoustic path)
    text_encoders.model.layers.0.self_attn.qkv_proj.lora_up.weight     (CLIP / planner path)
"""
from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import re

import torch
import torch.nn as nn

from .constants import CLIP_KEY_PREFIX, MODEL_KEY_PREFIX

TARGET_PRESETS = {
    "attention": ("self_attn.qkv_proj", "self_attn.o_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "attention+mlp": ("self_attn.qkv_proj", "self_attn.o_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                      "mlp.gate_up_proj", "mlp.down_proj", "mlp.gate_proj", "mlp.up_proj"),
    "mlp": ("mlp.gate_up_proj", "mlp.down_proj", "mlp.gate_proj", "mlp.up_proj"),
}
LAYER_RE = re.compile(r"(^|\.)layers\.\d+\.")
# Extra 2D linears of the acoustic head (outside the transformer layers).
ACOUSTIC_EXTRA = ("vae2llm", "llm2vae", "time_embedder.mlp.0", "time_embedder.mlp.2")


def select_target_modules(root: nn.Module, preset: str, include_acoustic_extra: bool = False,
                          name_prefix: str = "") -> list[tuple[str, nn.Module]]:
    """Return (name, module) pairs of nn.Linear-like modules that should receive LoRA."""
    suffixes = TARGET_PRESETS[preset]
    selected = []
    for name, module in root.named_modules():
        weight = getattr(module, "weight", None)
        if weight is None or not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            continue
        if not name.startswith(name_prefix):
            continue
        rel = name[len(name_prefix):]
        if LAYER_RE.search(rel) and any(rel.endswith(s) for s in suffixes):
            selected.append((name, module))
        elif include_acoustic_extra and any(rel == s or rel.endswith("." + s) for s in ACOUSTIC_EXTRA):
            selected.append((name, module))
    if not selected:
        raise ValueError(f"No LoRA target modules found for preset '{preset}' under prefix '{name_prefix}'")
    return selected


@dataclass
class LoraSetup:
    lora_sd: dict            # key -> Parameter/Tensor (lora_up.weight, lora_down.weight, alpha)
    trainable: list          # parameters passed to the optimizer
    adapters: list
    manager: object          # comfy.weight_adapter.bypass.BypassInjectionManager
    injections: list = None

    def inject(self, patcher, root: nn.Module):
        self.injections = self.manager.create_injections(root)
        for injection in self.injections:
            injection.inject(patcher)

    def eject(self, patcher):
        if self.injections:
            for injection in self.injections:
                injection.eject(patcher)
        self.injections = None

    def export(self, dtype=torch.bfloat16) -> dict:
        out = {}
        for key, value in self.lora_sd.items():
            tensor = value.detach().to("cpu")
            if key.endswith(".alpha"):
                tensor = tensor.to(torch.float32)
            else:
                tensor = tensor.to(dtype).contiguous()
            out[key] = tensor
        return out


def create_lora(root: nn.Module, targets: list[tuple[str, nn.Module]], rank: int, alpha: float,
                lora_dtype=torch.float32, existing: Optional[dict] = None, save_prefix: str = "") -> LoraSetup:
    """Create trainable LoRA adapters for ``targets`` (names relative to ``root``)."""
    from comfy.weight_adapter.bypass import BypassInjectionManager
    from comfy.weight_adapter.lora import LoRAAdapter

    existing = existing or {}
    manager = BypassInjectionManager()
    lora_sd, trainable, adapters = {}, [], []
    resumed = 0
    for name, module in targets:
        key = f"{save_prefix}{name}"
        adapter = None
        if f"{key}.lora_up.weight" in existing:
            found_alpha = float(existing.get(f"{key}.alpha", torch.tensor(alpha)).item())
            loaded = LoRAAdapter.load(key, existing, found_alpha, None)
            if loaded is not None:
                adapter = loaded.to_train()
                resumed += 1
        if adapter is None:
            adapter = LoRAAdapter.create_train(module.weight, rank=rank, alpha=float(alpha))
        adapter = adapter.to(lora_dtype).train()
        for pname, param in adapter.named_parameters():
            full = f"{key}.{pname}"
            if pname == "alpha":
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
                trainable.append(param)
            lora_sd[full] = param
        manager.add_adapter(name, adapter, strength=1.0)
        adapters.append(adapter)
    if resumed:
        logging.info("YuE2 trainer: resumed %d/%d LoRA modules from existing weights", resumed, len(targets))
    logging.info("YuE2 trainer: LoRA on %d modules, rank %d, alpha %g, %d trainable tensors",
                 len(targets), rank, alpha, len(trainable))
    return LoraSetup(lora_sd=lora_sd, trainable=trainable, adapters=adapters, manager=manager)


def count_parameters(params) -> int:
    return sum(p.numel() for p in params)


def save_lora_file(lora_sd: dict, path, metadata: Optional[dict] = None):
    import safetensors.torch
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"yue2_trainer": json.dumps(metadata or {}, ensure_ascii=False)}
    safetensors.torch.save_file(lora_sd, str(path), metadata=meta)


def load_lora_file(path) -> dict:
    import safetensors.torch
    return safetensors.torch.load_file(str(path), device="cpu")


def lora_metadata(path) -> dict:
    """The trainer metadata stored in a LoRA file (empty for files from other tools)."""
    import safetensors
    with safetensors.safe_open(str(path), "pt") as handle:
        meta = handle.metadata() or {}
    try:
        return json.loads(meta.get("yue2_trainer", "{}"))
    except json.JSONDecodeError:
        return {}


def merge_lora_files(paths, out_path) -> tuple[dict, dict]:
    """Write one LoRA file holding every tensor of ``paths`` (an acoustic and a planner LoRA: their keys never
    overlap, so one Load LoRA node then applies both). Files that patch the same weight are refused."""
    merged, parts = {}, []
    for path in paths:
        sd = load_lora_file(path)
        overlap = sorted(set(sd) & set(merged))
        if overlap:
            raise ValueError(f"{Path(path).name} patches weights already in the merge ({overlap[0]}, ...): only LoRAs "
                             "for different parts of the model (acoustic + planner) can be merged")
        merged.update(sd)
        meta = lora_metadata(path)
        parts.append({"file": Path(path).name, "keys": len(sd), **{k: meta.get(k) for k in ("kind", "steps", "rank", "alpha", "save_name", "kept_step")}})
    metadata = {"kind": "merged", "parts": parts,
                "model_keys": sum(1 for k in merged if k.startswith(MODEL_KEY_PREFIX)),
                "clip_keys": sum(1 for k in merged if k.startswith(CLIP_KEY_PREFIX))}
    save_lora_file(merged, out_path, metadata)
    return merged, metadata


@contextlib.contextmanager
def adapter_weights_as(lora, dtype):
    """Temporarily run the LoRA adapters in ``dtype`` (ComfyUI's samplers feed the bypass hooks bf16 activations
    and do not autocast, while the trainable weights are fp32). The fp32 tensors are put back afterwards."""
    stash = [(param, param.data) for adapter in lora.adapters for param in adapter.parameters()]
    try:
        for param, data in stash:
            param.data = data.to(dtype)
        yield
    finally:
        for param, data in stash:
            param.data = data


__all__ = ["select_target_modules", "create_lora", "LoraSetup", "save_lora_file", "load_lora_file", "lora_metadata",
           "merge_lora_files", "adapter_weights_as", "count_parameters", "TARGET_PRESETS", "ACOUSTIC_EXTRA"]
