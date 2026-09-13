"""Differentiable forward passes built from ComfyUI's YuE2 modules.

ComfyUI's ``TransformerBlock.forward`` writes its output in-place into the input tensor
(``torch.add(..., out=output)``), which autograd rejects, and its attention helper may be
a non-differentiable kernel (sage / flash decode). This module re-implements the block
math with the *same* submodules (so bypass-LoRA hooks on the Linear layers apply) using
plain PyTorch ops that support backward.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint


def rope_cos_sin(head_dim: int, positions: torch.Tensor, theta: float, device, dtype=torch.float32):
    """positions [1, S] -> cos, sin each [1, 1, S, head_dim] (halves duplicated)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = positions.float()[..., None] * inv_freq[None, None, :]          # [1, S, hd/2]
    emb = torch.cat((freqs, freqs), dim=-1)                                 # [1, S, hd]
    return emb.cos()[:, None].to(dtype), emb.sin()[:, None].to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B, H, S, hd]; identical to comfy.text_encoders.llama.apply_rope and YuE2 _apply_rotary."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos.to(x.dtype) + rotated * sin.to(x.dtype))


def attention_forward(attn, h: torch.Tensor, cos, sin, prefix_kv=None, causal: bool = False) -> torch.Tensor:
    B, S, _ = h.shape
    if getattr(attn, "merged_qkv", False):
        q, k, v = attn.qkv_proj(h).split((attn.inner_size, attn.kv_size, attn.kv_size), dim=-1)
    else:
        q, k, v = attn.q_proj(h), attn.k_proj(h), attn.v_proj(h)
    q = q.view(B, S, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = k.view(B, S, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
    v = v.view(B, S, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
    if attn.q_norm is not None:
        q = attn.q_norm(q)
    if attn.k_norm is not None:
        k = attn.k_norm(k)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    if prefix_kv is not None:
        pk, pv = prefix_kv
        k = torch.cat((pk.to(k.dtype), k), dim=2)
        v = torch.cat((pv.to(v.dtype), v), dim=2)
    if attn.num_heads != attn.num_kv_heads:
        groups = attn.num_heads // attn.num_kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    out = out.transpose(1, 2).reshape(B, S, attn.num_heads * attn.head_dim)
    return attn.o_proj(out)


def mlp_forward(mlp, h: torch.Tensor) -> torch.Tensor:
    if getattr(mlp, "merged_mlp", False):
        gate, up = mlp.gate_up_proj(h).chunk(2, dim=-1)
    else:
        gate, up = mlp.gate_proj(h), mlp.up_proj(h)
    return mlp.down_proj(mlp.activation(gate) * up)


def block_forward(layer, x: torch.Tensor, cos, sin, pk=None, pv=None, causal: bool = False) -> torch.Tensor:
    prefix = None if pk is None else (pk, pv)
    x = x + attention_forward(layer.self_attn, layer.input_layernorm(x), cos, sin, prefix, causal)
    x = x + mlp_forward(layer.mlp, layer.post_attention_layernorm(x))
    return x


def run_layers(layers, x: torch.Tensor, cos, sin, prefix_kv: Optional[torch.Tensor] = None,
               causal: bool = False, checkpointing: bool = True) -> torch.Tensor:
    """prefix_kv: [num_layers, 2, B, kv_heads, L, head_dim] or None."""
    for index, layer in enumerate(layers):
        pk = pv = None
        if prefix_kv is not None:
            pk, pv = prefix_kv[index, 0], prefix_kv[index, 1]
        if checkpointing and torch.is_grad_enabled():
            x = torch.utils.checkpoint.checkpoint(block_forward, layer, x, cos, sin, pk, pv, causal, use_reentrant=False)
        else:
            x = block_forward(layer, x, cos, sin, pk, pv, causal)
    return x


# ── Acoustic (NAR flow matching) ──────────────────────────────────────────────

def nar_forward(dm, x_t: torch.Tensor, sigma: torch.Tensor, prefix_kv: torch.Tensor, ar_length: int,
                frame_offset: int = 0, checkpointing: bool = True) -> torch.Tensor:
    """Velocity prediction of ComfyUI's YuE2 acoustic model for one chunk.

    dm: comfy.ldm.yue2.model.YuE2 (model_patcher.model.diffusion_model)
    x_t: [B, 64, T] noisy latents; sigma: [B] flow time (1 = pure noise)
    prefix_kv: [layers, 2, B, kv_heads, L, head_dim] cached AR prefix keys/values
    ar_length: RoPE position of the first NAR token (= AR prefix length incl. codec tokens)
    frame_offset: absolute frame index of x_t[..., 0] inside the chunk (for training crops)
    Returns v [B, 64, T].
    """
    config = dm.config
    B, _, T = x_t.shape
    length = T + 2
    state = F.pad(x_t.transpose(1, 2), (0, 0, 1, 1))                      # [B, T+2, 64]
    time = dm.time_embedder(sigma.to(state.dtype), state.dtype)[:, None]  # [B, 1, H]
    pe = dm.latent_pos_embed.pe[frame_offset: frame_offset + length]
    state = dm.vae2llm(state) + time + pe.to(dtype=state.dtype, device=state.device)[None]
    positions = torch.arange(ar_length + frame_offset, ar_length + frame_offset + length, device=state.device)[None]
    cos, sin = rope_cos_sin(config.head_dim, positions, config.rope_theta, state.device)
    state = run_layers(dm.model.layers, state, cos, sin, prefix_kv, causal=False, checkpointing=checkpointing)
    return dm.llm2vae(dm.model.norm(state))[:, 1:-1].transpose(1, 2)


# ── Planner / semantic (AR next-token) ────────────────────────────────────────

def ar_hidden(llm, ids: torch.Tensor, dtype, checkpointing: bool = True) -> torch.Tensor:
    """Causal hidden states (after final norm) of comfy.text_encoders.llama.Llama2_ for ids [B, S]."""
    config = llm.config
    x = llm.embed_tokens(ids, out_dtype=dtype)
    positions = torch.arange(ids.shape[1], device=ids.device)[None]
    cos, sin = rope_cos_sin(config.head_dim, positions, config.rope_theta, ids.device)
    x = run_layers(llm.layers, x, cos, sin, None, causal=True, checkpointing=checkpointing)
    return llm.norm(x) if llm.norm is not None else x


def chunked_cross_entropy(lm_head, hidden: torch.Tensor, targets: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    """Mean CE over ``hidden`` [N, H] -> ``targets`` [N] without materialising all logits."""
    def piece(h, t):
        return F.cross_entropy(lm_head(h).float(), t, reduction="sum")
    total = hidden.new_zeros((), dtype=torch.float32)
    n = hidden.shape[0]
    for start in range(0, n, chunk):
        h, t = hidden[start:start + chunk], targets[start:start + chunk]
        if torch.is_grad_enabled():
            total = total + torch.utils.checkpoint.checkpoint(piece, h, t, use_reentrant=False)
        else:
            total = total + piece(h, t)
    return total / max(1, n)


__all__ = ["rope_cos_sin", "apply_rope", "attention_forward", "mlp_forward", "block_forward", "run_layers",
           "nar_forward", "ar_hidden", "chunked_cross_entropy"]
