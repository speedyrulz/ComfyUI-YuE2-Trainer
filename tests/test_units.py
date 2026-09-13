"""Unit tests that do not need ComfyUI or the checkpoint:  python -m pytest tests/test_units.py"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yue2_trainer import forward as F  # noqa: E402
from yue2_trainer.dataset import cache_key, load_cache, save_cache, scan_folder  # noqa: E402
from yue2_trainer.lora import select_target_modules  # noqa: E402
from yue2_trainer.prefix import resolve_mode  # noqa: E402


def _wav(path, seconds=1.0, sr=48000):
    import soundfile as sf
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    sf.write(path, np.stack([np.sin(2 * np.pi * 220 * t), np.sin(2 * np.pi * 330 * t)], 1).astype("float32"), sr)


def test_scan_folder_sidecars(tmp_path):
    _wav(tmp_path / "a.wav")
    (tmp_path / "a.style.txt").write_text("rock", encoding="utf-8")
    (tmp_path / "a.lyrics.txt").write_text("[Verse]\nhi", encoding="utf-8")
    (tmp_path / "a.abc").write_text('X:1\nK:C\n"C"C D E F|', encoding="utf-8")
    np.save(tmp_path / "a.semantic.npy", np.arange(25))
    _wav(tmp_path / "b.wav")
    (tmp_path / "b.json").write_text(json.dumps({"style": "jazz", "lyrics": "la"}), encoding="utf-8")
    _wav(tmp_path / "c.wav")
    out = tmp_path / "yue2_out"
    out.mkdir()
    (out / "request.json").write_text(json.dumps({"id": "gen1", "style": "pop", "lyrics": "x", "cot": "full"}), encoding="utf-8")
    np.save(out / "semantic.npy", np.arange(50, dtype=np.int32))
    np.save(out / "latent.npy", np.zeros((50, 64), dtype=np.float32))
    (out / "score.abc").write_text("X:1\nK:C\nC D|", encoding="utf-8")
    _wav(out / "audio.flac", 2.0)

    ds = scan_folder(tmp_path, default_style="default", default_lyrics="")
    by_id = {i.id: i for i in ds.items}
    assert set(by_id) == {"a", "b", "c", "gen1"}
    assert by_id["a"].style == "rock" and by_id["a"].lyrics == "[Verse]\nhi" and by_id["a"].has_chords()
    assert by_id["a"].semantic == list(range(25))
    assert by_id["b"].style == "jazz" and by_id["b"].abc is None
    assert by_id["c"].style == "default"
    gen = by_id["gen1"]
    assert gen.latents.shape == (64, 50) and gen.semantic == list(range(50)) and gen.abc.startswith("X:1")
    assert gen.audio_path.endswith("audio.flac") and gen.seconds == 2.0


def test_cache_roundtrip(tmp_path):
    _wav(tmp_path / "a.wav")
    ds = scan_folder(tmp_path)
    key = cache_key(ds.items[0], "v1")
    save_cache(tmp_path / "cache", key, {"latents": torch.zeros(64, 3), "abc_melody": "X:1"})
    got = load_cache(tmp_path / "cache", key)
    assert got["latents"].shape == (64, 3) and got["abc_melody"] == "X:1"
    assert load_cache(tmp_path / "cache", "missing") is None


def test_resolve_mode():
    assert resolve_mode("auto", None, False) == "off"
    assert resolve_mode("auto", "X:1", False) == "melody"
    assert resolve_mode("auto", 'X:1 "C"', True) == "full"
    assert resolve_mode("melody", 'X:1 "C"', True) == "melody"
    assert resolve_mode("full", "", False) == "off"


class _Attn(nn.Module):
    def __init__(self, h, nh, nkv, hd):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = nh, nkv, hd
        self.inner_size, self.kv_size = nh * hd, nkv * hd
        self.merged_qkv = True
        self.qkv_proj = nn.Linear(h, self.inner_size + 2 * self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.inner_size, h, bias=False)
        self.q_norm = self.k_norm = None


class _Mlp(nn.Module):
    def __init__(self, h, i):
        super().__init__()
        self.merged_mlp = True
        self.gate_up_proj = nn.Linear(h, 2 * i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)
        self.activation = torch.nn.functional.silu


class _Block(nn.Module):
    def __init__(self, h=32, nh=4, nkv=2, hd=8, i=48):
        super().__init__()
        self.self_attn = _Attn(h, nh, nkv, hd)
        self.mlp = _Mlp(h, i)
        self.input_layernorm = nn.RMSNorm(h)
        self.post_attention_layernorm = nn.RMSNorm(h)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block() for _ in range(2)])
        self.vae2llm = nn.Linear(64, 32)
        self.llm2vae = nn.Linear(32, 64)


def test_select_target_modules():
    root = nn.Module()
    root.diffusion_model = _Model()
    names = [n for n, _ in select_target_modules(root, "attention", name_prefix="diffusion_model.")]
    assert names == ["diffusion_model.layers.0.self_attn.qkv_proj", "diffusion_model.layers.0.self_attn.o_proj",
                     "diffusion_model.layers.1.self_attn.qkv_proj", "diffusion_model.layers.1.self_attn.o_proj"]
    names = [n for n, _ in select_target_modules(root, "attention+mlp", include_acoustic_extra=True,
                                                 name_prefix="diffusion_model.")]
    assert "diffusion_model.layers.0.mlp.gate_up_proj" in names and "diffusion_model.vae2llm" in names
    assert "diffusion_model.llm2vae" in names and len(names) == 10


def test_rope_matches_reference_rotation():
    torch.manual_seed(0)
    x = torch.randn(1, 2, 5, 8)
    cos, sin = F.rope_cos_sin(8, torch.arange(3, 8)[None], 10000.0, x.device)
    out = F.apply_rope(x, cos, sin)
    # reference: HF-style half rotation
    half = 4
    x1, x2 = x[..., :half], x[..., half:]
    c, s = cos[..., :half], sin[..., :half]
    ref = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
    assert torch.allclose(out, ref, atol=1e-6)
    # position 0 is the identity
    cos0, sin0 = F.rope_cos_sin(8, torch.zeros(1, 5, dtype=torch.long), 10000.0, x.device)
    assert torch.allclose(F.apply_rope(x, cos0, sin0), x)


def test_causal_layers_are_causal_and_prefix_is_visible():
    torch.manual_seed(0)
    model = _Model()
    x = torch.randn(1, 6, 32)
    cos, sin = F.rope_cos_sin(8, torch.arange(6)[None], 10000.0, x.device)
    y = F.run_layers(model.layers, x, cos, sin, None, causal=True, checkpointing=False)
    x2 = x.clone()
    x2[:, 4:] += 1.0                      # perturb the future
    y2 = F.run_layers(model.layers, x2, cos, sin, None, causal=True, checkpointing=False)
    assert torch.allclose(y[:, :4], y2[:, :4], atol=1e-5)
    assert not torch.allclose(y[:, 4:], y2[:, 4:])
    # prefix K/V change the output when present
    pkv = torch.randn(2, 2, 1, 2, 3, 8)
    y3 = F.run_layers(model.layers, x, cos, sin, pkv, causal=False, checkpointing=False)
    assert y3.shape == y.shape and not torch.allclose(y3, y)


def test_gradient_checkpointing_matches_plain():
    torch.manual_seed(0)
    model = _Model()
    x = torch.randn(1, 6, 32, requires_grad=True)
    cos, sin = F.rope_cos_sin(8, torch.arange(6)[None], 10000.0, x.device)
    y1 = F.run_layers(model.layers, x, cos, sin, None, causal=True, checkpointing=False)
    y2 = F.run_layers(model.layers, x, cos, sin, None, causal=True, checkpointing=True)
    assert torch.allclose(y1, y2, atol=1e-6)
    (g1,) = torch.autograd.grad(y1.sum(), x, retain_graph=True)
    (g2,) = torch.autograd.grad(y2.sum(), x)
    assert torch.allclose(g1, g2, atol=1e-6)


def test_chunked_cross_entropy():
    torch.manual_seed(0)
    head = nn.Linear(16, 40, bias=False)
    h = torch.randn(10, 16, requires_grad=True)
    t = torch.randint(0, 40, (10,))
    ref = torch.nn.functional.cross_entropy(head(h), t)
    got = F.chunked_cross_entropy(head, h, t, chunk=3)
    assert torch.allclose(ref, got, atol=1e-5)
    got.backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()
