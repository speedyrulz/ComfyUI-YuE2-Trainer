"""Verification against ComfyUI's own YuE2 implementation (needs the checkpoint + a ComfyUI checkout).

    python tests/verify_against_comfy.py --comfy-root <ComfyUI> --models-root <ComfyUI>/models

Checks:
  1. acoustic training forward == comfy.ldm.yue2.model.YuE2.forward (same prefix KV, no LoRA)
  2. planner training forward  == comfy Llama2_ forward + lm_head
  3. LoRA state-dict keys map onto MODEL / CLIP through comfy.lora and load without warnings
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from train_cli import bootstrap_comfy, load_checkpoint, resolve_model_file  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", required=True)
    parser.add_argument("--models-root", action="append", default=[])
    parser.add_argument("--checkpoint", default="yue2_3b_bf16.safetensors")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    bootstrap_comfy(args.comfy_root, args.models_root)

    import torch
    import comfy.lora
    import comfy.model_management
    import comfy.sd
    from yue2_trainer import prefix as P
    from yue2_trainer.constants import CODEC_OFFSET, MUSIC_END
    from yue2_trainer.forward import ar_hidden, nar_forward
    from yue2_trainer.lora import create_lora, select_target_modules

    model, clip, vae = load_checkpoint(resolve_model_file("checkpoints", args.checkpoint))
    style = "English, warm piano pop, expressive female voice, 88 BPM"
    lyrics = "[Verse]\nNeon fades along the lane\nFootsteps keep the time of rain\n"
    ok = True

    with torch.inference_mode(False), torch.no_grad():
        # ── 1. acoustic ────────────────────────────────────────────────────
        device = P.load_clip_for_prefill(clip)
        te = clip.cond_stage_model
        dtype = torch.bfloat16
        g = torch.Generator().manual_seed(1)
        T = 120
        tokens = torch.randint(0, 32768, (T,), generator=g).tolist()
        prefix_ids, _ = P.music_prefix_ids(clip, style, lyrics, None, "off")
        # comfy reference conditioning (their prefill of prefix + codec + MUSIC_END)
        # comfy stores semantic tokens as vocabulary ids (already offset)
        conditioning, chunks = te._acoustic_conditioning(prefix_ids, [t + CODEC_OFFSET for t in tokens], dtype)
        # ours
        cache = P.build_acoustic_prefix(clip, style, lyrics, None, "off", semantic=tokens, chunk=(0, T), device=device)
        ref_ids = prefix_ids + [t + CODEC_OFFSET for t in tokens] + [MUSIC_END]
        assert cache.ids == ref_ids, "prefix ids differ from comfy's"
        assert cache.ar_length == chunks[0][3] - chunks[0][2], (cache.ar_length, chunks)
        comfy.model_management.unload_all_models()
        comfy.model_management.load_models_gpu([model], memory_required=1e20, force_full_load=True)
        dm = model.model.diffusion_model
        x = torch.randn((1, 64, T), generator=g).to(device, dtype)
        sigma = torch.tensor([0.37], device=device, dtype=dtype)
        ref = dm(x, sigma, conditioning.to(device, dtype), chunks).float()
        ours = nar_forward(dm, x, sigma, cache.kv.to(device), cache.ar_length, frame_offset=0, checkpointing=False).float()
        diff = (ref - ours).abs().mean().item()
        scale = ref.abs().mean().item()
        cos = torch.nn.functional.cosine_similarity(ref.flatten(), ours.flatten(), dim=0).item()
        print(f"[1] acoustic velocity: mean|diff|={diff:.4g} mean|ref|={scale:.4g} cos={cos:.5f}")
        ok &= cos > 0.999
        # crop consistency: frames [40:80] of the chunk computed with frame_offset must match the full pass
        ours_crop = nar_forward(dm, x[..., 40:80], sigma, cache.kv.to(device), cache.ar_length, frame_offset=40,
                                checkpointing=False).float()
        # NAR attends bidirectionally over all frames, so a crop is *not* expected to equal the full pass;
        # this only checks positions/embeddings are wired (values stay in the same range, no NaN).
        assert torch.isfinite(ours_crop).all()
        print(f"[1b] crop pass finite, mean|v|={ours_crop.abs().mean().item():.4g}")

        # ── 2. planner ─────────────────────────────────────────────────────
        comfy.model_management.unload_all_models()
        device = P.load_clip_for_prefill(clip)
        ids, loss_start = P.abc_sequence(clip, style, lyrics, "X:1\nT:test\nM:4/4\nL:1/8\nK:C\nV:Vocal\nC2 D2 E2 F2|G4 z4|\n", "melody")
        tok = torch.tensor([ids], device=device)
        llm = te.model
        ref_hidden = llm(tok, dtype=dtype)[0]
        ref_logits = llm.lm_head(ref_hidden).float()
        our_hidden = ar_hidden(llm, tok, dtype, checkpointing=False)
        our_logits = llm.lm_head(our_hidden).float()
        diff = (ref_logits - our_logits).abs().mean().item()
        scale = ref_logits.abs().mean().item()
        cos = torch.nn.functional.cosine_similarity(ref_logits.flatten(), our_logits.flatten(), dim=0).item()
        agree = (ref_logits.argmax(-1) == our_logits.argmax(-1)).float().mean().item()
        print(f"[2] planner logits: mean|diff|={diff:.4g} mean|ref|={scale:.4g} cos={cos:.5f} argmax agree={agree:.1%}")
        ok &= agree > 0.97 and cos > 0.999

        # ── 3. LoRA keys ───────────────────────────────────────────────────
        m_targets = select_target_modules(model.model, "attention+mlp", include_acoustic_extra=True, name_prefix="diffusion_model.")
        m_lora = create_lora(model.model, m_targets, rank=4, alpha=4.0, save_prefix="")
        c_targets = select_target_modules(te, "attention+mlp", name_prefix="")
        c_lora = create_lora(te, c_targets, rank=4, alpha=4.0, save_prefix="text_encoders.")
        sd = {**m_lora.export(), **c_lora.export()}
        for v in sd.values():
            if v.ndim == 2:
                v.normal_(0, 0.01)
        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        key_map = comfy.lora.model_lora_keys_clip(te, key_map)
        wanted = {k.rsplit(".", 2)[0] for k in sd if k.endswith(".lora_up.weight")}
        missing = [k for k in wanted if k not in key_map]
        print(f"[3] {len(wanted)} LoRA modules ({len(m_targets)} model, {len(c_targets)} clip); unmapped: {len(missing)} {missing[:3]}")
        ok &= not missing
        records = []

        class Catch(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())
        handler = Catch()
        logging.getLogger().addHandler(handler)
        new_model, new_clip = comfy.sd.load_lora_for_models(model, clip, sd, 1.0, 1.0)
        logging.getLogger().removeHandler(handler)
        not_loaded = [r for r in records if "not loaded" in r]
        print(f"[3b] load_lora_for_models: {len(not_loaded)} 'key not loaded' warnings; "
              f"model patches={len(new_model.patches)} clip patches={len(new_clip.patcher.patches)}")
        ok &= not not_loaded and len(new_model.patches) == len(m_targets) and len(new_clip.patcher.patches) == len(c_targets)

    print("ALL OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
