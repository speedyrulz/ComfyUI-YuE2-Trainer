"""Load a trained LoRA file through ComfyUI's stock loader path and check it changes the model output.

    python tests/verify_lora_file.py --comfy-root <ComfyUI> --models-root <ComfyUI>/models --lora path.safetensors
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_cli import bootstrap_comfy, load_checkpoint, resolve_model_file  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", required=True)
    parser.add_argument("--models-root", action="append", default=[])
    parser.add_argument("--lora", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    bootstrap_comfy(args.comfy_root, args.models_root)
    import torch
    import comfy.model_management
    import comfy.sd
    import comfy.utils
    from yue2_trainer import prefix as P

    lora = comfy.utils.load_torch_file(args.lora, safe_load=True)
    kinds = {"model": sum(k.startswith("diffusion_model.") for k in lora), "clip": sum(k.startswith("text_encoders.") for k in lora)}
    print(f"lora tensors: {len(lora)} ({kinds})")
    model, clip, vae = load_checkpoint(resolve_model_file("checkpoints", "yue2_3b_bf16.safetensors"))
    records = []

    class Catch(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())
    logging.getLogger().addHandler(Catch())
    model_l, clip_l = comfy.sd.load_lora_for_models(model, clip, lora, 1.0, 1.0)  # what LoraLoader does
    warnings = [r for r in records if "not loaded" in r]
    print(f"'lora key not loaded' warnings: {len(warnings)}", warnings[:2])
    ok = not warnings

    with torch.inference_mode(False), torch.no_grad():
        style, lyrics = "English, warm piano pop, 88 BPM", "[Verse]\nNeon fades along the lane\n"
        g = torch.Generator().manual_seed(3)
        T = 100
        tokens = torch.randint(0, 32768, (T,), generator=g).tolist()
        x_cpu = torch.randn((1, 64, T), generator=g)
        outputs = {}
        for tag, (m, c) in {"base": (model, clip), "lora": (model_l, clip_l)}.items():
            comfy.model_management.unload_all_models()
            device = P.load_clip_for_prefill(c)
            ids, _ = P.abc_sequence(c, style, lyrics, "X:1\nK:C\nC D E F|", "melody")
            te = c.cond_stage_model
            logits = te.model.lm_head(te.model(torch.tensor([ids], device=device), dtype=torch.bfloat16)[0]).float().cpu()
            cache = P.build_acoustic_prefix(c, style, lyrics, None, "off", semantic=tokens, chunk=(0, T), device=device)
            comfy.model_management.unload_all_models()
            comfy.model_management.load_models_gpu([m], memory_required=1e20, force_full_load=True)
            dm = m.model.diffusion_model
            x = x_cpu.to(device, torch.bfloat16)
            from yue2_trainer.forward import nar_forward
            v = nar_forward(dm, x, torch.tensor([0.5], device=device, dtype=torch.bfloat16), cache.kv.to(device),
                            cache.ar_length, 0, checkpointing=False).float().cpu()
            outputs[tag] = (logits, v)
        dl = (outputs["base"][0] - outputs["lora"][0]).abs().mean().item()
        dv = (outputs["base"][1] - outputs["lora"][1]).abs().mean().item()
        # A CLIP LoRA changes the acoustic prefix cache, so it also moves the velocity.
        print(f"mean|delta| planner logits: {dl:.5f} (expected >0 iff clip keys); acoustic velocity: {dv:.5f} (expected >0 iff model or clip keys)")
        finite = torch.isfinite(outputs["lora"][0]).all() and torch.isfinite(outputs["lora"][1]).all()
        ok &= bool(finite) and ((dv > 0) == (kinds["model"] > 0 or kinds["clip"] > 0)) and ((dl > 0) == (kinds["clip"] > 0))
    print("ALL OK" if ok else "FAILED")
    logging.shutdown()
    import os
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
