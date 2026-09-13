"""Headless YuE2 LoRA training using ComfyUI's runtime (no server needed).

Examples (run with ComfyUI's Python):

    python train_cli.py acoustic --comfy-root C:/ai/ComfyUI --checkpoint yue2_3b_bf16.safetensors ^
        --data D:/songs --style "English, indie pop, female vocal" --steps 300 --out my_style_acoustic

    python train_cli.py planner --comfy-root C:/ai/ComfyUI --checkpoint yue2_3b_bf16.safetensors ^
        --data D:/songs --transcribe melody --sheetsage sheetsage2_bf16.safetensors --out my_style_planner

The resulting .safetensors files land in ComfyUI/models/loras and load with LoraLoaderModelOnly
(acoustic) or the "YuE2 Load LoRA" node / LoraLoader clip strength (planner).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def bootstrap_comfy(comfy_root: str, extra_model_roots=()):
    root = Path(comfy_root).resolve()
    if not (root / "comfy").is_dir():
        raise SystemExit(f"{root} does not look like a ComfyUI checkout (no comfy/ package)")
    sys.path.insert(0, str(root))
    import folder_paths  # noqa: F401
    for extra in extra_model_roots:
        extra = Path(extra)
        for sub in ("checkpoints", "loras", "audio_encoders"):
            if (extra / sub).is_dir():
                folder_paths.add_model_folder_path(sub, str(extra / sub))
    return root


def resolve_model_file(kind: str, name: str) -> str:
    import folder_paths
    if Path(name).is_file():
        return str(Path(name).resolve())
    return folder_paths.get_full_path_or_raise(kind, name)


def load_checkpoint(path: str):
    import comfy.sd
    model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(path, output_vae=True, output_clip=True,
                                                                embedding_directory=None)
    return model, clip, vae


def build_dataset(args):
    from yue2_trainer.audio import audio_seconds
    from yue2_trainer.dataset import scan_folder
    dataset = scan_folder(args.data, args.style or "", args.lyrics or "", recursive=not args.no_recursive)
    for item in dataset.items:
        if item.seconds is None and item.audio_path:
            item.seconds = audio_seconds(item.audio_path)
    return dataset


def encode_dataset(dataset, vae, args, audio_encoder=None):
    import torch
    from yue2_trainer.audio import crop_audio, encode_latents, load_audio, to_stereo_48k
    from yue2_trainer.constants import FRAMES_PER_SECOND
    from yue2_trainer.dataset import cache_key, load_cache, save_cache
    cache = Path(args.cache_dir).resolve()
    for item in dataset.items:
        tag = f"v1|max={args.max_seconds}|win={args.window_seconds}"
        key = cache_key(item, tag)
        cached = load_cache(cache, key) if not args.force_reencode else None
        waveform = None
        if item.latents is None:
            if cached is not None and "latents" in cached:
                item.latents = cached["latents"]
            else:
                wave, sr = load_audio(item.audio_path)
                waveform = crop_audio(to_stereo_48k(wave, sr), args.max_seconds)
                t0 = time.perf_counter()
                item.latents = encode_latents(vae, waveform, window_seconds=args.window_seconds).to(torch.float16)
                logging.info("encoded %s: %d frames in %.1fs", item.id, item.latents.shape[-1], time.perf_counter() - t0)
                cached = {"latents": item.latents, **({k: v for k, v in (cached or {}).items() if k != "latents"})}
                save_cache(cache, key, cached)
            item.seconds = item.latents.shape[-1] / FRAMES_PER_SECOND
        if args.transcribe != "none" and not item.abc and audio_encoder is not None:
            field = f"abc_{args.transcribe}"
            if cached and cached.get(field):
                item.abc = cached[field]
            else:
                if waveform is None:
                    wave, sr = load_audio(item.audio_path)
                    waveform = crop_audio(to_stereo_48k(wave, sr), args.max_seconds)
                with torch.inference_mode():
                    scores = audio_encoder.generate_abc(waveform[None], 48000, melody_only=args.transcribe == "melody")
                item.abc = scores[0] if isinstance(scores, (list, tuple)) else scores
                cached = dict(cached or {})
                cached[field] = item.abc
                cached.setdefault("latents", item.latents)
                save_cache(cache, key, cached)
                logging.info("transcribed %s: %d ABC chars", item.id, len(item.abc or ""))
    return dataset


def load_audio_encoder(name: str):
    import comfy.sd  # noqa: F401  (registers comfy.model_patcher for the audio encoder module)
    import comfy.audio_encoders.audio_encoders
    import comfy.utils
    path = resolve_model_file("audio_encoders", name)
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    return comfy.audio_encoders.audio_encoders.load_audio_encoder_from_sd(sd)


def save_result(result, args, kind: str):
    import folder_paths
    from yue2_trainer.lora import save_lora_file
    out = Path(args.out)
    if out.suffix != ".safetensors":
        out = out.with_suffix(".safetensors")
    if not out.is_absolute() and out.parent == Path("."):
        out = Path(folder_paths.get_folder_paths("loras")[0]) / out
    out.parent.mkdir(parents=True, exist_ok=True)
    info = {**result.info, "final_loss": result.losses[-1] if result.losses else None}
    save_lora_file(result.lora_sd, out, info)
    (out.with_suffix(".loss.json")).write_text(json.dumps({"loss": result.losses, "info": info}, indent=1))
    logging.info("saved %s LoRA (%d tensors) to %s", kind, len(result.lora_sd), out)
    return out


def add_common(p):
    p.add_argument("--comfy-root", required=True, help="Path to the ComfyUI checkout (needs YuE2 support).")
    p.add_argument("--models-root", action="append", default=[], help="Extra ComfyUI models folder(s) to search.")
    p.add_argument("--checkpoint", default="yue2_3b_bf16.safetensors", help="Checkpoint name or path.")
    p.add_argument("--data", required=True, help="Dataset folder.")
    p.add_argument("--style", default="", help="Default style prompt for files without a sidecar.")
    p.add_argument("--lyrics", default="", help="Default lyrics for files without a sidecar.")
    p.add_argument("--no-recursive", action="store_true")
    p.add_argument("--cache-dir", default=str(HERE / "cache"))
    p.add_argument("--max-seconds", type=float, default=0.0)
    p.add_argument("--window-seconds", type=int, default=60)
    p.add_argument("--force-reencode", action="store_true")
    p.add_argument("--transcribe", choices=["none", "melody", "full"], default="none")
    p.add_argument("--sheetsage", default="sheetsage2_bf16.safetensors")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--targets", default="attention+mlp", choices=["attention", "attention+mlp", "mlp"])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accumulation", type=int, default=1)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--optimizer", default="AdamW")
    p.add_argument("--lora-dtype", default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--no-checkpointing", action="store_true")
    p.add_argument("--devices", default="auto",
                   help="auto | cuda:N | all (data parallel on every GPU) | cuda:0,cuda:1")
    p.add_argument("--existing-lora", default=None)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--out", required=True, help="Output LoRA name (goes to models/loras) or path.")
    p.add_argument("--dry-run", action="store_true", help="Prepare data and LoRA, run 0 steps.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    pa = sub.add_parser("acoustic", help="Train the acoustic (MODEL) LoRA")
    add_common(pa)
    pa.add_argument("--segment-seconds", type=float, default=30.0)
    pa.add_argument("--prefix-mode", default="auto", choices=["auto", "full", "melody", "off"])
    pa.add_argument("--no-semantic", action="store_true", help="Ignore semantic tokens even when present.")
    pa.add_argument("--train-acoustic-head", action="store_true")
    pa.add_argument("--timestep-sampling", default="uniform", choices=["uniform", "logit_normal"])
    pa.add_argument("--shift", type=float, default=1.0)
    pp = sub.add_parser("planner", help="Train the planner / semantic (CLIP) LoRA")
    add_common(pp)
    pp.add_argument("--no-abc", action="store_true", help="Do not train the ABC target.")
    pp.add_argument("--semantic", action="store_true", help="Also train the semantic-token target.")
    pp.add_argument("--abc-mode", default="auto", choices=["auto", "full", "melody"])
    pp.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bootstrap_comfy(args.comfy_root, args.models_root)
    sys.path.insert(0, str(HERE))
    import torch
    import comfy.model_management
    from yue2_trainer.lora import load_lora_file

    ckpt = resolve_model_file("checkpoints", args.checkpoint)
    logging.info("loading %s", ckpt)
    model, clip, vae = load_checkpoint(ckpt)
    dataset = build_dataset(args)
    logging.info("dataset:\n%s", dataset.describe())
    audio_encoder = None
    if args.transcribe != "none" and any(not item.abc for item in dataset.items):
        audio_encoder = load_audio_encoder(args.sheetsage)
    dataset = encode_dataset(dataset, vae, args, audio_encoder)
    if audio_encoder is not None:
        del audio_encoder
    comfy.model_management.unload_all_models()
    logging.info("encoded dataset:\n%s", dataset.describe())
    existing = load_lora_file(resolve_model_file("loras", args.existing_lora)) if args.existing_lora else None
    steps = 0 if args.dry_run else args.steps

    def progress(done, total, loss):
        if done == 1 or done % 10 == 0 or done == total:
            logging.info("step %d/%d loss %.5f", done, total, loss)

    def save_partial(sd, n):
        from yue2_trainer.lora import save_lora_file
        path = Path(args.out).with_suffix("")
        target = Path(f"{path}_{n:06d}.safetensors")
        if not target.is_absolute() and target.parent == Path("."):
            import folder_paths
            target = Path(folder_paths.get_folder_paths("loras")[0]) / target
        save_lora_file(sd, target, {"partial": True, "steps": n})
        logging.info("saved intermediate LoRA %s", target)

    with torch.inference_mode(False):
        if args.command == "acoustic":
            from yue2_trainer.acoustic import AcousticConfig, train_acoustic_lora
            cfg = AcousticConfig(steps=steps, batch_size=args.batch_size, grad_accumulation=args.grad_accumulation,
                                 learning_rate=args.lr, rank=args.rank, alpha=args.alpha, targets=args.targets,
                                 train_acoustic_head=args.train_acoustic_head, segment_seconds=args.segment_seconds,
                                 mode=args.prefix_mode, use_semantic_tokens=not args.no_semantic,
                                 timestep_sampling=args.timestep_sampling, shift=args.shift, warmup_steps=args.warmup,
                                 seed=args.seed, lora_dtype=args.lora_dtype,
                                 gradient_checkpointing=not args.no_checkpointing, optimizer=args.optimizer,
                                 devices=args.devices, existing_lora=existing, save_every=args.save_every, save_callback=save_partial)
            result = train_acoustic_lora(model, clip, dataset, cfg, progress=progress)
        else:
            from yue2_trainer.planner import PlannerConfig, train_planner_lora
            cfg = PlannerConfig(steps=steps, batch_size=args.batch_size, grad_accumulation=args.grad_accumulation,
                                learning_rate=args.lr, rank=args.rank, alpha=args.alpha, targets=args.targets,
                                train_abc=not args.no_abc, train_semantic=args.semantic, abc_mode=args.abc_mode,
                                max_tokens=args.max_tokens, warmup_steps=args.warmup, seed=args.seed,
                                lora_dtype=args.lora_dtype, gradient_checkpointing=not args.no_checkpointing,
                                optimizer=args.optimizer, devices=args.devices, existing_lora=existing,
                                save_every=args.save_every, save_callback=save_partial)
            result = train_planner_lora(clip, dataset, cfg, progress=progress)
    out = save_result(result, args, args.command)
    print(f"done: {out}", flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    # ComfyUI's ModelPatcher.__del__ touches module globals during interpreter teardown; skip it.
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
