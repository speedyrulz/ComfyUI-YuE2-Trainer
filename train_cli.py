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
        tag = f"v2|max={args.max_seconds}|win={args.window_seconds}|{args.vae_precision}"
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
                item.latents = encode_latents(vae, waveform, window_seconds=args.window_seconds,
                                              precision=args.vae_precision).to(torch.float16)
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
                    try:
                        scores = audio_encoder.generate_abc(waveform[None], 48000, melody_only=args.transcribe == "melody")
                        item.abc = scores[0] if isinstance(scores, (list, tuple)) else scores
                    except Exception as exc:  # noqa: BLE001
                        logging.warning("SheetSage2 transcription failed for %s (%s); training without an ABC score", item.id, exc)
                        item.abc = None
                        continue
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
    (out.with_suffix(".loss.json")).write_text(json.dumps({"loss": result.losses, "eval": result.evals, "info": info}, indent=1))
    if getattr(result, "state", None):
        from yue2_trainer.resume import save_state, state_path
        save_state(result.state, state_path(out))
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
    p.add_argument("--vae-precision", default="fp32", choices=["fp32", "fp16"])
    p.add_argument("--force-reencode", action="store_true")
    p.add_argument("--transcribe", choices=["none", "melody", "full"], default="none")
    p.add_argument("--semantic-head", default=None, metavar="FILE",
                   help="Predict semantic tokens for every song with the Mothersuperior v4 head (name in "
                        "models/audio_encoders or a path); writes <song>.semantic.npy sidecars.")
    p.add_argument("--mert", default="m-a-p/MERT-v2-FullSong", help="MERT-v2-FullSong folder or Hugging Face id.")
    p.add_argument("--semantic-force", action="store_true", help="Recompute semantic tokens even when sidecars exist.")
    p.add_argument("--sheetsage", default="sheetsage2_bf16.safetensors")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-schedule", default="cosine", choices=["cosine", "constant", "linear"])
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
    p.add_argument("--log-every", type=int, default=1, help="Console step/loss line every N steps.")
    p.add_argument("--eval-every", type=int, default=50,
                   help="Score a fixed evaluation set (same crops/sigmas/noise) every N steps and before step 1; 0 = off.")
    p.add_argument("--eval-samples", type=int, default=8, help="Size of the fixed evaluation set.")
    p.add_argument("--eval-holdout", type=int, default=1,
                   help="Songs kept out of training for the evaluation set (0 = score training crops).")
    p.add_argument("--keep", default="final", choices=["final", "best_eval"],
                   help="Save the final weights or the checkpoint with the lowest evaluation loss.")
    p.add_argument("--tensorboard", default=None, metavar="DIR",
                   help="Log loss/lr/grad-norm to TensorBoard under DIR (tensorboard --logdir DIR).")
    p.add_argument("--run-name", default="", help="TensorBoard run name (default: output name).")
    p.add_argument("--existing-lora", default=None)
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore the .resume state next to --existing-lora (fresh optimizer and schedule).")
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--out", required=True, help="Output LoRA name (goes to models/loras) or path.")
    p.add_argument("--dry-run", action="store_true", help="Prepare data and LoRA, run 0 steps.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    pa = sub.add_parser("acoustic", help="Train the acoustic (MODEL) LoRA")
    add_common(pa)
    pa.add_argument("--segment-seconds", type=float, default=30.0)
    pa.add_argument("--prefix-mode", default="full", choices=["full", "melody", "off", "auto"])
    pa.add_argument("--conditioning", default="compact", choices=["compact", "inference_like"])
    pa.add_argument("--use-semantic", action="store_true", help="Condition on semantic tokens for items that carry them.")
    pa.add_argument("--train-acoustic-head", action="store_true")
    pa.add_argument("--caption-dropout", type=float, default=0.1,
                    help="Fraction of steps trained on the unconditional (instruction-only) prefix.")
    pa.add_argument("--timestep-sampling", default="uniform", choices=["uniform", "logit_normal"])
    pa.add_argument("--shift", type=float, default=1.0)
    pa.add_argument("--sample-every", type=int, default=0,
                    help="Render a fixed music-token stream with the LoRA under training every N steps (0 = off).")
    pa.add_argument("--sample-tokens", default="", help="The stream: a .semantic.npy (default: the first song with semantic tokens).")
    pa.add_argument("--sample-seconds", type=float, default=30.0)
    pa.add_argument("--sample-seed", type=int, default=0)
    pp = sub.add_parser("planner", help="Train the planner / semantic (CLIP) LoRA")
    add_common(pp)
    pp.add_argument("--no-abc", action="store_true", help="Do not train the ABC target.")
    pp.add_argument("--semantic", action="store_true", help="Also train the semantic-token target.")
    pp.add_argument("--abc-mode", default="full", choices=["full", "melody", "auto"])
    pp.add_argument("--max-tokens", type=int, default=4096)
    pp.add_argument("--regularization", default=None, metavar="DIR",
                    help="Folder of base-model scores (YuE2 output directories / .abc sidecars) mixed into training.")
    pp.add_argument("--regularization-fraction", type=float, default=0.5)
    pp.add_argument("--kl-weight", type=float, default=0.0,
                    help="Trust region: weight of KL(base || LoRA) on the trained positions (0 = off).")
    pp.add_argument("--abc-dropout", type=float, default=0.0,
                    help="Share of semantic draws trained behind the no-sheet prompt (0 = always with the sheet).")
    pp.add_argument("--probe-every", type=int, default=0,
                    help="Generate a whole ABC score with the current LoRA every N steps (and before step 1); 0 = off.")
    pp.add_argument("--probe-style", default="", help="Probe style prompt (default: first training item's).")
    pp.add_argument("--probe-lyrics", default="", help="Probe lyrics, or @file to read them from a file.")
    pp.add_argument("--probe-max-tokens", type=int, default=8192)
    pp.add_argument("--probe-seed", type=int, default=0)
    pp.add_argument("--probe-music-seconds", type=float, default=0.0,
                    help="Also write the music-token stream for the probe prompt with the current LoRA, up to N seconds "
                         "(saved as step_NNNNNN.semantic.npy next to the score); 0 = off.")
    pp.add_argument("--probe-abc", default="",
                    help="Fixed ABC score for the music probes, or @file (default: the score each probe writes).")
    pp.add_argument("--probe-dir", default=None, help="Where probe scores are written (default: <out>_probes/).")
    pp.add_argument("--probe-render", action="store_true",
                    help="Render every music probe to step_NNNNNN.wav on --probe-render-device (needs a GPU training does not use).")
    pp.add_argument("--probe-render-device", default="auto", help="auto (a GPU not used for training) or cuda:N.")
    pm = sub.add_parser("merge", help="Write an acoustic and a planner LoRA into one file")
    pm.add_argument("parts", nargs="+", help="LoRA files to merge (their keys must not overlap).")
    pm.add_argument("--out", required=True, help="Output .safetensors path.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "merge":
        sys.path.insert(0, str(HERE))
        from yue2_trainer.lora import merge_lora_files
        merged, meta = merge_lora_files(args.parts, args.out)
        print(f"merged {len(args.parts)} files -> {args.out}: {len(merged)} tensors "
              f"({meta['model_keys']} model, {meta['clip_keys']} clip)", flush=True)
        return 0
    bootstrap_comfy(args.comfy_root, args.models_root)
    sys.path.insert(0, str(HERE))
    import torch
    import comfy.model_management
    from yue2_trainer.lora import load_lora_file

    ckpt = resolve_model_file("checkpoints", args.checkpoint)
    logging.info("loading %s", ckpt)
    model, clip, vae = load_checkpoint(ckpt)
    dataset = build_dataset(args)
    if args.semantic_head:
        from yue2_trainer.parallel import resolve_devices
        from yue2_trainer.semantic import SemanticTokenizer, summarize, tokenize_dataset
        head = resolve_model_file("audio_encoders", args.semantic_head)
        device = resolve_devices(args.devices if args.devices != "all" else "auto")[0]
        with torch.inference_mode(), SemanticTokenizer(head, args.mert, device) as tok:
            logging.info("%s", summarize(tokenize_dataset(dataset, tok, force=args.semantic_force,
                                                          cache_dir=Path(args.cache_dir) / "semantic")))
    logging.info("dataset:\n%s", dataset.describe())
    audio_encoder = None
    if args.transcribe != "none" and any(not item.abc for item in dataset.items):
        audio_encoder = load_audio_encoder(args.sheetsage)
    dataset = encode_dataset(dataset, vae, args, audio_encoder)
    if audio_encoder is not None:
        del audio_encoder
    comfy.model_management.unload_all_models()
    logging.info("encoded dataset:\n%s", dataset.describe())
    existing = resume = None
    if args.existing_lora:
        from yue2_trainer.resume import load_state, state_path
        lora_path = resolve_model_file("loras", args.existing_lora)
        existing = load_lora_file(lora_path)
        if not args.no_resume:
            resume = load_state(state_path(lora_path))
            if resume is None:
                logging.info("no resume state next to %s; continuing from its weights with a fresh optimizer", lora_path)
    steps = 0 if args.dry_run else args.steps

    def progress(done, total, loss):
        pass  # the trainer's TrainMonitor prints step/loss lines

    def save_partial(sd, n, info, state=None):
        from yue2_trainer.lora import save_lora_file
        from yue2_trainer.resume import save_state, state_path
        path = Path(args.out).with_suffix("")
        target = Path(f"{path}_{n:06d}.safetensors")
        if not target.is_absolute() and target.parent == Path("."):
            import folder_paths
            target = Path(folder_paths.get_folder_paths("loras")[0]) / target
        save_lora_file(sd, target, info)
        if state:
            save_state(state, state_path(target))
        logging.info("saved intermediate LoRA %s%s", target, " (+ resume state)" if state else "")

    def probe_folder():
        folder = Path(args.probe_dir) if getattr(args, "probe_dir", None) else Path(str(Path(args.out).with_suffix("")) + "_probes")
        if not folder.is_absolute() and folder.parent == Path("."):
            import folder_paths
            folder = Path(folder_paths.get_folder_paths("loras")[0]) / folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def audio_writer(prefix):
        def write(step, audio, rate):
            from yue2_trainer.render import save_wav
            return save_wav(probe_folder() / f"{prefix}_{step:06d}.wav", audio, rate)
        return write

    def probe_writer(step, abc, meta, music=None):
        folder = probe_folder()
        target = folder / f"step_{step:06d}.abc"
        target.write_text(abc, encoding="utf-8")
        target.with_suffix(".json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
        if music is not None:
            import numpy as np
            np.save(folder / f"step_{step:06d}.semantic.npy", np.asarray(music, dtype=np.int32))
        return str(target)

    with torch.inference_mode(False):
        if args.command == "acoustic":
            from yue2_trainer.acoustic import AcousticConfig, train_acoustic_lora
            sample = None
            if args.sample_every > 0:
                from yue2_trainer.render import sample_stream
                sample = sample_stream(dataset.items, args.sample_tokens or None, args.sample_seconds)
                if sample is None:
                    logging.warning("--sample-every: nothing to render (give --sample-tokens or a dataset with semantic tokens)")
            cfg = AcousticConfig(steps=steps, batch_size=args.batch_size, grad_accumulation=args.grad_accumulation,
                                 learning_rate=args.lr, lr_schedule=args.lr_schedule, rank=args.rank, alpha=args.alpha, targets=args.targets,
                                 train_acoustic_head=args.train_acoustic_head, segment_seconds=args.segment_seconds,
                                 mode=args.prefix_mode, conditioning=args.conditioning,
                                 use_semantic_tokens=args.use_semantic,
                                 caption_dropout=args.caption_dropout,
                                 timestep_sampling=args.timestep_sampling, shift=args.shift, warmup_steps=args.warmup,
                                 seed=args.seed, lora_dtype=args.lora_dtype,
                                 gradient_checkpointing=not args.no_checkpointing, optimizer=args.optimizer,
                                 devices=args.devices, existing_lora=existing, resume_state=resume,
                                 log_every=args.log_every, eval_every=args.eval_every, eval_samples=args.eval_samples,
                                 eval_holdout=args.eval_holdout, keep=args.keep,
                                 tensorboard_dir=args.tensorboard or "",
                                 run_name=args.run_name or Path(args.out).stem, save_every=args.save_every, save_callback=save_partial,
                                 sample_every=args.sample_every if sample else 0, sample_seconds=args.sample_seconds,
                                 sample_seed=args.sample_seed, sample_codes=sample["codes"] if sample else None,
                                 sample_prompt=sample, sample_callback=audio_writer("sample"))
            result = train_acoustic_lora(model, clip, dataset, cfg, progress=progress, vae=vae)
        else:
            from yue2_trainer.dataset import scan_folder
            from yue2_trainer.planner import PlannerConfig, train_planner_lora
            probe_lyrics = args.probe_lyrics
            if probe_lyrics.startswith("@"):
                probe_lyrics = Path(probe_lyrics[1:]).read_text(encoding="utf-8")
            probe_abc = args.probe_abc
            if probe_abc.startswith("@"):
                probe_abc = Path(probe_abc[1:]).read_text(encoding="utf-8")
            regularization = scan_folder(args.regularization) if args.regularization else None
            if regularization is not None:
                logging.info("regularization scores:\n%s", regularization.describe())
            render_callback = None
            if args.probe_render and args.probe_every > 0 and args.probe_music_seconds > 0:
                from yue2_trainer.parallel import resolve_devices
                from yue2_trainer.render import Renderer, pick_render_device
                target = pick_render_device(args.probe_render_device, resolve_devices(args.devices))
                if target is None:
                    logging.warning("--probe-render: no GPU free for rendering (training uses %s); probes stay as token files", args.devices)
                else:
                    renderer = Renderer(model, vae, target)
                    write_audio = audio_writer("step")

                    def render_callback(step, conditioning, frames, meta):
                        audio, rate = renderer.render(conditioning, frames, args.probe_seed)
                        return write_audio(step, audio, rate)
            cfg = PlannerConfig(steps=steps, batch_size=args.batch_size, grad_accumulation=args.grad_accumulation,
                                learning_rate=args.lr, lr_schedule=args.lr_schedule, rank=args.rank, alpha=args.alpha, targets=args.targets,
                                train_abc=not args.no_abc, train_semantic=args.semantic, abc_mode=args.abc_mode,
                                max_tokens=args.max_tokens, warmup_steps=args.warmup, seed=args.seed,
                                lora_dtype=args.lora_dtype, gradient_checkpointing=not args.no_checkpointing,
                                optimizer=args.optimizer, devices=args.devices, existing_lora=existing, resume_state=resume,
                                regularization_fraction=args.regularization_fraction,
                                kl_weight=args.kl_weight, abc_dropout=args.abc_dropout,
                                probe_every=args.probe_every, probe_style=args.probe_style, probe_lyrics=probe_lyrics,
                                probe_max_tokens=args.probe_max_tokens, probe_seed=args.probe_seed, probe_callback=probe_writer,
                                probe_music_seconds=args.probe_music_seconds, probe_abc=probe_abc, render_callback=render_callback,
                                log_every=args.log_every, eval_every=args.eval_every, eval_samples=args.eval_samples,
                                eval_holdout=args.eval_holdout, keep=args.keep,
                                tensorboard_dir=args.tensorboard or "",
                                run_name=args.run_name or Path(args.out).stem,
                                save_every=args.save_every, save_callback=save_partial)
            result = train_planner_lora(clip, dataset, cfg, progress=progress, regularization=regularization)
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
