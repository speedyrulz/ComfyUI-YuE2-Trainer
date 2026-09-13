"""Measure SheetSage2 peak VRAM through ComfyUI's audio encoder, with and without bf16 autocast."""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_cli import bootstrap_comfy, load_audio_encoder
p = argparse.ArgumentParser(); p.add_argument("--comfy-root", required=True); p.add_argument("--models-root", action="append", default=[]); p.add_argument("--audio", required=True)
a = p.parse_args(); bootstrap_comfy(a.comfy_root, a.models_root)
import torch
from yue2_trainer.audio import load_audio, to_stereo_48k
enc = load_audio_encoder("sheetsage2_bf16.safetensors")
print("encoder dtype", enc.model.dtype, "param dtype", next(enc.model.encoder.parameters()).dtype)
wave, sr = load_audio(a.audio); wave = to_stereo_48k(wave, sr)
for label, ctx in (("inference_mode", torch.inference_mode()), ("no_grad", torch.no_grad())):
    for seconds in (20, 90, 240):
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache(); t0 = time.perf_counter()
        try:
            with ctx:
                abc = enc.generate_abc(wave[None, :, :seconds * 48000], 48000, melody_only=True)[0]
            print(f"{label} {seconds}s: peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, {time.perf_counter() - t0:.1f}s, abc {len(abc)} chars")
            if seconds == 20: print(abc[:300].replace("\n", " | "))
        except torch.OutOfMemoryError:
            print(f"{label} {seconds}s: OOM (peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB)"); torch.cuda.empty_cache(); break
import os, logging; logging.shutdown(); sys.stdout.flush(); os._exit(0)
