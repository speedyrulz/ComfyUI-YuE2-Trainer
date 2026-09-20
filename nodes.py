"""ComfyUI nodes for training YuE2 LoRAs (acoustic MODEL LoRA and planner CLIP LoRA)."""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from typing_extensions import override

import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, io

from .yue2_trainer.acoustic import LR_SCHEDULES, AcousticConfig, train_acoustic_lora
from .yue2_trainer.audio import audio_seconds, crop_audio, encode_latents, load_audio, to_stereo_48k
from .yue2_trainer.constants import FRAMES_PER_SECOND
from .yue2_trainer.dataset import Dataset, Item, cache_key, clone_dataset, load_cache, save_cache, scan_folder
from .yue2_trainer.lora import TARGET_PRESETS, load_lora_file, lora_metadata, merge_lora_files, save_lora_file
from .yue2_trainer.parallel import device_choices
from .yue2_trainer.sidecars import DEFAULT_CLAUDE, PRECISIONS, SidecarConfig, WHISPER_CHOICES, prepare_folder, summarize
from .yue2_trainer.monitor import probe_summary
from .yue2_trainer.constants import CODEC_OFFSET, CODEC_SIZE
from .yue2_trainer.dataset import CHORD_RE
from .yue2_trainer.planner import (MUSIC_SAMPLING, PROBE_SAMPLING, PlannerConfig, conditioning_from_tokens, generate_abc,
                                   generate_music, train_planner_lora)
from .yue2_trainer.prefix import resolve_mode
from .yue2_trainer.render import Renderer, pick_render_device, sample_stream, save_wav, tokens_and_prompt
from .yue2_trainer.resume import load_state, save_state, state_path
from .yue2_trainer.semantic import DEFAULT_MERT, HEAD_FILENAME, SemanticTokenizer, head_sort_key, tokenize_dataset
from .yue2_trainer.semantic import summarize as summarize_semantic
from .yue2_trainer.parallel import resolve_devices

DATASET = io.Custom("YUE2_DATASET")
LORA_MODEL = io.Custom("LORA_MODEL")
LOSS_MAP = io.Custom("LOSS_MAP")
CATEGORY = "YuE2/training"


def _interrupt():
    comfy.model_management.throw_exception_if_processing_interrupted()


def _lora_dir() -> Path:
    return Path(folder_paths.get_folder_paths("loras")[0])


def _existing_lora(name: str):
    if not name or name == "[None]":
        return None
    return load_lora_file(folder_paths.get_full_path_or_raise("loras", name))


def _resume_state(name: str, enabled: bool):
    """The ``.resume`` file written next to ``name`` by save_every / YuE2 Save LoRA, if any."""
    if not enabled or not name or name == "[None]":
        return None
    path = state_path(folder_paths.get_full_path_or_raise("loras", name))
    state = load_state(path)
    if state is None:
        logging.info("YuE2 trainer: no resume state next to %s; continuing from its weights with a fresh optimizer", name)
    return state


def _probe_dir(save_name: str) -> Path:
    safe = "".join(c for c in save_name.strip() if c not in '\\/:*?"<>|') or "yue2_lora"
    return Path(folder_paths.get_output_directory()) / "yue2_probes" / safe


def _probe_writer(save_name: str):
    """Saves every probe score as output/yue2_probes/<save_name>/step_000025.abc (+ .json with its metadata,
    + .semantic.npy with the music tokens when the probe wrote a music stream)."""
    folder = _probe_dir(save_name)

    def write(step: int, abc: str, meta: dict, music=None):
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"step_{step:06d}.abc"
        target.write_text(abc, encoding="utf-8")
        target.with_suffix(".json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
        if music is not None:
            np.save(folder / f"step_{step:06d}.semantic.npy", np.asarray(music, dtype=np.int32))
        return str(target)

    return write


def _audio_writer(folder: Path, prefix: str):
    """Saves rendered audio as <folder>/<prefix>_000025.wav."""
    def write(step: int, audio, rate: int):
        return save_wav(folder / f"{prefix}_{step:06d}.wav", audio, rate)
    return write


def _render_device_choices():
    return ["auto", "off"] + [d for d in device_choices() if d.startswith("cuda")]


def _lora_choices():
    return ["[None]"] + folder_paths.get_filename_list("loras")


def _tensorboard_dir(enabled: bool, folder: str):
    if not enabled:
        return ""
    path = Path((folder or "yue2_tensorboard").strip().strip('"'))
    if not path.is_absolute():
        path = Path(folder_paths.get_output_directory()) / path
    return str(path)


def _save_checkpoint(lora_sd, name: str, steps: int, info: dict, state=None) -> str:
    target = _lora_dir() / f"{name}_{steps:06d}.safetensors"
    save_lora_file(lora_sd, target, info)
    if state:
        save_state(state, state_path(target))
    logging.info("YuE2 trainer: saved intermediate LoRA %s%s", target, " (+ resume state)" if state else "")
    return str(target)


class YuE2TrainerDatasetFolder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YuE2TrainerDatasetFolder",
            display_name="YuE2 Dataset From Folder",
            category=CATEGORY,
            description="Scans a folder for audio files (+ .style.txt / .lyrics.txt / .abc / .json / .semantic.npy sidecars) "
                        "and YuE2 output directories (request.json + audio.flac + semantic.npy + latent.npy).",
            inputs=[
                io.String.Input("folder", default="", tooltip="Absolute path, or a folder name inside ComfyUI/input."),
                io.String.Input("default_style", multiline=True, default="",
                                tooltip="Style / tags prompt used for files without a .style.txt or .json sidecar."),
                io.String.Input("default_lyrics", multiline=True, default="",
                                tooltip="Lyrics used for files without a .lyrics.txt / .txt / .json sidecar."),
                io.Boolean.Input("recursive", default=True),
            ],
            outputs=[DATASET.Output("dataset", display_name="dataset"), io.String.Output("summary", display_name="summary")],
        )

    @classmethod
    def execute(cls, folder, default_style, default_lyrics, recursive):
        path = Path(folder.strip().strip('"'))
        if not path.is_absolute():
            candidate = Path(folder_paths.get_input_directory()) / path
            if candidate.is_dir():
                path = candidate
        dataset = scan_folder(path, default_style, default_lyrics, recursive)
        for item in dataset.items:
            if item.seconds is None and item.audio_path:
                item.seconds = audio_seconds(item.audio_path)
        return io.NodeOutput(dataset, dataset.describe())


class YuE2TrainerPrepareDataset(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YuE2TrainerPrepareDataset",
            display_name="YuE2 Prepare Dataset (auto style + lyrics)",
            category=CATEGORY,
            description="Writes .style.txt and .lyrics.txt sidecars for every song in a folder: lyrics from LRCLIB "
                        "(by tags / 'Artist - Title' names) with Whisper transcription as fallback, section tags by "
                        "repetition heuristic or Claude, style prompts from CLAP tags + tempo/key + language. "
                        "Existing sidecars are kept unless overwrite is on. Outputs the scanned dataset.",
            inputs=[
                io.String.Input("folder", default="", tooltip="Absolute path, or a folder name inside ComfyUI/input."),
                io.Combo.Input("lyrics_source",
                               options=["lrclib+whisper", "whisper", "lrclib", "lrclib+moss-audio", "moss-audio", "none"],
                               default="lrclib+whisper",
                               tooltip="Where lyrics come from. lrclib = lookup by artist/title (needs tags or 'Artist - "
                                       "Title' names). whisper (recommended) / moss-audio = transcription of the vocal "
                                       "stem; the lrclib+ variants transcribe only when no verified database hit exists."),
                io.Combo.Input("whisper_model", options=WHISPER_CHOICES, default=WHISPER_CHOICES[0], advanced=True),
                io.String.Input("language", default="auto",
                                tooltip="Whisper language code (en, zh, ja, ko, es, ...) or auto."),
                io.Boolean.Input("separate_vocals", default=True,
                                 tooltip="Isolate the vocal stem with demucs before transcribing (much more accurate on music)."),
                io.Combo.Input("section_tags", options=["heuristic", "claude", "none"], default="heuristic",
                               tooltip="How [Verse]/[Chorus] tags are added. 'claude' uses the Anthropic API "
                                       "(ANTHROPIC_API_KEY) and falls back to the heuristic on any failure."),
                io.String.Input("claude_model", default=DEFAULT_CLAUDE, advanced=True),
                io.Combo.Input("style_source", options=["moss-audio", "qwen-omni", "clap", "none"], default="moss-audio",
                               tooltip="moss-audio: MOSS-Audio-4B-Instruct (~10 GB download) listens and writes a descriptive "
                                       "tag line. qwen-omni: Qwen2.5-Omni-3B (~7 GB). clap: fixed-vocabulary tags. none: use "
                                       "default_style. All add BPM; language is added when Whisper runs."),
                io.Combo.Input("precision", options=PRECISIONS, default="bf16",
                               tooltip="Weight precision for the audio LLM (moss-audio / qwen-omni). bf16 needs ~10 GB "
                                       "(MOSS) / ~8 GB (Omni); nf4 about 4 GB (needs bitsandbytes)."),
                io.String.Input("artist", default="",
                                tooltip="Prepended to every style prompt, e.g. an artist or album name. Use the same "
                                        "phrase in your generation prompts to trigger the LoRA."),
                io.String.Input("default_style", multiline=True, default="", tooltip="Style text when style_source is none."),
                io.Boolean.Input("overwrite", default=False, tooltip="Regenerate sidecars that already exist."),
                io.Combo.Input("device", options=[d for d in device_choices() if d != "all"], default="auto"),
                io.Boolean.Input("recursive", default=True),
            ],
            outputs=[DATASET.Output("dataset", display_name="dataset"), io.String.Output("report", display_name="report")],
        )

    @classmethod
    def execute(cls, folder, lyrics_source, whisper_model, language, separate_vocals, section_tags, claude_model,
                style_source, precision, artist, default_style, overwrite, device, recursive):
        path = Path(folder.strip().strip('"'))
        if not path.is_absolute():
            candidate = Path(folder_paths.get_input_directory()) / path
            if candidate.is_dir():
                path = candidate
        cfg = SidecarConfig(lyrics_source=lyrics_source, whisper_model=whisper_model, style_source=style_source,
                            language=language.strip().lower() or "auto", separate_vocals=separate_vocals,
                            section_tags=section_tags, claude_model=claude_model, default_style=default_style,
                            artist=artist, precision=precision, overwrite=overwrite,
                            device="auto" if device == "auto" else device)
        if cfg.device == "auto":
            cfg.device = str(comfy.model_management.get_torch_device())
        comfy.model_management.unload_all_models()
        pbar = comfy.utils.ProgressBar(1)

        def progress(done, total, name):
            pbar.update_absolute(done, total)

        reports = prepare_folder(path, cfg, recursive=recursive, progress=progress, interrupt=_interrupt)
        comfy.model_management.soft_empty_cache()
        dataset = scan_folder(path, default_style, "", recursive)
        for item in dataset.items:
            if item.seconds is None and item.audio_path:
                item.seconds = audio_seconds(item.audio_path)
        return io.NodeOutput(dataset, summarize(reports))


class YuE2TrainerDatasetFromAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YuE2TrainerDatasetFromAudio",
            display_name="YuE2 Dataset From Audio",
            category=CATEGORY,
            description="Builds a one-item dataset from a loaded AUDIO plus its style, lyrics and optional ABC score.",
            inputs=[
                io.Audio.Input("audio"),
                io.String.Input("style", multiline=True, default=""),
                io.String.Input("lyrics", multiline=True, default=""),
                io.String.Input("abc", multiline=True, default="", optional=True),
                io.String.Input("name", default="clip"),
                DATASET.Input("append_to", optional=True, tooltip="Optional dataset to append the item to."),
            ],
            outputs=[DATASET.Output("dataset", display_name="dataset")],
        )

    @classmethod
    def execute(cls, audio, style, lyrics, name, abc="", append_to=None):
        wave = audio["waveform"]
        if wave.ndim == 3:
            wave = wave[0]
        item = Item(id=name or "clip", audio_path=None, style=style, lyrics=lyrics, abc=abc or None, source="audio")
        item.extra["waveform"] = wave.detach().cpu().float()
        item.extra["sample_rate"] = int(audio["sample_rate"])
        item.seconds = wave.shape[-1] / float(audio["sample_rate"])
        items = list(append_to.items) if append_to is not None else []
        items.append(item)
        return io.NodeOutput(Dataset(items=items, meta={} if append_to is None else dict(append_to.meta)))


class YuE2TrainerEncodeDataset(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YuE2TrainerEncodeDataset",
            display_name="YuE2 Encode Dataset",
            category=CATEGORY,
            description="Encodes every item's audio into YuE2 VAE latents (cached on disk) and optionally transcribes "
                        "missing ABC scores with SheetSage2.",
            inputs=[
                DATASET.Input("dataset"),
                io.Vae.Input("vae", tooltip="The VAE from the YuE2 checkpoint."),
                io.AudioEncoder.Input("audio_encoder", optional=True, tooltip="SheetSage2 (Load Audio Encoder) for transcription."),
                io.Combo.Input("transcribe", options=["none", "melody", "full"], default="full",
                               tooltip="Transcribe items without an ABC score. full = melody + chord symbols "
                                       "(matches the default generation mode); melody = no chords (cover workflows)."),
                io.Float.Input("max_seconds", default=0.0, min=0.0, max=900.0, step=1.0,
                               tooltip="Truncate each song to this many seconds (0 = whole song)."),
                io.Int.Input("window_seconds", default=60, min=10, max=600,
                             tooltip="Encoding window; larger is faster but needs more VRAM."),
                io.Combo.Input("vae_precision", options=["fp32", "fp16"], default="fp32",
                               tooltip="VAE compute precision for the training latents. fp32 matches the reference "
                                       "pipeline; fp16 (ComfyUI's generation default) deviates by ~2%."),
                io.String.Input("cache_dir", default="yue2_trainer_cache",
                                tooltip="Latent cache folder (relative paths live in ComfyUI/output)."),
                io.Boolean.Input("force_reencode", default=False),
            ],
            outputs=[DATASET.Output("dataset", display_name="dataset"), io.String.Output("summary", display_name="summary")],
        )

    @classmethod
    def execute(cls, dataset, vae, transcribe, max_seconds, window_seconds, vae_precision, cache_dir, force_reencode,
                audio_encoder=None):
        cache = Path(cache_dir)
        if not cache.is_absolute():
            cache = Path(folder_paths.get_output_directory()) / cache
        items = []
        pbar = comfy.utils.ProgressBar(len(dataset.items))
        for index, src in enumerate(dataset.items):
            _interrupt()
            item = Item(**{k: v for k, v in src.__dict__.items()})
            tag = f"v2|max={max_seconds}|win={window_seconds}|{vae_precision}"
            key = cache_key(item, tag)
            cached = None if force_reencode else load_cache(cache, key)
            need_abc = transcribe != "none" and not (item.abc and item.abc.strip()) and audio_encoder is not None
            waveform = None
            if item.latents is None:
                if cached is not None and "latents" in cached:
                    item.latents = cached["latents"]
                else:
                    if item.audio_path:
                        wave, sr = load_audio(item.audio_path)
                    else:
                        wave, sr = item.extra["waveform"], item.extra["sample_rate"]
                    waveform = crop_audio(to_stereo_48k(wave, sr), max_seconds)
                    item.latents = encode_latents(vae, waveform, window_seconds=window_seconds,
                                                  precision=vae_precision).to(torch.float16)
                    payload = {"latents": item.latents}
                    if cached and "abc" in cached:
                        payload["abc"] = cached["abc"]
                    save_cache(cache, key, payload)
                    cached = payload
                item.seconds = item.latents.shape[-1] / FRAMES_PER_SECOND
            if need_abc:
                cache_field = f"abc_{transcribe}"
                if cached is not None and cached.get(cache_field):
                    item.abc = cached[cache_field]
                else:
                    if waveform is None:
                        if item.audio_path:
                            wave, sr = load_audio(item.audio_path)
                        else:
                            wave, sr = item.extra["waveform"], item.extra["sample_rate"]
                        waveform = crop_audio(to_stereo_48k(wave, sr), max_seconds)
                    with torch.inference_mode():
                        try:
                            scores = audio_encoder.generate_abc(waveform[None], 48000, melody_only=transcribe == "melody")
                            item.abc = scores[0] if isinstance(scores, (list, tuple)) else scores
                        except Exception as exc:  # noqa: BLE001 - SheetSage2 can fail to rebuild an ABC for some songs
                            logging.warning("YuE2 trainer: SheetSage2 transcription failed for %s (%s); training without an ABC score",
                                            item.id, exc)
                            item.abc = None
                            item.extra["abc_error"] = str(exc)
                        else:
                            payload = dict(cached or {})
                            payload[cache_field] = item.abc
                            if "latents" not in payload:
                                payload["latents"] = item.latents
                            save_cache(cache, key, payload)
                            cached = payload
            if item.semantic is not None and abs(len(item.semantic) - item.latents.shape[-1]) > 2:
                logging.warning("YuE2 trainer: %s semantic tokens (%d) do not match latent frames (%d)",
                                item.id, len(item.semantic), item.latents.shape[-1])
            item.extra.pop("waveform", None)
            items.append(item)
            pbar.update_absolute(index + 1)
        out = Dataset(items=items, meta={**dataset.meta, "cache_dir": str(cache)})
        return io.NodeOutput(out, out.describe())


def _head_choices():
    names = [n for n in folder_paths.get_filename_list("audio_encoders") if "tokenizer_head" in n]
    return sorted(names, key=head_sort_key) or [HEAD_FILENAME]


class YuE2TrainerSemanticTokens(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YuE2TrainerSemanticTokens",
            display_name="YuE2 Semantic Tokens (community head)",
            category=CATEGORY,
            description="Predicts YuE2 semantic tokens for every recording with a Mothersuperior tokenizer head (v9 current) "
                        "(MERT-v2-FullSong layer 20 -> 32,768 codes, 25 per second) and writes <song>.semantic.npy next to "
                        "the audio. An approximation of YuE2's unreleased tokenizer: a round trip keeps a song's rhythm and "
                        "most of its harmony. Enables train_semantic on the planner node and use_semantic_tokens on the "
                        "acoustic node for your own songs. Items from YuE2 output folders keep their exact tokens.",
            inputs=[
                DATASET.Input("dataset"),
                io.Combo.Input("head", options=_head_choices(),
                               tooltip=f"A tokenizer_head_*.safetensors from huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-"
                                       f"tokenizer-v4 placed in models/audio_encoders (CC BY-NC 4.0); {HEAD_FILENAME} is the current "
                                       "one (trained with an audio-domain loss; pair it with nar_lora_joint_v9_comfyui on the MODEL "
                                       "path when rendering its tokens). Tokens record the head that wrote them "
                                       "(<song>.semantic.json); a head change re-tokenizes."),
                io.String.Input("mert", default=DEFAULT_MERT,
                                tooltip="MERT-v2-FullSong: a local folder with the model files, or the Hugging Face id "
                                        "(downloaded to the HF cache on first use, about 630 MB)."),
                io.Combo.Input("device", options=[d for d in device_choices() if d != "all"], default="auto"),
                io.Boolean.Input("force", default=False, tooltip="Recompute even when a .semantic.npy sidecar from this head exists."),
                io.Boolean.Input("write_sidecars", default=True,
                                 tooltip="Write <song>.semantic.npy next to the audio (reused by later runs and by the "
                                         "Dataset From Folder node). Off: cache under output/yue2_trainer_cache/semantic."),
            ],
            outputs=[DATASET.Output("dataset", display_name="dataset"), io.String.Output("report", display_name="report")],
        )

    @classmethod
    def execute(cls, dataset, head, mert, device, force, write_sidecars):
        head_path = folder_paths.get_full_path_or_raise("audio_encoders", head)
        out = clone_dataset(dataset)
        pbar = comfy.utils.ProgressBar(max(1, sum(1 for i in out.items if i.audio_path)))
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        with torch.inference_mode(), SemanticTokenizer(head_path, mert, resolve_devices(device)[0]) as tok:
            summary = tokenize_dataset(out, tok, force=force, write_sidecars=write_sidecars,
                                       cache_dir=Path(folder_paths.get_output_directory()) / "yue2_trainer_cache" / "semantic",
                                       progress=lambda done, total: pbar.update_absolute(done, total), interrupt=_interrupt)
        report = summarize_semantic(summary) + "\n" + out.describe()
        logging.info("YuE2 trainer: %s", summarize_semantic(summary))
        return io.NodeOutput(out, report)


class YuE2TrainerDatasetMerge(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YuE2TrainerDatasetMerge",
            display_name="YuE2 Merge Datasets",
            category=CATEGORY,
            inputs=[DATASET.Input("dataset_a"), DATASET.Input("dataset_b")],
            outputs=[DATASET.Output("dataset", display_name="dataset")],
        )

    @classmethod
    def execute(cls, dataset_a, dataset_b):
        return io.NodeOutput(Dataset(items=list(dataset_a.items) + list(dataset_b.items),
                                     meta={**dataset_a.meta, **dataset_b.meta}))


def _common_training_inputs(default_lr, default_steps):
    return [
        io.Int.Input("steps", default=default_steps, min=1, max=100000),
        io.Float.Input("learning_rate", default=default_lr, min=1e-7, max=1.0, step=1e-6),
        io.Combo.Input("lr_schedule", options=LR_SCHEDULES, default="cosine",
                       tooltip="cosine: decay to 10% of learning_rate; constant: no decay; linear: straight-line "
                               "decay to 10%. warmup_steps ramps up first in all modes."),
        io.Int.Input("rank", default=16, min=1, max=256),
        io.Float.Input("alpha", default=16.0, min=0.01, max=1024.0, step=0.5,
                       tooltip="LoRA alpha; effective scale is alpha / rank."),
        io.Combo.Input("targets", options=list(TARGET_PRESETS.keys()), default="attention+mlp"),
        io.Int.Input("batch_size", default=1, min=1, max=64, tooltip="Items per optimizer step (processed one at a time)."),
        io.Int.Input("grad_accumulation", default=1, min=1, max=64),
        io.Int.Input("warmup_steps", default=10, min=0, max=10000),
        io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
        io.Combo.Input("optimizer", options=["AdamW", "Adam", "SGD", "Adafactor"], default="AdamW"),
        io.Combo.Input("lora_dtype", options=["fp32", "bf16"], default="fp32", advanced=True),
        io.Boolean.Input("gradient_checkpointing", default=True, advanced=True),
        io.Float.Input("max_grad_norm", default=1.0, min=0.0, max=100.0, step=0.1, advanced=True),
        io.Combo.Input("devices", options=device_choices(), default="auto",
                       tooltip="GPU to train on. 'all' trains data-parallel on every GPU (each holds a full model "
                               "copy; set batch_size x grad_accumulation >= number of GPUs)."),
        io.Combo.Input("existing_lora", options=_lora_choices(), default="[None]",
                       tooltip="Continue training from a LoRA file in models/loras."),
        io.Boolean.Input("resume_state", default=True,
                         tooltip="When existing_lora has a .resume file next to it (written by save_every checkpoints and "
                                 "by YuE2 Save LoRA), restore its optimizer state, random state and step count and "
                                 "continue the same run; steps is then the total length of the run. Off: fresh "
                                 "optimizer and schedule starting from the LoRA weights."),
        io.Int.Input("save_every", default=0, min=0, max=100000, advanced=True,
                     tooltip="Write an intermediate LoRA to models/loras every N steps (0 = off)."),
        io.String.Input("save_name", default="yue2_lora",
                        tooltip="Name of the LoRA: used by YuE2 Save LoRA (when its name is blank), for intermediate "
                                "saves and for the TensorBoard run."),
        io.Int.Input("log_every", default=1, min=1, max=10000,
                     tooltip="Print step / loss / lr / grad-norm / ETA to the console every N steps."),
        io.Int.Input("eval_every", default=50, min=0, max=100000,
                     tooltip="Every N steps, score a fixed evaluation set (same crops, same sigmas, same noise every "
                             "time) so the curve is not buried in per-step sampling noise; also scored before step 1. "
                             "0 = off."),
        io.Int.Input("eval_samples", default=8, min=1, max=256, advanced=True,
                     tooltip="Size of the fixed evaluation set (forward passes per evaluation)."),
        io.Int.Input("eval_holdout", default=1, min=0, max=64,
                     tooltip="Songs kept OUT of training and used for the evaluation set. A held-out loss that stops "
                             "falling or rises means the LoRA is over-training; the loss on training songs keeps "
                             "falling while it memorises them. 0 scores training crops instead (all songs train). "
                             "Never more than items - 3 are held out."),
        io.Combo.Input("keep", options=["final", "best_eval"], default="final",
                       tooltip="Which weights the node outputs: the final step, or the checkpoint with the lowest evaluation "
                               "loss (best_eval; needs eval_every > 0). With eval_holdout 0 the evaluation measures fit, so "
                               "best_eval is then simply the last step. The resume state follows the kept step."),
        io.Float.Input("ema_decay", default=0.0, min=0.0, max=0.9999, step=0.0001, round=0.0001,
                       tooltip="Exponential moving average of the LoRA weights (0 = off). The average is what the node "
                               "outputs and what evaluation, probes, samples and checkpoints use; the last step's raw "
                               "weights are saved next to the final LoRA as <name>_raw.safetensors for comparison. "
                               "Warm-started, so early steps are not frozen in. Smooths the checkpoint-to-checkpoint "
                               "variance: roughly the last 1 / (1 - decay) steps count. Planner (80-100 steps): 0.9; "
                               "acoustic (1000 steps): 0.99."),
        io.Boolean.Input("tensorboard", default=False,
                         tooltip="Log loss, learning rate and grad norm to TensorBoard (pip install tensorboard)."),
        io.String.Input("tensorboard_dir", default="yue2_tensorboard", advanced=True,
                        tooltip="TensorBoard log folder (relative paths live in ComfyUI/output). "
                                "View with: tensorboard --logdir <folder>"),
    ]


class YuE2TrainerAcousticLoRA(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YuE2TrainerAcousticLoRA",
            display_name="YuE2 Train Acoustic LoRA (MODEL)",
            category=CATEGORY,
            description="Flow-matching LoRA training of the YuE2 acoustic model on VAE latents of your songs. "
                        "Load the result with LoraLoaderModelOnly. Learns timbre, production and mix character.",
            inputs=[
                io.Model.Input("model", tooltip="MODEL from the YuE2 checkpoint."),
                io.Clip.Input("clip", tooltip="CLIP from the YuE2 checkpoint (computes the text/ABC prefix)."),
                DATASET.Input("dataset", tooltip="Encoded dataset (with latents)."),
                io.Float.Input("segment_seconds", default=30.0, min=0.0, max=600.0, step=1.0,
                               tooltip="Random crop length per step; 0 trains on whole songs (needs more VRAM)."),
                io.Combo.Input("conditioning", options=["compact", "inference_like"], default="compact",
                               tooltip="compact: style-only prefix (cot off) with the NAR tokens right behind it and positions "
                                       "restarting per segment; fast, needs no ABC/lyrics. inference_like: style+lyrics+ABC "
                                       "prefix, song split into inference-sized chunks, NAR positions as at generation. "
                                       "Both scored the same in A/B tests (see README)."),
                io.Combo.Input("prefix_mode", options=["full", "melody", "off", "auto"], default="full",
                               tooltip="inference_like only: planning instruction in the conditioning prefix; match the mode "
                                       "you generate with. Items without an ABC score always use off. auto: chords -> full, "
                                       "else melody. Ignored by compact conditioning."),
                io.Boolean.Input("use_semantic_tokens", default=False,
                                 tooltip="Condition on YuE2 semantic tokens for items that carry them (only YuE2 output "
                                         "folders do); otherwise text-only conditioning is used."),
                io.Boolean.Input("train_acoustic_head", default=False,
                                 tooltip="Also adapt vae2llm / llm2vae / time embedder projections."),
                io.Float.Input("caption_dropout", default=0.1, min=0.0, max=0.9, step=0.05,
                               tooltip="Fraction of steps trained on YuE2's unconditional prefix (instruction only, "
                                       "no style/lyrics). Keeps the base behaviour reachable and regularises small datasets."),
                io.Combo.Input("timestep_sampling", options=["uniform", "logit_normal"], default="uniform", advanced=True),
                io.Float.Input("shift", default=1.0, min=0.1, max=10.0, step=0.1, advanced=True,
                               tooltip="Sigma shift applied to sampled timesteps (1 = none)."),
                io.Int.Input("sample_every", default=0, min=0, max=100000,
                             tooltip="With the vae connected: every N steps (and before step 1) render a fixed music-token "
                                     "stream with the LoRA under training to output/yue2_probes/<save_name>/sample_NNNNNN.wav. "
                                     "Step 0 is the base model on the same tokens, so what changes is the LoRA. 0 = off."),
                io.String.Input("sample_tokens", default="",
                                tooltip="The stream to render: a .semantic.npy (a probe's, or a song's sidecar; absolute or "
                                        "relative to ComfyUI/output). Blank = the first training song's semantic tokens with "
                                        "its own style, lyrics and score."),
                io.Float.Input("sample_seconds", default=30.0, min=0.0, max=900.0, step=1.0,
                               tooltip="Render only the first N seconds of the stream (0 = all)."),
                io.Int.Input("sample_seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, advanced=True),
                *_common_training_inputs(1e-4, 300),
                io.Vae.Input("vae", optional=True, tooltip="VAE from the checkpoint; needed for sample_every."),
            ],
            outputs=[LORA_MODEL.Output("lora", display_name="lora"), LOSS_MAP.Output("loss_map", display_name="loss_map"),
                     io.Int.Output("steps", display_name="steps"), io.String.Output("report", display_name="report")],
        )

    @classmethod
    def execute(cls, model, clip, dataset, segment_seconds, conditioning, prefix_mode, use_semantic_tokens, train_acoustic_head,
                caption_dropout, timestep_sampling, shift, sample_every, sample_tokens, sample_seconds, sample_seed,
                steps, learning_rate, lr_schedule, rank, alpha, targets, batch_size, grad_accumulation,
                warmup_steps, seed, optimizer, lora_dtype, gradient_checkpointing, max_grad_norm, devices,
                existing_lora, resume_state, save_every, save_name, log_every, eval_every, eval_samples, eval_holdout,
                keep, ema_decay, tensorboard, tensorboard_dir, vae=None):
        sample = None
        if sample_every > 0:
            if vae is None:
                logging.warning("YuE2 trainer: sample_every needs the vae input; no samples will be rendered")
            else:
                path = str(_output_file(sample_tokens)) if sample_tokens.strip() else None
                sample = sample_stream(dataset.items, path, sample_seconds)
                if sample is None:
                    logging.warning("YuE2 trainer: nothing to render for sample_every (no sample_tokens file and no song "
                                    "with semantic tokens); no samples will be rendered")
        cfg = AcousticConfig(
            steps=steps, batch_size=batch_size, grad_accumulation=grad_accumulation, learning_rate=learning_rate,
            lr_schedule=lr_schedule,
            rank=rank, alpha=alpha, targets=targets, train_acoustic_head=train_acoustic_head,
            segment_seconds=segment_seconds, mode=prefix_mode, conditioning=conditioning,
            use_semantic_tokens=use_semantic_tokens,
            caption_dropout=caption_dropout,
            timestep_sampling=timestep_sampling, shift=shift, warmup_steps=warmup_steps, max_grad_norm=max_grad_norm,
            seed=seed, lora_dtype=lora_dtype, gradient_checkpointing=gradient_checkpointing, optimizer=optimizer,
            devices=devices, existing_lora=_existing_lora(existing_lora), save_every=save_every,
            resume_state=_resume_state(existing_lora, resume_state), keep=keep, ema_decay=ema_decay,
            log_every=log_every, eval_every=eval_every, eval_samples=eval_samples, eval_holdout=eval_holdout,
            tensorboard_dir=_tensorboard_dir(tensorboard, tensorboard_dir), run_name=save_name,
            sample_every=sample_every if sample else 0, sample_seconds=sample_seconds, sample_seed=sample_seed,
            sample_codes=sample["codes"] if sample else None, sample_prompt=sample,
            sample_callback=_audio_writer(_probe_dir(save_name), "sample"),
        )
        cfg.save_callback = lambda sd, n, info, state=None: _save_checkpoint(sd, save_name, n, {**info, "save_name": save_name}, state)
        pbar = comfy.utils.ProgressBar(steps)

        def progress(done, total, loss):
            pbar.update_absolute(done, total)

        with torch.inference_mode(False):
            result = train_acoustic_lora(model, clip, dataset, cfg, progress=progress, interrupt_check=_interrupt, vae=vae)
        result.info["save_name"] = save_name
        report = _report(result)
        return io.NodeOutput(result.lora_sd, _loss_map(result), result.steps, report)


class YuE2TrainerPlannerLoRA(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YuE2TrainerPlannerLoRA",
            display_name="YuE2 Train Planner LoRA (CLIP)",
            category=CATEGORY,
            description="Next-token LoRA training of the YuE2 language model: style+lyrics -> ABC score (and optionally "
                        "-> semantic tokens for items that have them). Load the result with the YuE2 Load LoRA node "
                        "or LoraLoader (clip strength).",
            inputs=[
                io.Clip.Input("clip", tooltip="CLIP from the YuE2 checkpoint."),
                DATASET.Input("dataset", tooltip="Dataset whose items carry ABC scores (transcribed or provided)."),
                io.Boolean.Input("train_abc", default=True, tooltip="Train style+lyrics -> ABC score."),
                io.Boolean.Input("train_semantic", default=False,
                                 tooltip="Train -> semantic codec tokens for items that have them (YuE2 output folders)."),
                io.Combo.Input("abc_mode", options=["full", "melody", "auto"], default="full",
                               tooltip="Planning instruction the LoRA is trained under; use the mode you generate with. "
                                       "auto: chords in the score -> full, else melody."),
                io.Int.Input("max_tokens", default=4096, min=64, max=20000,
                             tooltip="Longest trained span (ABC or codec tokens) per step. Longer scores are trained "
                                     "through head, tail and middle windows (the ending is always learned). 8192 "
                                     "fits most whole songs on a 16 GB card but also teaches the album's song "
                                     "lengths; keep 4096 for normal-length songs."),
                io.Float.Input("regularization_fraction", default=0.5, min=0.0, max=0.9, step=0.05,
                               tooltip="With a regularization dataset connected: share of the training draws taken from "
                                       "its base-model scores instead of your songs. Keeps the planner writing "
                                       "well-formed, normal-length scores while it picks up the album's style. "
                                       "Ignored when nothing is connected."),
                io.Float.Input("kl_weight", default=0.0, min=0.0, max=10.0, step=0.01,
                               tooltip="Trust region: adds kl_weight x KL(base || LoRA) of the next-token distributions on "
                                       "every trained position, with the base model computed by switching the LoRA off. "
                                       "Holds the planner close to base-model writing on your own songs without extra "
                                       "data (complements the regularization scores). 0 = off; 0.5-1 is a normal "
                                       "setting. Costs one extra no-gradient forward per micro-step."),
                io.Float.Input("abc_dropout", default=0.0, min=0.0, max=0.9, step=0.05,
                               tooltip="train_semantic only: share of the music-token draws trained behind the no-sheet "
                                       "prompt (what YuE2GenerateMusic runs with an empty ABC input), so one LoRA also "
                                       "serves generation without a score. 0 = always with the sheet."),
                io.Int.Input("probe_every", default=0, min=0, max=100000,
                             tooltip="Every N steps (and before step 1), write a whole ABC score with the current LoRA "
                                     "through YuE2's own sampler and report its length and whether it ended. A probe "
                                     "that runs to probe_max_tokens is the over-training signal the held-out loss "
                                     "misses. Scores land in output/yue2_probes/<save_name>/. Set it to save_every. "
                                     "0 = off. Each probe takes as long as one YuE2GenerateABC run."),
                io.String.Input("probe_style", default="", multiline=True,
                                tooltip="Style prompt for the probes (blank = the first training song's style)."),
                io.String.Input("probe_lyrics", default="", multiline=True,
                                tooltip="Lyrics for the probes (blank = the first training song's lyrics)."),
                io.Int.Input("probe_max_tokens", default=8192, min=256, max=20000, advanced=True,
                             tooltip="Token budget of a probe score; a probe that uses all of it did not end."),
                io.Int.Input("probe_seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, advanced=True,
                             tooltip="Fixed sampling seed shared by all probes so they are comparable."),
                io.String.Input("probe_abc", default="", multiline=True, advanced=True,
                                tooltip="Fixed ABC score for the samples, so every checkpoint is measured on the same "
                                        "score (blank = the score each probe writes; if that one did not end, the first "
                                        "training song's score)."),
                io.Int.Input("sample_every", default=0, min=0, max=100000,
                             tooltip="Every N steps (and before step 1), write a song clip with the current LoRA: the "
                                     "planner writes a score for the probe prompt (the probe_abc score if given), then "
                                     "sample_seconds of music tokens for it, and with model and vae connected the clip is "
                                     "rendered to output/yue2_probes/<save_name>/step_NNNNNN.wav next to its score and "
                                     "tokens. Step 0 is the base model. Also reports the stream's length, whether it "
                                     "ended and how repetitive it is (the over-training signal for train_semantic). "
                                     "0 = off. About a minute per 60 s of tokens plus the render."),
                io.Float.Input("sample_seconds", default=30.0, min=1.0, max=900.0, step=1.0,
                               tooltip="Length of each sample's music."),
                io.Int.Input("sample_seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, advanced=True,
                             tooltip="Fixed seed for the samples' music tokens and rendering, so steps are comparable."),
                io.Combo.Input("sample_device", options=_render_device_choices(), default="auto",
                               tooltip="GPU that renders the samples (needs model and vae connected). auto = a GPU training "
                                       "does not use; the acoustic model does not fit next to the planner and its training "
                                       "state on a 16 GB card, so with one GPU set off (tokens are still written) and "
                                       "render afterwards with YuE2 Conditioning From Tokens."),
                *_common_training_inputs(5e-5, 100),
                DATASET.Input("regularization", optional=True,
                              tooltip="Scores the BASE model wrote (YuE2 Regularization Scores node, or a folder of YuE2 "
                                      "output directories). Mixed into training at regularization_fraction so the "
                                      "LoRA does not forget how to write and end a score."),
                io.Model.Input("model", optional=True, tooltip="MODEL from the checkpoint: with vae and sample_every, renders the samples."),
                io.Vae.Input("vae", optional=True, tooltip="VAE from the checkpoint: with model and sample_every, renders the samples."),
            ],
            outputs=[LORA_MODEL.Output("lora", display_name="lora"), LOSS_MAP.Output("loss_map", display_name="loss_map"),
                     io.Int.Output("steps", display_name="steps"), io.String.Output("report", display_name="report")],
        )

    @classmethod
    def execute(cls, clip, dataset, train_abc, train_semantic, abc_mode, max_tokens, regularization_fraction,
                kl_weight, abc_dropout,
                probe_every, probe_style, probe_lyrics, probe_max_tokens, probe_seed, probe_abc,
                sample_every, sample_seconds, sample_seed, sample_device, steps, learning_rate, lr_schedule,
                rank, alpha,
                targets, batch_size, grad_accumulation, warmup_steps, seed, optimizer, lora_dtype,
                gradient_checkpointing, max_grad_norm, devices, existing_lora, resume_state, save_every, save_name, log_every,
                eval_every, eval_samples, eval_holdout, keep, ema_decay, tensorboard, tensorboard_dir, regularization=None,
                model=None, vae=None):
        render_callback = None
        if sample_every > 0 and (model is None or vae is None):
            logging.warning("YuE2 trainer: sample_every is set but model/vae are not connected; the samples are written "
                            "as token files only (render them with YuE2 Conditioning From Tokens)")
        elif sample_every > 0:
            target = pick_render_device(sample_device, resolve_devices(devices))
            if target is None:
                logging.warning("YuE2 trainer: no GPU free for rendering the samples (training uses %s, sample_device=%s); "
                                "they are written as token files only (render them with YuE2 Conditioning From Tokens)",
                                devices, sample_device)
            else:
                renderer = Renderer(model, vae, target)
                write_audio = _audio_writer(_probe_dir(save_name), "step")

                def render_callback(step, conditioning, frames, meta):
                    audio, rate = renderer.render(conditioning, frames, sample_seed)
                    return write_audio(step, audio, rate)
        cfg = PlannerConfig(
            steps=steps, batch_size=batch_size, grad_accumulation=grad_accumulation, learning_rate=learning_rate,
            lr_schedule=lr_schedule,
            rank=rank, alpha=alpha, targets=targets, train_abc=train_abc, train_semantic=train_semantic,
            abc_mode=abc_mode, max_tokens=max_tokens, warmup_steps=warmup_steps, max_grad_norm=max_grad_norm,
            seed=seed, lora_dtype=lora_dtype, gradient_checkpointing=gradient_checkpointing, optimizer=optimizer,
            devices=devices, existing_lora=_existing_lora(existing_lora), save_every=save_every,
            resume_state=_resume_state(existing_lora, resume_state), keep=keep, ema_decay=ema_decay,
            regularization_fraction=regularization_fraction, kl_weight=kl_weight, abc_dropout=abc_dropout,
            probe_every=probe_every, probe_style=probe_style, probe_lyrics=probe_lyrics,
            probe_max_tokens=probe_max_tokens, probe_seed=probe_seed, probe_callback=_probe_writer(save_name),
            sample_every=sample_every, sample_seconds=sample_seconds, sample_seed=sample_seed,
            probe_abc=probe_abc, render_callback=render_callback,
            log_every=log_every, eval_every=eval_every, eval_samples=eval_samples, eval_holdout=eval_holdout,
            tensorboard_dir=_tensorboard_dir(tensorboard, tensorboard_dir), run_name=save_name,
        )
        cfg.save_callback = lambda sd, n, info, state=None: _save_checkpoint(sd, save_name, n, {**info, "save_name": save_name}, state)
        pbar = comfy.utils.ProgressBar(steps)

        def progress(done, total, loss):
            pbar.update_absolute(done, total)

        with torch.inference_mode(False):
            result = train_planner_lora(clip, dataset, cfg, progress=progress, interrupt_check=_interrupt,
                                        regularization=regularization)
        result.info["save_name"] = save_name
        report = _report(result)
        return io.NodeOutput(result.lora_sd, _loss_map(result), result.steps, report)


def _loss_map(result) -> dict:
    out = {"loss": result.losses, "eval": result.evals, "info": result.info}
    for key in ("drift", "probe", "kl", "samples"):
        if result.info.get(key):
            out[key] = result.info[key]
    if getattr(result, "state", None):
        out["state"] = result.state
    if getattr(result, "raw_lora_sd", None):
        out["raw_lora"] = result.raw_lora_sd
    return out


def _report(result) -> str:
    losses = result.losses
    head = sum(losses[:10]) / max(1, len(losses[:10]))
    tail = sum(losses[-10:]) / max(1, len(losses[-10:]))
    lines = [f"{result.info.get('kind')} LoRA: {result.steps} steps in {result.seconds / 60:.1f} min",
             f"loss first10={head:.4f} last10={tail:.4f} min={min(losses):.4f}" if losses else "no steps"]
    evals = getattr(result, "evals", None) or []
    label = "held-out" if result.info.get("eval_holdout") else "fixed-set"
    if len(evals) > 1:
        best = min(evals, key=lambda e: e[1])
        lines.append(f"{label} eval loss: {evals[0][1]:.4f} before training -> {evals[-1][1]:.4f} at the end "
                     f"(best {best[1]:.4f} at step {best[0]})")
    if result.info.get("kept_step") not in (None, result.steps):
        lines.append(f"kept the weights from step {result.info['kept_step']} (best evaluation loss "
                     f"{result.info.get('kept_eval', float('nan')):.4f}) instead of the final step")
    if result.info.get("ema_decay"):
        lines.append(f"weights: EMA of the LoRA (decay {result.info['ema_decay']}, {result.info.get('ema_updates', 0)} "
                     "updates); the last step's raw weights are saved next to the final LoRA as <name>_raw.safetensors")
    drift = result.info.get("drift") or []
    if len(drift) > 1:
        lines.append(f"regularizer loss (base-model scores): {drift[0][1]:.4f} -> {drift[-1][1]:.4f}")
    kls = result.info.get("kl") or []
    if kls:
        lines.append(f"KL to the base model (weight {result.info.get('kl_weight', 0)}): first10={sum(kls[:10]) / len(kls[:10]):.4f} "
                     f"last10={sum(kls[-10:]) / len(kls[-10:]):.4f} max={max(kls):.4f}")
    probes = result.info.get("probe") or []
    if probes:
        lines.append("probes: " + "; ".join(probe_summary(p).replace("(budget hit)", "(BUDGET HIT, did not end)")
                                            for p in probes))
        if probes[-1].get("path"):
            lines.append(f"probe scores: {Path(probes[-1]['path']).parent}")
    samples = result.info.get("samples") or []
    if samples:
        lines.append("rendered samples: " + "; ".join(f"step {s['step']}: {s['seconds']} s" for s in samples))
        if samples[-1].get("path"):
            lines.append(f"samples: {Path(samples[-1]['path']).parent}")
    if result.info.get("tensorboard"):
        lines.append(f"tensorboard run: {result.info['tensorboard']}")
    lines.append(json.dumps({k: v for k, v in result.info.items() if k not in ("eval", "drift", "probe", "kl", "samples")},
                            ensure_ascii=False))
    return "\n".join(lines)


class YuE2TrainerSaveLoRA(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YuE2TrainerSaveLoRA",
            display_name="YuE2 Save LoRA",
            category=CATEGORY,
            description="Saves a trained LoRA straight into models/loras so LoraLoader can pick it up.",
            is_output_node=True,
            inputs=[
                LORA_MODEL.Input("lora"),
                io.String.Input("name", default="",
                                tooltip="File name. Leave blank to use the trainer's save_name (connect loss_map)."),
                io.Boolean.Input("add_timestamp", default=True),
                LOSS_MAP.Input("loss_map", optional=True, tooltip="Connect the trainer's loss_map: supplies save_name and metadata."),
            ],
            outputs=[io.String.Output("path", display_name="path")],
        )

    @classmethod
    def execute(cls, lora, name, add_timestamp, loss_map=None):
        if not name.strip() and loss_map:
            name = str((loss_map.get("info") or {}).get("save_name") or "")
        name = "".join(c for c in name.strip() if c not in '\\/:*?"<>|') or "yue2_lora"
        if add_timestamp:
            name = f"{name}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        target = _lora_dir() / f"{name}.safetensors"
        target.parent.mkdir(parents=True, exist_ok=True)
        info = dict((loss_map or {}).get("info", {}))
        if loss_map and loss_map.get("loss"):
            info["final_loss"] = float(loss_map["loss"][-1])
        save_lora_file(lora, target, info)
        if loss_map and loss_map.get("state"):
            save_state(loss_map["state"], state_path(target))
            logging.info("YuE2 trainer: saved resume state next to %s", target)
        if loss_map and loss_map.get("raw_lora"):
            raw_target = target.with_name(f"{name}_raw.safetensors")
            save_lora_file(loss_map["raw_lora"], raw_target, {**info, "weights": "raw (not averaged)"})
            logging.info("YuE2 trainer: saved the last step's raw (not averaged) weights as %s", raw_target)
        return io.NodeOutput(str(target))


class YuE2TrainerRegularizationScores(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YuE2TrainerRegularizationScores",
            display_name="YuE2 Regularization Scores",
            category=CATEGORY,
            description="Writes ABC scores with the BASE model (no LoRA) for a few prompts and returns them as a dataset "
                        "for the planner trainer's regularization input. Like DreamBooth's class images: mixed into "
                        "training, they keep the planner writing well-formed, normal-length scores. Scores are saved "
                        "under output/<folder> as YuE2 output directories and reused on the next run. With "
                        "music_seconds the base model also writes its music tokens for every score, so the set "
                        "covers train_semantic as well.",
            inputs=[
                io.Clip.Input("clip", tooltip="CLIP from the YuE2 checkpoint WITHOUT any LoRA applied."),
                io.String.Input("styles", default="", multiline=True,
                                tooltip="One style prompt per line. Blank: the style prompts of the connected dataset's "
                                        "songs are used (each with that song's lyrics)."),
                io.String.Input("lyrics", default="", multiline=True,
                                tooltip="Lyrics used with every prompt line above (blank = instrumental). Section tags "
                                        "like [Verse] / [Chorus] as usual."),
                io.Int.Input("scores_per_prompt", default=2, min=1, max=32,
                             tooltip="Scores written per prompt (different seeds)."),
                io.Combo.Input("mode", options=["full", "melody"], default="full",
                               tooltip="Planning mode; use the abc_mode you train the planner with."),
                io.Float.Input("music_seconds", default=0.0, min=0.0, max=900.0, step=1.0,
                               tooltip="Also have the base model write its music tokens for every score, up to this many "
                                       "seconds, and keep them as semantic.npy: the regularization set then covers "
                                       "train_semantic as well (prior preservation for the music-token target). 0 = "
                                       "scores only. About a minute per 60 s per score; reused on later runs."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Int.Input("max_abc_tokens", default=8192, min=256, max=20000, advanced=True,
                             tooltip="Scores that do not end within this budget are discarded (they would teach endless scores)."),
                io.Float.Input("temperature", default=PROBE_SAMPLING["temperature"], min=0.0, max=5.0, step=0.05, advanced=True),
                io.Float.Input("top_p", default=PROBE_SAMPLING["top_p"], min=0.01, max=1.0, step=0.01, advanced=True),
                io.Int.Input("top_k", default=PROBE_SAMPLING["top_k"], min=1, max=1000, advanced=True),
                io.Float.Input("repetition_penalty", default=PROBE_SAMPLING["repetition_penalty"], min=0.01, max=10.0,
                               step=0.005, advanced=True),
                io.String.Input("folder", default="yue2_regularization",
                                tooltip="Where the scores are kept (relative paths live in ComfyUI/output). Existing "
                                        "scores for the same prompt/seed/mode are reused, not regenerated."),
                DATASET.Input("dataset", optional=True,
                              tooltip="Training dataset: its songs' style prompts and lyrics are used when styles is blank."),
            ],
            outputs=[DATASET.Output("regularization", display_name="regularization"),
                     io.String.Output("report", display_name="report")],
        )

    @classmethod
    def execute(cls, clip, styles, lyrics, scores_per_prompt, mode, music_seconds, seed, max_abc_tokens, temperature,
                top_p, top_k, repetition_penalty, folder, dataset=None):
        import hashlib
        prompts = [(line.strip(), lyrics) for line in styles.splitlines() if line.strip()]
        if not prompts:
            if dataset is None or not dataset.items:
                raise ValueError("Enter at least one style prompt or connect a dataset to take the prompts from")
            prompts = [(item.style, item.lyrics) for item in dataset.items if item.style]
        root = Path(folder.strip().strip('"') or "yue2_regularization")
        if not root.is_absolute():
            root = Path(folder_paths.get_output_directory()) / root
        sampling = {"temperature": temperature, "top_p": top_p, "top_k": top_k, "repetition_penalty": repetition_penalty}
        items, written, reused, dropped, music_written, music_reused = [], 0, 0, 0, 0, 0
        total = len(prompts) * scores_per_prompt
        pbar = comfy.utils.ProgressBar(total)
        for p_index, (style, lyric) in enumerate(prompts):
            for copy in range(scores_per_prompt):
                _interrupt()
                this_seed = (seed + p_index * 1000 + copy) % (2 ** 63)
                key = hashlib.sha1(json.dumps([style, lyric, mode, this_seed, sampling], ensure_ascii=False).encode()).hexdigest()[:12]
                folder_i = root / f"reg_{p_index:03d}_{copy:02d}_{key}"
                request = folder_i / "request.json"
                score = folder_i / "score.abc"
                if request.is_file() and score.is_file():
                    reused += 1
                else:
                    abc, count, ended = generate_abc(clip, style, lyric, mode, this_seed, max_abc_tokens, sampling)
                    if not ended:
                        logging.warning("YuE2 trainer: regularization score for prompt %d seed %d used the whole %d-token "
                                        "budget without ending; discarded", p_index, this_seed, max_abc_tokens)
                        dropped += 1
                        pbar.update_absolute(p_index * scores_per_prompt + copy + 1, total)
                        continue
                    folder_i.mkdir(parents=True, exist_ok=True)
                    score.write_text(abc, encoding="utf-8")
                    request.write_text(json.dumps({"id": folder_i.name, "style": style, "lyrics": lyric, "cot": mode,
                                                   "seed": this_seed, "abc_tokens": count, "sampling": sampling,
                                                   "generated_by": "YuE2TrainerRegularizationScores"},
                                                  indent=1, ensure_ascii=False), encoding="utf-8")
                    written += 1
                abc_text = score.read_text(encoding="utf-8")
                semantic, extra = None, {}
                if music_seconds > 0:
                    music_file = folder_i / "semantic.npy"
                    meta = json.loads(request.read_text(encoding="utf-8"))
                    if music_file.is_file() and (meta.get("semantic_ended")
                                                 or float(meta.get("music_seconds", 0)) >= music_seconds):
                        music_reused += 1
                    else:
                        _interrupt()
                        codes, music_ended = generate_music(clip, style, lyric, abc_text, mode, this_seed, music_seconds)
                        np.save(music_file, np.asarray(codes, dtype=np.int32))
                        meta.update(music_seconds=music_seconds, music_seed=this_seed, semantic_tokens=len(codes),
                                    semantic_ended=music_ended, music_sampling=MUSIC_SAMPLING)
                        request.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
                        music_written += 1
                        logging.info("YuE2 trainer: base-model music tokens for %s: %d tokens (%.1f s), %s", folder_i.name,
                                     len(codes), len(codes) / FRAMES_PER_SECOND, "ended" if music_ended else "budget reached")
                    meta = json.loads(request.read_text(encoding="utf-8"))
                    semantic = [int(t) for t in np.load(music_file).reshape(-1).tolist()] or None
                    extra = {"semantic_ended": bool(meta.get("semantic_ended", True))}
                items.append(Item(id=folder_i.name, audio_path=None, style=style, lyrics=lyric, abc=abc_text,
                                  semantic=semantic, source="regularization", extra=extra))
                pbar.update_absolute(p_index * scores_per_prompt + copy + 1, total)
        if not items:
            raise ValueError("No regularization scores: every score ran past max_abc_tokens; raise it or lower temperature")
        music = (f", music tokens for {sum(1 for item in items if item.semantic)} of them "
                 f"({music_written} written, {music_reused} reused)" if music_seconds > 0 else "")
        report = (f"{len(items)} regularization scores from {len(prompts)} prompts ({written} written, {reused} reused, "
                  f"{dropped} discarded for not ending){music} in {root}")
        logging.info("YuE2 trainer: %s", report)
        return io.NodeOutput(Dataset(items=items, meta={"folder": str(root), "kind": "regularization"}), report)


def _output_file(path: str) -> Path:
    p = Path(path.strip().strip('"'))
    if not p.is_absolute():
        p = Path(folder_paths.get_output_directory()) / p
    if not p.is_file():
        raise FileNotFoundError(f"Token file not found: {p}")
    return p


class YuE2TrainerTokensConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YuE2TrainerTokensConditioning",
            display_name="YuE2 Conditioning From Tokens",
            category=CATEGORY,
            description="Acoustic-stage conditioning for a saved music-token stream, in place of YuE2 Generate Music: "
                        "render a probe's step_NNNNNN.semantic.npy to hear what a checkpoint wrote, or a song's "
                        ".semantic.npy sidecar to hear the tokenizer round trip. Feed the outputs to Empty YuE2 Latent "
                        "Audio and the KSampler as usual. With a probe file, style, lyrics and score default to the "
                        "ones the probe used (its .json); with a dataset sidecar, to the song's own sidecars.",
            inputs=[
                io.Clip.Input("clip", tooltip="CLIP from the checkpoint, with the same LoRA the tokens were written with."),
                io.String.Input("tokens", default="yue2_probes/yue2_semantic_planner/step_000030.semantic.npy",
                                tooltip="A .semantic.npy file: absolute, or relative to ComfyUI/output."),
                io.String.Input("style", default="", multiline=True, tooltip="Blank = the probe's / song's style prompt."),
                io.String.Input("lyrics", default="", multiline=True, tooltip="Blank = the probe's / song's lyrics."),
                io.String.Input("abc", default="", multiline=True,
                                tooltip="Score the tokens were written under (blank = the probe's / song's score)."),
                io.Combo.Input("mode", options=["auto", "full", "melody", "off"], default="auto",
                               tooltip="Planning mode of the prompt; auto = the probe's, else full/melody by the score "
                                       "(off without a score)."),
                io.Float.Input("max_seconds", default=0.0, min=0.0, max=900.0, step=1.0,
                               tooltip="Render only the first N seconds of the stream (0 = all)."),
            ],
            outputs=[io.Conditioning.Output(display_name="conditioning"), io.Float.Output(display_name="seconds"),
                     io.String.Output(display_name="info")],
        )

    @classmethod
    def execute(cls, clip, tokens, style, lyrics, abc, mode, max_seconds):
        path = _output_file(tokens)
        got = tokens_and_prompt(path, style, lyrics, abc, mode, max_seconds)
        codes, style, lyrics, abc, mode = got["codes"], got["style"], got["lyrics"], got["abc"], got["mode"]
        with torch.inference_mode():
            conditioning = conditioning_from_tokens(clip, style, lyrics, abc, mode, codes)
        seconds = len(codes) / FRAMES_PER_SECOND
        info = (f"{path.name}: {len(codes)} tokens ({seconds:.1f} s), mode {mode if abc and abc.strip() else 'off'}, "
                f"{len(set(codes)) / max(1, len(codes)) * 100:.0f}% distinct, style {style[:60]!r}")
        logging.info("YuE2 trainer: %s", info)
        return io.NodeOutput(conditioning, seconds, info)


class YuE2TrainerMergeLoRA(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YuE2TrainerMergeLoRA",
            display_name="YuE2 Merge LoRAs",
            category=CATEGORY,
            description="Writes an acoustic LoRA and a planner LoRA into one file (their keys never overlap), so a "
                        "single YuE2 Load LoRA node with both model and clip connected applies both. Saved to "
                        "models/loras/<name>.safetensors with both trainings' metadata.",
            inputs=[
                io.Combo.Input("lora_a", options=folder_paths.get_filename_list("loras"), tooltip="Typically the acoustic LoRA."),
                io.Combo.Input("lora_b", options=folder_paths.get_filename_list("loras"), tooltip="Typically the planner LoRA."),
                io.String.Input("name", default="yue2_merged"),
                io.Boolean.Input("add_timestamp", default=False),
            ],
            outputs=[io.String.Output(display_name="path"), io.String.Output(display_name="report")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, lora_a, lora_b, name, add_timestamp):
        safe = "".join(c for c in name.strip() if c not in '\\/:*?"<>|') or "yue2_merged"
        if add_timestamp:
            safe += "_" + _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        target = _lora_dir() / f"{safe}.safetensors"
        paths = [folder_paths.get_full_path_or_raise("loras", lora_a), folder_paths.get_full_path_or_raise("loras", lora_b)]
        merged, meta = merge_lora_files(paths, target)
        report = (f"{target.name}: {len(merged)} tensors ({meta['model_keys']} model, {meta['clip_keys']} clip) from "
                  + " + ".join(f"{p['file']} ({p.get('kind') or 'unknown'}, {p['keys']} tensors)" for p in meta["parts"]))
        logging.info("YuE2 trainer: merged %s", report)
        return io.NodeOutput(str(target), report)


class YuE2TrainerLoadLoRA(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YuE2TrainerLoadLoRA",
            display_name="YuE2 Load LoRA",
            category=CATEGORY,
            description="Applies an acoustic (MODEL) and/or planner (CLIP) YuE2 LoRA. Either input may be left unconnected.",
            inputs=[
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                io.Float.Input("strength_model", default=1.0, min=-10.0, max=10.0, step=0.01),
                io.Float.Input("strength_clip", default=1.0, min=-10.0, max=10.0, step=0.01),
                io.Model.Input("model", optional=True),
                io.Clip.Input("clip", optional=True),
            ],
            outputs=[io.Model.Output("model_out", display_name="model"), io.Clip.Output("clip_out", display_name="clip")],
        )

    @classmethod
    def execute(cls, lora_name, strength_model, strength_clip, model=None, clip=None):
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = comfy.utils.load_torch_file(path, safe_load=True)
        new_model, new_clip = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return io.NodeOutput(new_model if model is not None else None, new_clip if clip is not None else None)


class YuE2TrainerExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [
            YuE2TrainerDatasetFolder,
            YuE2TrainerPrepareDataset,
            YuE2TrainerDatasetFromAudio,
            YuE2TrainerDatasetMerge,
            YuE2TrainerSemanticTokens,
            YuE2TrainerEncodeDataset,
            YuE2TrainerAcousticLoRA,
            YuE2TrainerPlannerLoRA,
            YuE2TrainerRegularizationScores,
            YuE2TrainerSaveLoRA,
            YuE2TrainerLoadLoRA,
            YuE2TrainerMergeLoRA,
            YuE2TrainerTokensConditioning,
        ]
