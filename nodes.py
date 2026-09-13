"""ComfyUI nodes for training YuE2 LoRAs (acoustic MODEL LoRA and planner CLIP LoRA)."""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
from pathlib import Path

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
from .yue2_trainer.dataset import Dataset, Item, cache_key, load_cache, save_cache, scan_folder
from .yue2_trainer.lora import TARGET_PRESETS, load_lora_file, save_lora_file
from .yue2_trainer.parallel import device_choices
from .yue2_trainer.sidecars import DEFAULT_CLAUDE, PRECISIONS, SidecarConfig, WHISPER_CHOICES, prepare_folder, summarize
from .yue2_trainer.planner import PlannerConfig, train_planner_lora

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


def _lora_choices():
    return ["[None]"] + folder_paths.get_filename_list("loras")


def _tensorboard_dir(enabled: bool, folder: str):
    if not enabled:
        return ""
    path = Path((folder or "yue2_tensorboard").strip().strip('"'))
    if not path.is_absolute():
        path = Path(folder_paths.get_output_directory()) / path
    return str(path)


def _save_checkpoint(lora_sd, name: str, steps: int, info: dict) -> str:
    target = _lora_dir() / f"{name}_{steps:06d}.safetensors"
    save_lora_file(lora_sd, target, info)
    logging.info("YuE2 trainer: saved intermediate LoRA %s", target)
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
                        scores = audio_encoder.generate_abc(waveform[None], 48000, melody_only=transcribe == "melody")
                    item.abc = scores[0] if isinstance(scores, (list, tuple)) else scores
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
        io.Int.Input("save_every", default=0, min=0, max=100000, advanced=True,
                     tooltip="Write an intermediate LoRA to models/loras every N steps (0 = off)."),
        io.String.Input("save_name", default="yue2_lora",
                        tooltip="Name of the LoRA: used by YuE2 Save LoRA (when its name is blank), for intermediate "
                                "saves and for the TensorBoard run."),
        io.Int.Input("log_every", default=1, min=1, max=10000,
                     tooltip="Print step / loss / lr / grad-norm / ETA to the console every N steps."),
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
                io.Combo.Input("prefix_mode", options=["full", "melody", "off", "auto"], default="full",
                               tooltip="Planning instruction in the conditioning prefix; match the mode you generate with. "
                                       "Items without an ABC score always use off. auto: chords -> full, else melody."),
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
                *_common_training_inputs(1e-4, 300),
            ],
            outputs=[LORA_MODEL.Output("lora", display_name="lora"), LOSS_MAP.Output("loss_map", display_name="loss_map"),
                     io.Int.Output("steps", display_name="steps"), io.String.Output("report", display_name="report")],
        )

    @classmethod
    def execute(cls, model, clip, dataset, segment_seconds, prefix_mode, use_semantic_tokens, train_acoustic_head,
                caption_dropout, timestep_sampling, shift, steps, learning_rate, lr_schedule, rank, alpha, targets, batch_size, grad_accumulation,
                warmup_steps, seed, optimizer, lora_dtype, gradient_checkpointing, max_grad_norm, devices,
                existing_lora, save_every, save_name, log_every, tensorboard, tensorboard_dir):
        cfg = AcousticConfig(
            steps=steps, batch_size=batch_size, grad_accumulation=grad_accumulation, learning_rate=learning_rate,
            lr_schedule=lr_schedule,
            rank=rank, alpha=alpha, targets=targets, train_acoustic_head=train_acoustic_head,
            segment_seconds=segment_seconds, mode=prefix_mode, use_semantic_tokens=use_semantic_tokens,
            caption_dropout=caption_dropout,
            timestep_sampling=timestep_sampling, shift=shift, warmup_steps=warmup_steps, max_grad_norm=max_grad_norm,
            seed=seed, lora_dtype=lora_dtype, gradient_checkpointing=gradient_checkpointing, optimizer=optimizer,
            devices=devices, existing_lora=_existing_lora(existing_lora), save_every=save_every,
            log_every=log_every, tensorboard_dir=_tensorboard_dir(tensorboard, tensorboard_dir), run_name=save_name,
        )
        cfg.save_callback = lambda sd, n, info: _save_checkpoint(sd, save_name, n, {**info, "save_name": save_name})
        pbar = comfy.utils.ProgressBar(steps)

        def progress(done, total, loss):
            pbar.update_absolute(done, total)

        with torch.inference_mode(False):
            result = train_acoustic_lora(model, clip, dataset, cfg, progress=progress, interrupt_check=_interrupt)
        result.info["save_name"] = save_name
        report = _report(result)
        return io.NodeOutput(result.lora_sd, {"loss": result.losses, "info": result.info}, result.steps, report)


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
                             tooltip="Random crop of the trained span (ABC or codec tokens) per step."),
                *_common_training_inputs(1e-4, 300),
            ],
            outputs=[LORA_MODEL.Output("lora", display_name="lora"), LOSS_MAP.Output("loss_map", display_name="loss_map"),
                     io.Int.Output("steps", display_name="steps"), io.String.Output("report", display_name="report")],
        )

    @classmethod
    def execute(cls, clip, dataset, train_abc, train_semantic, abc_mode, max_tokens, steps, learning_rate, lr_schedule,
                rank, alpha,
                targets, batch_size, grad_accumulation, warmup_steps, seed, optimizer, lora_dtype,
                gradient_checkpointing, max_grad_norm, devices, existing_lora, save_every, save_name, log_every,
                tensorboard, tensorboard_dir):
        cfg = PlannerConfig(
            steps=steps, batch_size=batch_size, grad_accumulation=grad_accumulation, learning_rate=learning_rate,
            lr_schedule=lr_schedule,
            rank=rank, alpha=alpha, targets=targets, train_abc=train_abc, train_semantic=train_semantic,
            abc_mode=abc_mode, max_tokens=max_tokens, warmup_steps=warmup_steps, max_grad_norm=max_grad_norm,
            seed=seed, lora_dtype=lora_dtype, gradient_checkpointing=gradient_checkpointing, optimizer=optimizer,
            devices=devices, existing_lora=_existing_lora(existing_lora), save_every=save_every,
            log_every=log_every, tensorboard_dir=_tensorboard_dir(tensorboard, tensorboard_dir), run_name=save_name,
        )
        cfg.save_callback = lambda sd, n, info: _save_checkpoint(sd, save_name, n, {**info, "save_name": save_name})
        pbar = comfy.utils.ProgressBar(steps)

        def progress(done, total, loss):
            pbar.update_absolute(done, total)

        with torch.inference_mode(False):
            result = train_planner_lora(clip, dataset, cfg, progress=progress, interrupt_check=_interrupt)
        result.info["save_name"] = save_name
        report = _report(result)
        return io.NodeOutput(result.lora_sd, {"loss": result.losses, "info": result.info}, result.steps, report)


def _report(result) -> str:
    losses = result.losses
    head = sum(losses[:10]) / max(1, len(losses[:10]))
    tail = sum(losses[-10:]) / max(1, len(losses[-10:]))
    lines = [f"{result.info.get('kind')} LoRA: {result.steps} steps in {result.seconds / 60:.1f} min",
             f"loss first10={head:.4f} last10={tail:.4f} min={min(losses):.4f}" if losses else "no steps"]
    if result.info.get("tensorboard"):
        lines.append(f"tensorboard run: {result.info['tensorboard']}")
    lines.append(json.dumps(result.info, ensure_ascii=False))
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
        return io.NodeOutput(str(target))


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
            YuE2TrainerEncodeDataset,
            YuE2TrainerAcousticLoRA,
            YuE2TrainerPlannerLoRA,
            YuE2TrainerSaveLoRA,
            YuE2TrainerLoadLoRA,
        ]
