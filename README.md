# ComfyUI-YuE2-Trainer

Train LoRAs for [YuE2](https://github.com/multimodal-art-projection/YuE) inside ComfyUI, and use them
with the native YuE2 nodes. Two kinds of LoRA are produced, matching the two halves of the
`yue2_3b_bf16.safetensors` checkpoint:

| LoRA | Trains | Learns | Load with |
|---|---|---|---|
| **Acoustic (MODEL)** | the NAR flow-matching path (the `MODEL` output of the checkpoint loader) on VAE latents of your songs | timbre, production, mix character, instrument sound | `LoraLoaderModelOnly` or **YuE2 Load LoRA** |
| **Planner (CLIP)** | the AR language model (the `CLIP` output) on `style + lyrics -> ABC score`, optionally `-> semantic tokens` | melodic / harmonic writing style; song structure | **YuE2 Load LoRA** or `LoraLoader` (clip strength) |

Training runs entirely with ComfyUI's own model code, memory management and LoRA machinery, so the
resulting `.safetensors` files are ordinary ComfyUI LoRAs.

## Requirements

- **ComfyUI with native YuE2 support** (September 2026 or newer: `comfy/ldm/yue2`, `YuE2GenerateMusic`
  node). Older builds cannot even load the checkpoint. Update with `git pull` in your ComfyUI folder and
  `pip install -r requirements.txt` in its venv.
- `models/checkpoints/yue2_3b_bf16.safetensors` from [Comfy-Org/YuE2](https://huggingface.co/Comfy-Org/YuE2)
  (the BF16 file; the INT8 repack is not suitable for training).
- Optional: `models/audio_encoders/sheetsage2_bf16.safetensors` to transcribe ABC scores from audio
  (needed for the planner LoRA unless you supply `.abc` files yourself).
- NVIDIA GPU with BF16. Acoustic training with 30 s segments fits comfortably in 16 GB; planner training
  with `max_tokens` 4096 fits in 16 GB. Whole-song acoustic segments (`segment_seconds = 0`) on 4-minute
  songs want 24 GB.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/speedyrulz/ComfyUI-YuE2-Trainer.git
```

or drop/symlink this folder into `custom_nodes`, then `pip install -r requirements.txt` with ComfyUI's Python
(`soundfile`, `librosa`, `mutagen`, `requests`; `anthropic` only for Claude section tagging). Everything else ships with ComfyUI.

## Dataset

Put songs in a folder. Each audio file (`.wav .flac .mp3 .ogg .m4a ...`) may have sidecars with the same stem:

```
songs/
  track1.mp3
  track1.style.txt      genre, instruments, vocal character, language, tempo   (what you would put in "style")
  track1.lyrics.txt     [Verse] ... [Chorus] ...   (track1.txt also works)
  track1.abc            optional ABC score (melody or with chords)
  track1.json           optional: {"style": ..., "lyrics": ..., "abc": ..., "semantic": [...]}
  track1.semantic.npy   optional YuE2 semantic tokens (25 per second)
```

Files without a style sidecar use the node's `default_style`; instrumental tracks can leave lyrics empty.
Output folders written by the native `yue2` runtime (`save_artifacts`: `request.json`, `audio.flac`,
`score.abc`, `semantic.npy`, `latent.npy`) are picked up as-is, including their exact semantic tokens
and latents. That is the only source of semantic tokens: the audio-to-semantic tokenizer used to train
YuE2 is not public (see *Limitations*).

### Automatic style and lyrics sidecars

**YuE2 Prepare Dataset** (node) or `prepare_dataset.py folder/` (CLI) writes the sidecars for you:

| field | how it is produced |
|---|---|
| lyrics | `lrclib` looks the song up by the file's artist/title tags or an `Artist - Title` file name (exact words, but only for released songs). `whisper` (default fallback, large-v3) or `moss-audio` transcribe the demucs-isolated vocal stem; Whisper is markedly better on sung lyrics, MOSS-Audio tends to loop. The `lrclib+...` modes look up first and transcribe only when there is no hit; every database hit is cross-checked against a transcript and rejected if the words don't match. Tracks with almost no singing get an empty lyrics file (instrumental). |
| `[Verse]` / `[Chorus]` tags | `heuristic` (repeated stanzas become choruses; long silent openings get `[Intro]`) or `claude` (Claude formats and fixes ASR slips; needs `ANTHROPIC_API_KEY`, falls back to the heuristic on any error) |
| style | `moss-audio` (default): MOSS-Audio-4B-Instruct listens to three excerpts and writes a descriptive tag line, e.g. `Mandarin pop, electronic pop, male vocal, clear and expressive, synthesizer, drum machine, synth bass, 1980s synth-pop, danceable rhythm`. `qwen-omni`: Qwen2.5-Omni-3B, same idea, a bit less specific. `clap`: fixed-vocabulary zero-shot tags. All get the librosa tempo appended, and the detected language when Whisper runs. `artist` is prepended to every style line. |
| `precision` | weight precision of the audio LLM: `bf16` (MOSS about 10 GB VRAM, Omni about 8 GB) or `nf4` (MOSS measured at 4.2 GB peak; needs `bitsandbytes`). GGUF is not an option here: MOSS-Audio's architecture isn't supported by llama.cpp, and the community GGUF build targets a separate C++ runtime. |

Existing sidecars are never overwritten unless `overwrite` is on, so you can hand-correct files and rerun.
The node outputs the scanned dataset (chain it into **YuE2 Encode Dataset**) plus a report that lists which files
came from transcription and deserve a read-through; the folder also gets a `_prepare_report.json` with details.
Models download from Hugging Face on first use (MOSS-Audio about 10 GB, Whisper large-v3 about 3 GB, demucs about
0.3 GB, CLAP about 0.6 GB). The MOSS-Audio model code is vendored under `yue2_trainer/vendor/moss_audio` (Apache-2.0).

```bash
C:/ai/ComfyUI/venv/Scripts/python.exe prepare_dataset.py D:/songs --sections claude
```

## Nodes (category `YuE2/training`)

| Node | Purpose |
|---|---|
| **YuE2 Dataset From Folder** | Scan a folder (absolute path or a folder inside `ComfyUI/input`) into a `YUE2_DATASET`. |
| **YuE2 Prepare Dataset** | Generate missing `.style.txt` / `.lyrics.txt` sidecars (LRCLIB + Whisper, CLAP tags) and output the scanned dataset. |
| **YuE2 Dataset From Audio** | One-item dataset from a `LoadAudio` output plus style / lyrics / ABC (chain with `append_to`). |
| **YuE2 Merge Datasets** | Concatenate two datasets. |
| **YuE2 Encode Dataset** | VAE-encode every item to latents (cached under `output/yue2_trainer_cache`), optionally transcribe missing ABC with a SheetSage2 `AUDIO_ENCODER`. |
| **YuE2 Train Acoustic LoRA (MODEL)** | Flow-matching LoRA training of the acoustic model. Outputs `LORA_MODEL`, `LOSS_MAP`, steps, a text report. |
| **YuE2 Train Planner LoRA (CLIP)** | Next-token LoRA training of the language model on ABC (and semantic tokens when present). |
| **YuE2 Save LoRA** | Writes the LoRA to `models/loras/<name>.safetensors` with training metadata; with `name` blank and `loss_map` connected it uses the trainer's `save_name`. The core `SaveLoRA` node also works (it writes to `output/`). |
| **YuE2 Load LoRA** | Applies a LoRA to MODEL and/or CLIP; either input can be left unconnected. |

The core `LossGraphNode` plots the `LOSS_MAP` output; `PreviewAny` shows the report and dataset summary.

### Typical graphs

Ready-to-load API-format graphs are in [`example_workflows/`](example_workflows) (drag onto the canvas
or use *Load*; recent frontends import API-format JSON):

- `yue2_train_acoustic_lora_api.json` – checkpoint → dataset → encode → acoustic LoRA → save + loss plot
- `yue2_prepare_and_train_acoustic_api.json` – same, but **Prepare Dataset** generates the style/lyrics sidecars first
- `yue2_train_planner_lora_api.json` – same with SheetSage2 transcription (`full`, with chords) → planner LoRA
- `yue2_generate_with_lora_api.json` – the stock YuE2 generation graph with **YuE2 Load LoRA** between the
  checkpoint loader and the YuE2 nodes

Generation with a LoRA is the standard graph: `CheckpointLoaderSimple → YuE2 Load LoRA → YuE2GenerateABC /
YuE2GenerateMusic → ModelSamplingAuraFlow(shift 3) → KSampler(64 steps, cfg 2, euler, simple) → VAEDecodeAudio`.
An acoustic LoRA only affects the KSampler stage; a planner LoRA only affects the two YuE2 generate nodes.

### Settings that matter

**Acoustic LoRA**

- `segment_seconds` (30): random crop per step. The crop keeps its absolute position inside the song
  (RoPE positions and the latent position table), so training crops look like inference. `0` trains whole
  songs (more VRAM).
- `prefix_mode` (full): planning instruction used in the conditioning prefix; match the mode you generate with.
  Items without an ABC score always use `off`. `auto` picks `full` when the ABC has chord symbols, else `melody`.
- `use_semantic_tokens` (off): when on, items with semantic tokens (YuE2 output folders) are conditioned exactly
  like inference (prefix + codec tokens). Otherwise items use the model's codec-dropout ("text-only") conditioning: only the
  text/ABC prefix is visible, and the NAR tokens keep the positions they would have after the codec tokens.
- `train_acoustic_head` (off): also adapt `vae2llm` / `llm2vae` / time embedder projections.
- `timestep_sampling` / `shift`: sigma distribution (uniform by default, matching the reference solver).
- `rank`/`alpha` (16/16 → scale 1), `learning_rate` (1e-4), `lr_schedule` = `cosine` (decay to 10%), `constant`,
  or `linear`; `warmup_steps` ramps up first in every mode. Same options on the planner node.
- `save_every` writes `models/loras/<save_name>_<steps>.safetensors` checkpoints; `existing_lora` resumes.

**Planner LoRA**

- `train_abc` (on): `style + lyrics -> ABC + </abc>`; the loss covers only the ABC tokens.
- `train_semantic` (off): `style + lyrics + ABC -> semantic tokens` for items that carry them.
- `abc_mode` (full): the planning instruction the LoRA is trained under; match the mode you generate with. `auto` picks `full` when the score has chords, else `melody`.
- `max_tokens` (4096): random crop of the trained span so long scores fit in memory.

Both trainers use gradient checkpointing, bf16 autocast, fp32 LoRA weights, grad clipping, and run
one item per micro-step (`batch_size × grad_accumulation` items per optimizer step).

### Watching a run

Every step (or every `log_every` steps) the console shows

```
YuE2 acoustic step 120/300  loss 0.8412  avg20 0.8630  lr 7.65e-05 grad 0.412  elapsed 4:10  eta 6:15
```

Turn on `tensorboard` to also log `loss/step`, `loss/avg20`, `lr` and `grad_norm` per step, plus the run
configuration and final result as text. Runs land in `ComfyUI/output/yue2_tensorboard/<save_name>_<timestamp>`
(`tensorboard_dir` changes the parent folder); view them with

```bash
tensorboard --logdir ComfyUI/output/yue2_tensorboard
```

The CLI equivalent is `--tensorboard DIR` (plus `--log-every N`, `--run-name`).

### Choosing GPUs / training on both GPUs

Both training nodes (and the CLI's `--devices`) take a `devices` setting:

| value | behaviour |
|---|---|
| `auto` | ComfyUI's default device (the one the server runs on) |
| `cuda:1` (any listed GPU) | train on that GPU only; the model is loaded there through ComfyUI's model management |
| `all` | **data parallel on every GPU**: each GPU holds a full copy of the frozen base model and its own LoRA copy; per step each GPU processes its share of the micro-batches in its own thread, LoRA gradients are summed on the first GPU, the optimizer steps there and the LoRA weights are broadcast back |
| `cuda:0,cuda:1` (CLI) | explicit subset for data parallel |

With `all`, set `batch_size × grad_accumulation` to at least the number of GPUs (2 with two cards) so every
GPU gets work each step; the throughput gain is then close to the GPU count because only the LoRA weights
(a few million values) cross PCIe. Each GPU needs room for the full model (about 3 GB for the acoustic
model, 4.5 GB for the planner) plus activations, exactly as single-GPU training does. Mixed GPUs are fine:
the step waits for the slower card.

## Headless CLI

`train_cli.py` runs the same code without the server, using ComfyUI's Python:

```bash
C:/ai/ComfyUI/venv/Scripts/python.exe train_cli.py acoustic --comfy-root C:/ai/ComfyUI ^
    --data D:/songs --style "English, indie pop, breathy female vocal, 100 BPM" ^
    --steps 400 --segment-seconds 30 --out indie_pop_acoustic
```

```bash
C:/ai/ComfyUI/venv/Scripts/python.exe train_cli.py planner --comfy-root C:/ai/ComfyUI ^
    --data D:/songs --transcribe melody --steps 400 --out indie_pop_planner
```

`--out` names go to `models/loras`; `--save-every N` writes checkpoints; `--existing-lora` resumes;
`--dry-run` prepares everything and trains 0 steps. `--devices cuda:1` picks a GPU, `--devices all` uses every
GPU data-parallel (see *Choosing GPUs*).

## How it works

The Comfy-Org checkpoint splits YuE2's mixture-of-transformers into `model.diffusion_model.*` (the NAR
attention/MLP experts, `vae2llm`, `llm2vae`, time embedder) and `text_encoders.model.*` (the AR experts,
embeddings, `lm_head`). Both share the 28-layer layout with merged `qkv_proj` / `gate_up_proj`.

- **Acoustic training**: latents `x0` (64 ch, 25 fps) are noised as `x_t = (1-σ)x0 + σε`; the model's
  velocity output is regressed on `ε - x0` (the same parametrisation ComfyUI's flow sampler assumes). The
  AR prefix key/value cache is computed once per item with the CLIP model (exactly as `YuE2GenerateMusic`
  does) and fed to the NAR layers.
- **Planner training**: causal next-token cross-entropy over the AR path with chunked logits.
- **LoRA**: ComfyUI's own `LoRAAdapter.create_train` + bypass hooks (no base weights are modified). Keys
  are written in ComfyUI's generic format (`diffusion_model.…lora_up.weight`, `text_encoders.…lora_up.weight`)
  so the stock loaders map them without conversion.

`tests/verify_against_comfy.py` checks, against ComfyUI's own forward passes, that the training forward
matches inference (cosine 0.9999 for both paths on the real checkpoint) and that LoRA keys load with
zero unmatched keys; `tests/verify_lora_file.py` loads a trained file through `load_lora_for_models` and
confirms it changes the model output; `tests/verify_node_schemas.py` imports the pack the way ComfyUI does and
validates every node schema and the example workflows. `python -m pytest tests/test_units.py` runs ComfyUI-free
unit tests.

## Limitations

- **No semantic-token LoRA from arbitrary audio.** YuE2's semantic tokenizer (audio → 32 768-token codec)
  has not been released, so the semantic stage can only be trained on YuE2's own outputs (e.g. a
  best-of-N selection of songs you liked). For real recordings the acoustic LoRA is conditioned in
  text-only mode, which is a mode the base model was trained with (codec dropout) but not the mode used
  at inference; expect it to transfer sound/timbre well and rhythm/phrasing less.
- **ABC transcription is only as good as SheetSage2.** It attends to a fixed 300-second window (about
  2 GB VRAM, roughly 20 s per 4-minute song) and can miss notes or meter; check `.abc` files you care
  about, or supply your own scores as sidecars.
- Multi-chunk songs (longer than about 470 s) are trained per original chunk; songs are truncated to
  `max_seconds` at encode time if you set it.
- Training is single-process (optionally data-parallel over the GPUs in that process); ComfyUI stays busy
  while a training node runs (the queue shows progress and *Cancel* stops it cleanly).

## License

Apache-2.0 for this repository. YuE2 weights are CC BY-NC 4.0 (see the model card); LoRAs derived from
them inherit those terms.
