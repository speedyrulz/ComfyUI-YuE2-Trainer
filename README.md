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
and latents. For your own recordings the **YuE2 Semantic Tokens** node writes the `.semantic.npy` sidecar
with a community tokenizer head (see *Semantic tokens for your own recordings*); YuE2's own audio-to-semantic
tokenizer is not public (see *Limitations*).

### Semantic tokens for your own recordings (community tokenizer, experimental)

YuE2's AR stage writes *semantic tokens* (32,768 codes, 25 per second) that fix the composition, and the
acoustic stage renders them. The encoder that turns audio into those tokens has not been released, so
training the planner or the acoustic model on real songs normally has no semantic tokens to work with.
[Mothersuperior's realaudio tokenizer v4](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4)
is a community stand-in: an 8-layer transformer head on MERT-v2-FullSong layer-20 features, fitted on
YuE2's own generations. **YuE2 Semantic Tokens (community head)** runs it over a dataset and writes the
`.semantic.npy` sidecars, so afterwards `train_semantic` on the planner node and `use_semantic_tokens` on
the acoustic node work on your own recordings. Files to download yourself (both CC BY-NC 4.0):
`tokenizer_head_joint_v4.pt` into `models/audio_encoders`, and
[m-a-p/MERT-v2-FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong) (a local folder for the `mert`
input, or leave the Hugging Face id and it is fetched into the HF cache on first use, about 630 MB). The
head's NAR companion LoRA is not needed and not used: in our test it made renders less similar to the
original. Note for anyone running the head elsewhere: with transformers 5, `AutoModel.from_pretrained`
leaves MERT-v2's rotary `inv_freq` buffer uninitialised (features silently wrong or NaN, differently on
every load); the node builds the model from its config and loads the weights into it, which is what the
numbers below used.

How good is it? We measured a round trip (real 60-second excerpts -> head -> tokens -> base acoustic model,
64 steps, shift 3, cfg 2) against the originals, frame aligned:

| | chroma corr | onset corr | CLAP cos |
|---|---|---|---|
| Master of Puppets 1:00-2:00, round trip | 0.65 | 0.85 | 0.83 |
| Battery 1:10-2:10, round trip | 0.60 | 0.90 | 0.80 |
| a plain base-model generation for the same prompt | -0.02 | 0.03 | 0.55 |
| two different real songs from the album | 0.12 | 0.05 | 0.92 |
| a YuE2 generation, tokens re-predicted from its audio | 0.78 | 0.73 | 0.92 |

Exact top-1 agreement with YuE2's true codes is only about 11%, but near-miss codes render almost the
same: the round trip keeps the rhythm and most of the harmony of a real recording, far above what a
generation from the prompt alone shares with it. Expect the timbre and vocal detail to be YuE2's, not the
recording's. Treat sidecars from this head as approximate: when the official tokenizer ships, delete the
`.semantic.npy` files and re-run.

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
| **YuE2 Semantic Tokens (community head)** | Predict YuE2 semantic tokens for every recording with the Mothersuperior v4 head (MERT-v2-FullSong + small transformer) and write `<song>.semantic.npy`. Enables `train_semantic` / `use_semantic_tokens` on real songs. |
| **YuE2 Encode Dataset** | VAE-encode every item to latents in fp32 (cached under `output/yue2_trainer_cache`), optionally transcribe missing ABC with a SheetSage2 `AUDIO_ENCODER`. |
| **YuE2 Train Acoustic LoRA (MODEL)** | Flow-matching LoRA training of the acoustic model. Outputs `LORA_MODEL`, `LOSS_MAP`, steps, a text report. |
| **YuE2 Train Planner LoRA (CLIP)** | Next-token LoRA training of the language model on ABC (and semantic tokens when present). Optional regularization input and checkpoint probes (see below). |
| **YuE2 Regularization Scores** | Writes ABC scores with the base model for a few prompts (and, with `music_seconds`, the base model's music tokens for each score) and returns them as a dataset for the planner trainer's `regularization` input (saved under `output/yue2_regularization`, reused on later runs). |
| **YuE2 Save LoRA** | Writes the LoRA to `models/loras/<name>.safetensors` with training metadata; with `name` blank and `loss_map` connected it uses the trainer's `save_name`. The core `SaveLoRA` node also works (it writes to `output/`). |
| **YuE2 Load LoRA** | Applies a LoRA to MODEL and/or CLIP; either input can be left unconnected. |

The core `LossGraphNode` plots the `LOSS_MAP` output; `PreviewAny` shows the report and dataset summary.

### Typical graphs

Ready-to-load API-format graphs are in [`example_workflows/`](example_workflows) (drag onto the canvas
or use *Load*; recent frontends import API-format JSON):

- `yue2_train_acoustic_lora_api.json` – checkpoint → dataset → encode (SheetSage2 `full` transcription) →
  **YuE2 Semantic Tokens** → acoustic LoRA (`inference_like` conditioning on the songs' semantic tokens, rank 32,
  1000 steps, `keep` = best_eval, checkpoints every 250) → save + loss plot; needs the community tokenizer head
- `yue2_prepare_and_train_acoustic_api.json` – the text-only variant (no transcription or tokenizer head):
  **Prepare Dataset** generates the style/lyrics sidecars, then `compact` acoustic training
- `yue2_train_planner_lora_api.json` – same with SheetSage2 transcription (`full`, with chords) → planner LoRA,
  100 steps at `5e-5` with a checkpoint and a probe score every 10 steps and a **YuE2 Regularization Scores**
  node (base scores for the dataset's own prompts) on the `regularization` input
- `yue2_train_semantic_planner_api.json` – the planner graph with **YuE2 Semantic Tokens** between encoding and
  training and both targets on (`train_abc` + `train_semantic`, rank 32, 80 linear steps, a checkpoint and eval
  every 5 steps, an ABC + 60-s music probe every 10, regularization scores with 120 s of base-model music
  tokens at 0.2); needs the community tokenizer head (see *Semantic tokens for your own recordings*)
- `yue2_generate_with_lora_api.json` – the stock YuE2 generation graph with **YuE2 Load LoRA** between the
  checkpoint loader and the YuE2 nodes

Generation with a LoRA is the standard graph: `CheckpointLoaderSimple → YuE2 Load LoRA → YuE2GenerateABC /
YuE2GenerateMusic → ModelSamplingAuraFlow(shift 3) → KSampler(32 steps, cfg 1, euler, simple) → VAEDecodeAudio`.
The stock template uses 64 steps and cfg 2; in a same-seed comparison the step count, solver and shift changed the
render by less than 0.1% while cfg 2 (guidance against a zeroed conditioning the model never saw in training)
moved it *away* from the training album by 0.07 CLAP. The reference implementation runs the acoustic stage
without guidance, so the example uses cfg 1, which also halves sampling time.
An acoustic LoRA only affects the KSampler stage; a planner LoRA only affects the two YuE2 generate nodes.

### Settings that matter

**Acoustic LoRA**

- `segment_seconds` (30): random crop per step. The crop keeps its absolute position inside the song
  (RoPE positions and the latent position table), so training crops look like inference. `0` trains whole
  songs (more VRAM).
- `conditioning` (compact): how the text-only training context is laid out. `compact` uses a `cot=off`,
  style-only prefix with the NAR tokens directly behind it and latent positions restarting at every segment
  (the regime of the standalone trainers; needs no ABC transcription or lyrics, 3x faster). `inference_like`
  uses the style + lyrics + ABC prefix, splits long songs into the same chunks generation would use, and
  places the NAR tokens where they sit at generation (after that chunk's codec tokens). In the A/B test
  below both produced the same album similarity; keep `compact` unless you generate with a hand-written ABC
  and want the LoRA to see scores during training.
- `prefix_mode` (full, `inference_like` only): planning instruction used in the conditioning prefix; match
  the mode you generate with. Items without an ABC score always use `off`. `auto` picks `full` when the ABC
  has chord symbols, else `melody`.
- `use_semantic_tokens` (off): when on, items with semantic tokens (YuE2 output folders) are conditioned exactly
  like inference (prefix + codec tokens). Otherwise items use the model's codec-dropout ("text-only") conditioning: only the
  text/ABC prefix is visible, and the NAR tokens keep the positions they would have after the codec tokens.
- `train_acoustic_head` (off): also adapt `vae2llm` / `llm2vae` / time embedder projections.
- `caption_dropout` (0.1): fraction of steps trained on YuE2's own unconditional prefix (the instruction with
  no style or lyrics, the same prefix its CFG negative branch uses). Keeps the base behaviour reachable at
  generation time and regularises small datasets; 0 disables it.
- `timestep_sampling` / `shift`: sigma distribution (uniform by default, matching the reference solver).
- `rank`/`alpha` (16/16 → scale 1), `learning_rate` (1e-4), `lr_schedule` = `cosine` (decay to 10%), `constant`,
  or `linear`; `warmup_steps` ramps up first in every mode. Same options on the planner node. Note that
  ComfyUI's LoRA adapter initialises the fixed matrix about 8x larger than PEFT/kohya trainers do, so
  `1e-4` here moves the weights roughly as much as `8e-4` would in a PEFT-style trainer (about 6% of the
  weight norm after 1000 steps at rank 32); do not copy a higher learning rate from other trainers.
- `save_every` writes `models/loras/<save_name>_<steps>.safetensors` checkpoints, each with a `<name>.resume`
  file next to it (optimizer moments, every replica's random state, step count, loss history). **YuE2 Save
  LoRA** writes the same file next to the final LoRA. `existing_lora` + `resume_state` (on) continues that
  exact run: `steps` is then the run's total length (a checkpoint from step 50 with `steps` 100 trains 50
  more on the same cosine schedule), the loss / eval curves carry on, and a state whose trainer, rank,
  targets or optimizer differ is ignored with a warning. `resume_state` off (or no `.resume` file) starts a
  new run from the LoRA weights with a fresh optimizer and schedule, as before.
- `eval_every` (50) / `eval_samples` (8) / `eval_holdout` (1): score a fixed evaluation set before step 1 and
  every N steps. By default one song is held out of training and the set is drawn from it (fixed crops; for
  the acoustic trainer also fixed sigmas, stratified over the sigma distribution, and fixed noise), so the
  number only moves when the LoRA does and it measures how the LoRA handles a song it has not seen. With
  `eval_holdout` 0 every song trains and the set is drawn from the training songs instead, which measures
  fit only. Never more than `items - 3` songs are held out. See *Watching a run*.
- `keep` (final): which weights the node outputs. `best_eval` returns the checkpoint with the lowest
  evaluation loss of the run (its resume state follows), so the run can be scheduled long and still hand
  back the checkpoint at the minimum. The final step stays the default because, for the planner, the
  checkpoint that sounds closest to the album is often a little past the held-out minimum; `save_every`
  checkpoints let you compare both.

**Planner LoRA**

- `train_abc` (on): `style + lyrics -> ABC + </abc>`; the loss covers only the ABC tokens.
- `train_semantic` (off): `style + lyrics + ABC -> semantic tokens` for items that carry them.
- `abc_mode` (full): the planning instruction the LoRA is trained under; match the mode you generate with. `auto` picks `full` when the score has chords, else `melody`.
- `max_tokens` (4096): longest trained span per step. A score longer than this is trained through one of
  three windows per step: its head, its tail (ending with the closing token) or a random middle window, and
  windows that do not start at the beginning keep their first 256 tokens as unsupervised context. So the
  model always keeps learning how a score opens and how it ends. 8192 lets most whole songs train in one
  piece on a 16 GB card, but whole songs also teach the album's song *lengths*: a LoRA trained on 6-9-minute
  songs writes 6,000-9,000-token scores, which hits `YuE2GenerateABC`'s `max_abc_tokens` (raise it to
  12,000+ and `max_duration` to match, or keep `max_tokens` at 4096 so the endings the model sees sit near
  the 4,000-token mark and it keeps writing normal-length songs for your lyrics). (Before September 2026 the crop was a uniformly random window, which for 4-9 minute
  songs almost never contained the closing token; planner LoRAs from that version stop ending their scores
  after a few dozen steps and should be retrained.)
- `regularization` (optional input) + `regularization_fraction` (0.5): scores the *base* model wrote, mixed
  into training. This is prior preservation, the DreamBooth "class images" idea: with a handful of album
  scores the planner starts imitating them so hard after a few dozen steps that it forgets how a score ends
  and runs to the token cap. Drawing half the steps from the base model's own scores keeps it anchored to
  well-formed, normal-length writing while it picks up the album's melodic and harmonic habits. Make the
  scores with **YuE2 Regularization Scores** (base CLIP, no LoRA): give it a few style prompts, one per line,
  or connect the training dataset to use your songs' own prompts and lyrics; 2 scores per prompt and 4-8
  prompts is plenty. Scores that do not end within `max_abc_tokens` are discarded. A folder of YuE2 output
  directories (`request.json` + `score.abc`) or `.abc` sidecars works too. When a regularization set is
  connected, every evaluation also reports the `regularizer loss` on fixed crops of those scores: it starts
  at the base model's own value and should stay close to it; a steady climb means the LoRA is drifting away
  from base-model writing faster than the regularization can hold it (lower the learning rate or raise the
  fraction). With `train_semantic` on, set the node's `music_seconds` (120 is plenty): the base model then
  also writes its own music-token stream for every score, saved as `semantic.npy` next to it, and the
  regularization covers the music-token target too. A stream cut off at the budget is trained without a
  closing token, so it never teaches a false ending.
- `probe_every` (0 = off; set it to `save_every`): every N steps, and before step 1, the trainer writes one
  whole ABC score with the current LoRA through YuE2's own sampler (same defaults as `YuE2GenerateABC`,
  fixed `probe_seed`) and reports its length and whether it produced the closing token. A probe that uses
  its whole `probe_max_tokens` budget is the over-training signal the held-out loss misses. `probe_style` /
  `probe_lyrics` default to the first training song's. The scores land in
  `output/yue2_probes/<save_name>/step_000025.abc` (with a `.json` of the numbers), so you can drop any of
  them into `YuE2GenerateMusic`'s `abc` input and listen to what each checkpoint writes. Each probe costs as
  much as one `YuE2GenerateABC` run (a minute or two for a 3,000-token score, more for one that hits the
  cap).
- `probe_music_seconds` (0 = off): with probes on, each probe also writes the music-token stream for the
  probe prompt with the current LoRA, conditioned on the score it just wrote (or a fixed `probe_abc`), up to
  that many seconds, and reports its length, whether it ended and the share of distinct tokens (a collapsing
  LoRA repeats itself). The tokens land next to the score as `step_000025.semantic.npy`. This is the
  over-training signal for `train_semantic`: compare each checkpoint with step 0, and treat a stream that
  ends far earlier than the base model's, or whose distinct share drops, as over-trained on the music-token
  target. About a minute per 60 s.
- The planner learns fast: every step supervises thousands of score tokens, so 25-100 steps at `5e-5`
  (the node defaults are 100 steps, `5e-5`, cosine) already reshape the writing. Watch the held-out eval
  line and the probes, and keep `save_every` small (5-25) so you can pick the best checkpoint. Treat the
  held-out minimum as advisory: on a handful of songs the checkpoint that sounds closest to the album is
  often a little past it. A planner LoRA that makes `YuE2GenerateABC` run to `max_abc_tokens` instead of
  finishing is either over-trained (use an earlier checkpoint, or retrain with a regularization set) or has
  learned album-length scores (see `max_tokens` above).

Both trainers use gradient checkpointing, bf16 autocast, fp32 LoRA weights, grad clipping, and run
one item per micro-step (`batch_size × grad_accumulation` items per optimizer step).

### What an acoustic LoRA can and cannot change

Generation in ComfyUI is deterministic (same seed, same graph → bit-identical audio), so any difference you
hear with the LoRA loaded is real. But the **composition is decided before the acoustic model runs**: the
frozen AR stage writes the ABC score (or, with `cot=off`, goes straight to semantic tokens), and those
tokens fix melody, chords, rhythm, arrangement and the vocal line. The acoustic LoRA only changes how that
plan is rendered: timbre, guitar and drum sounds, vocal character, mix and production. With the same seed
and prompt, every acoustic LoRA we measured (ours and the Starnodes trainer's, strength 1.0) kept a waveform
correlation of 0.88-0.91 with the base render — the same song, re-recorded. That is the expected size of the
effect; to change what YuE2 *writes*, train the planner LoRA on the same dataset and load both.

Measured on 8 Master of Puppets songs (rank 32, 1000 steps, lr 1e-4, style prompt with the artist first,
5 seeds, CLAP cosine to the album's audio embedding, higher = closer; seed-to-seed spread ±0.02):

| LoRA (strength 1.0) | CLAP → album | Δ vs base, paired | corr with base render |
|---|---|---|---|
| none (base model) | 0.723 | – | 1.000 |
| this trainer, `compact` (5.5 min) | 0.734 | +0.014 (3/4 seeds) | 0.879 |
| this trainer, `inference_like` (17.5 min) | 0.733 | +0.016 (4/4 seeds) | 0.885 |
| Starnodes ComfyUI-YuE2-Trainer (28 min, 24 GB) | 0.713 | -0.004 (1/4 seeds) | 0.911 |

The pull toward the album is small but consistent, and slightly stronger at strength 1.5. Practical advice:

- Put the artist (or your trigger word) first in the style prompt at generation, exactly as in the training
  `.style.txt` files, and describe the sound rather than the genre alone.
- Rank 32, 1000-3000 steps and strength 1.0-1.5 are good starting points; a LoRA that changes the sound too
  much (muffled or noisy) is over-trained — lower the steps or the strength.
- `full` planning mode is fine for LoRA generation. `cot=off` (no ABC) gave the same relative gain but much
  lower album similarity overall for our prompt.
- Songs longer than about 4 minutes used to be trained at RoPE positions far beyond anything generation
  uses (a bug fixed in September 2026); retrain LoRAs made before that fix.

### Watching a run

Every step (or every `log_every` steps) the console shows

```
YuE2 acoustic step 120/300  loss 0.8412  avg20 0.8630  lr 7.65e-05 grad 0.412  elapsed 4:10  eta 6:15
```

The per-step loss of the acoustic trainer is a noisy estimate near an irreducible floor (a perfect model
still cannot predict the random noise it was given), so it goes flat after a few dozen steps while the LoRA
keeps changing. The fixed-set evaluation line is the one to watch:

```
YuE2 acoustic eval step 200/1000  fixed-set loss 0.9127  (start 1.0418, best 0.9127, change -12.4%)
```

Because the crops, sigmas and noise never change, differences of a few thousandths are real. With the
default `eval_holdout` of 1 the line says `held-out loss` and is scored on a song the LoRA never trains on,
which is the signal you want: still falling = still learning something that transfers; flat for several
evaluations = done; rising after a minimum = over-training, so use the checkpoint from the best step
(`save_every` 10-25 for the planner, 250 for the acoustic trainer). A loss on *training* crops
(`eval_holdout` 0, label `fixed-set`) keeps falling while the planner memorises the scores and stops
generating properly, so it cannot tell you when to stop; use it only to confirm that training is moving at
all. For the planner the held-out cross-entropy typically bottoms out after a few dozen steps on a small
album; the acoustic held-out loss moves by hundredths over a thousand steps.

For the planner, add probes (`probe_every`): the console then also shows

```
YuE2 planner probe step 25/100  3103 ABC tokens in 1:52, ended normally  (step 0: 2871 tokens)  -> .../step_000025.abc
YuE2 planner music probe step 25/100  1500 music tokens (60.0 s, 61% distinct) in 1:05, ran the whole 60-s budget  (step 0: 1500 tokens, 63% distinct)
YuE2 planner probe step 50/100  8192 ABC tokens in 4:40, HIT THE TOKEN BUDGET without ending (over-trained or album-length scores)
```

and the run summary lists every probe. The step whose probe last ended normally is the latest checkpoint
worth keeping; with a regularization set connected the `regularizer loss` line should stay near its starting
value.

Turn on `tensorboard` to also log `loss/step`, `loss/avg20`, `loss/eval_heldout` (or `loss/eval_fixed`),
`loss/eval_regularizer`, `probe/abc_tokens`, `probe/ended`, `probe/music_tokens`, `probe/music_ended`,
`probe/music_distinct`, `lr` and `grad_norm` per step, plus the run
configuration and final result as text. Runs land in `ComfyUI/output/yue2_tensorboard/<save_name>_<timestamp>`
(`tensorboard_dir` changes the parent folder); view them with

```bash
tensorboard --logdir ComfyUI/output/yue2_tensorboard
```

The CLI equivalent is `--tensorboard DIR` (plus `--log-every N`, `--eval-every N`, `--eval-samples N`,
`--run-name`); the CLI also writes the step and evaluation losses to `<out>.loss.json`.

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

`--out` names go to `models/loras`; `--save-every N` writes checkpoints (each with a `.resume` state file);
`--existing-lora` continues a run, restoring its `.resume` state unless `--no-resume`; `--dry-run` prepares
everything and trains 0 steps. `--devices cuda:1` picks a GPU, `--devices all` uses every GPU data-parallel
(see *Choosing GPUs*). Planner extras: `--regularization DIR` (+ `--regularization-fraction`) mixes in a
folder of base-model scores, `--probe-every N` (+ `--probe-style`, `--probe-lyrics` or `@file`,
`--probe-max-tokens`, `--probe-music-seconds`, `--probe-abc`, `--probe-dir`) writes probe scores (and music
tokens) to `<out>_probes/`.

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

- **Semantic tokens for real audio are approximate.** YuE2's semantic tokenizer (audio → 32 768-token codec)
  has not been released. Exact tokens exist only for YuE2's own outputs; for your recordings the
  **YuE2 Semantic Tokens** node predicts them with a community head (see *Semantic tokens for your own
  recordings*), which keeps rhythm and most harmony but not every code. Without those sidecars the
  acoustic LoRA is conditioned in text-only mode, which the base model was trained with (codec dropout) but
  is not the mode used at inference; expect it to transfer sound/timbre and leave composition to the
  (frozen or planner-LoRA) AR stage — see *What an acoustic LoRA can and cannot change* above.
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
