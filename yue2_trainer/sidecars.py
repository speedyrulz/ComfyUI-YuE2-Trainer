"""Automatic ``.style.txt`` / ``.lyrics.txt`` sidecars for a dataset folder.

Lyrics: LRCLIB lookup (artist/title from tags or the file name) with Whisper transcription
as fallback; section tags ([Verse]/[Chorus]) from a repetition heuristic or from Claude.
Style: CLAP zero-shot tags (genre, mood, instruments, vocals) + librosa tempo/key + the
language Whisper detected, assembled into a YuE2-style comma-separated prompt.

This module only needs torch/transformers/librosa; it does not import ComfyUI, so the
same code runs from the node and from ``prepare_dataset.py``.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from .audio import load_audio, to_stereo_48k
from .constants import AUDIO_EXTENSIONS

LOG = logging.getLogger("yue2_trainer.sidecars")

DEFAULT_WHISPER = "openai/whisper-large-v3-turbo"
DEFAULT_CLAP = "laion/larger_clap_music_and_speech"
DEFAULT_CLAUDE = "claude-opus-5"
WHISPER_CHOICES = [DEFAULT_WHISPER, "openai/whisper-large-v3", "openai/whisper-medium", "openai/whisper-small"]

GENRES = ["pop", "rock", "hip hop", "rap", "r&b", "soul", "funk", "jazz", "blues", "country", "folk", "metal",
          "punk", "indie rock", "electronic", "house", "techno", "trance", "drum and bass", "dubstep", "ambient",
          "lo-fi", "classical", "orchestral", "cinematic", "latin", "reggae", "reggaeton", "k-pop", "j-pop",
          "city pop", "synthwave", "disco", "gospel", "edm", "trap", "afrobeats", "bossa nova",
          "singer-songwriter", "acoustic"]
MOODS = ["upbeat", "energetic", "melancholic", "romantic", "dark", "dreamy", "epic", "chill", "aggressive",
         "uplifting", "nostalgic", "sad", "joyful", "mysterious", "groovy", "intimate", "anthemic", "danceable"]
INSTRUMENTS = ["acoustic guitar", "electric guitar", "distorted guitar", "piano", "synthesizer", "strings",
               "brass", "saxophone", "violin", "drums", "808 bass", "bass guitar", "organ", "flute", "trumpet",
               "drum machine", "harp", "choir", "ukulele", "cello"]
VOCALS = ["female vocals", "male vocals", "instrumental"]
GENRE_TEMPLATES = ["This is {} music.", "This is a {} song.", "{}", "The genre of this music is {}."]
MOOD_TEMPLATES = ["This music sounds {}.", "{} music", "A {} song."]
INSTRUMENT_TEMPLATES = ["This music features {}.", "{}", "A song with {}."]
VOCAL_TEMPLATES = ["{}", "A song with {}.", "This is {}."]
LANGUAGES = {"en": "English", "zh": "Mandarin", "ja": "Japanese", "ko": "Korean", "es": "Spanish", "fr": "French",
             "de": "German", "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ar": "Arabic", "hi": "Hindi",
             "nl": "Dutch", "sv": "Swedish", "pl": "Polish", "tr": "Turkish", "vi": "Vietnamese", "th": "Thai",
             "id": "Indonesian", "yue": "Cantonese", "tl": "Tagalog", "uk": "Ukrainian", "cs": "Czech"}
KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass
class SidecarConfig:
    lyrics_source: str = "lrclib+whisper"   # lrclib+whisper | lrclib | whisper | none
    whisper_model: str = DEFAULT_WHISPER
    style_source: str = "clap"              # clap | none
    clap_model: str = DEFAULT_CLAP
    section_tags: str = "heuristic"         # heuristic | claude | none
    claude_model: str = DEFAULT_CLAUDE
    default_style: str = ""                 # used when style_source is none
    overwrite: bool = False
    device: str = "auto"
    excerpt_seconds: float = 10.0
    excerpts: int = 6
    temperature: float = 50.0
    min_lyrics_chars: int = 40


@dataclass
class ItemReport:
    file: str
    style: Optional[str] = None
    style_source: Optional[str] = None
    lyrics_source: Optional[str] = None
    lyrics_chars: int = 0
    language: Optional[str] = None
    tempo: Optional[float] = None
    key: Optional[str] = None
    tags: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    seconds: float = 0.0


# ── metadata ──────────────────────────────────────────────────────────────────

def file_metadata(path: Path) -> dict:
    """artist/title/duration from tags (mutagen) or an 'Artist - Title' file name."""
    meta = {"artist": None, "title": None, "duration": None}
    try:
        import mutagen
        tags = mutagen.File(str(path), easy=True)
        if tags is not None:
            meta["artist"] = (tags.get("artist") or [None])[0]
            meta["title"] = (tags.get("title") or [None])[0]
            meta["duration"] = float(tags.info.length) if getattr(tags, "info", None) else None
    except Exception as exc:  # noqa: BLE001
        LOG.debug("mutagen failed on %s: %s", path, exc)
    if not meta["title"]:
        artist, title = parse_stem(path.stem)
        meta["artist"] = meta["artist"] or artist
        meta["title"] = title
    return meta


def parse_stem(stem: str) -> tuple[Optional[str], str]:
    stem = re.sub(r"^\d{1,3}[\s._-]+", "", stem)                      # track numbers
    stem = re.sub(r"\[(official|lyrics?|audio|video|hd|4k)[^\]]*\]", "", stem, flags=re.I)
    stem = re.sub(r"\((official|lyrics?|audio|video|hd|4k)[^)]*\)", "", stem, flags=re.I)
    parts = re.split(r"\s+-\s+|\s+–\s+|_-_", stem, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip() or None, parts[1].strip()
    return None, stem.replace("_", " ").strip()


# ── lyrics: LRCLIB ─────────────────────────────────────────────────────────────

def fetch_lrclib(artist: Optional[str], title: str, duration: Optional[float], timeout: float = 15.0) -> Optional[str]:
    import requests
    headers = {"User-Agent": "ComfyUI-YuE2-Trainer/0.1 (https://github.com/speedyrulz/ComfyUI-YuE2-Trainer)"}
    try:
        if artist and duration:
            r = requests.get("https://lrclib.net/api/get", timeout=timeout, headers=headers,
                             params={"artist_name": artist, "track_name": title, "duration": int(round(duration))})
            if r.status_code == 200:
                data = r.json()
                if data.get("plainLyrics"):
                    return data["plainLyrics"].strip()
        if not artist:
            # title-only search is too ambiguous ("Passion" matches many songs); leave it to Whisper
            return None
        params = {"track_name": title, "artist_name": artist}
        r = requests.get("https://lrclib.net/api/search", timeout=timeout, headers=headers, params=params)
        if r.status_code != 200:
            return None
        candidates = [c for c in r.json() if c.get("plainLyrics")]
        if not candidates:
            return None
        if duration:
            candidates.sort(key=lambda c: abs(float(c.get("duration") or 0) - duration))
            if abs(float(candidates[0].get("duration") or 0) - duration) > 15:
                return None
        return candidates[0]["plainLyrics"].strip()
    except Exception as exc:  # noqa: BLE001
        LOG.warning("LRCLIB lookup failed for %s - %s: %s", artist, title, exc)
        return None


# ── lyrics: Whisper ────────────────────────────────────────────────────────────

class WhisperTranscriber:
    """Whisper through the model API directly (no ASR pipeline, no torchcodec dependency)."""

    CHUNK = 30 * 16000

    def __init__(self, model_name: str = DEFAULT_WHISPER, device: str = "cuda"):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        LOG.info("loading %s", model_name)
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=self.dtype)
        self.model.to(self.device).eval()

    def close(self):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _features(self, pieces: list[np.ndarray]) -> torch.Tensor:
        feats = self.processor(pieces, sampling_rate=16000, return_tensors="pt").input_features
        return feats.to(self.device, self.dtype)

    @torch.no_grad()
    def detect_language(self, wave16k: np.ndarray) -> Optional[str]:
        try:
            # use the loudest 30 s window so instrumental intros do not confuse detection
            best = max(range(0, max(1, len(wave16k) - self.CHUNK + 1), self.CHUNK // 2),
                       key=lambda i: float(np.abs(wave16k[i:i + self.CHUNK]).mean()))
            ids = self.model.detect_language(self._features([wave16k[best:best + self.CHUNK]]))
            token = self.processor.tokenizer.decode(ids.reshape(-1)[:1])
            code = token.strip("<|>")
            return code if code and code != "nospeech" else None
        except Exception as exc:  # noqa: BLE001
            LOG.debug("language detection failed: %s", exc)
            return None

    @staticmethod
    def _segments(text: str, offset: float, chunk_seconds: float) -> list[tuple[float, float, str]]:
        parts = re.split(r"<\|(\d+\.\d+)\|>", text)
        out = []
        for i in range(1, len(parts) - 1, 2):
            start, body = float(parts[i]), parts[i + 1].strip()
            if not body:
                continue
            end = float(parts[i + 2]) if i + 2 < len(parts) else chunk_seconds
            out.append((offset + start, offset + end, body))
        if not out and text.strip():
            plain = re.sub(r"<\|[^|]*\|>", "", text).strip()
            if plain:
                out.append((offset, offset + chunk_seconds, plain))
        return out

    @torch.no_grad()
    def transcribe(self, wave16k: np.ndarray, language: Optional[str] = None,
                   batch_size: int = 8) -> list[tuple[float, float, str]]:
        pieces = [wave16k[i:i + self.CHUNK] for i in range(0, len(wave16k), self.CHUNK)]
        pieces = [pc for pc in pieces if len(pc) > 16000]
        segments = []
        for b in range(0, len(pieces), batch_size):
            batch = pieces[b:b + batch_size]
            kwargs = {"task": "transcribe", "return_timestamps": True, "max_new_tokens": 440}
            if language:
                kwargs["language"] = language
            ids = self.model.generate(self._features(batch), **kwargs)
            texts = [self.processor.tokenizer.decode(seq.tolist(), skip_special_tokens=True, decode_with_timestamps=True)
                     for seq in ids]
            for j, text in enumerate(texts):
                offset = (b + j) * 30.0
                segments.extend(self._segments(text, offset, len(batch[j]) / 16000))
        return segments


def _norm(line: str) -> str:
    return re.sub(r"[^\w\s]", "", line.lower()).strip()


def _similar(a: str, b: str) -> bool:
    if a == b:
        return True
    if min(len(a), len(b)) < 10 or abs(len(a) - len(b)) > max(3, len(a) // 3):
        return False  # short lines must match exactly
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.85


def _block_overlap(lines: list[str], other: list[str]) -> float:
    """Fraction of ``lines`` that (fuzzily) occur in ``other`` - tolerant to ASR spelling variance."""
    if not lines or not other:
        return 0.0
    return sum(any(_similar(l, o) for o in other) for l in lines) / len(lines)


HALLUCINATION_RE = re.compile(
    r"(subtitles? by|subscribe|thanks? for watching|we'll be right back|amara\.org|字幕|作曲|作词|編曲|翻译|"
    r"転載|ご視聴ありがとう|チャンネル登録|시청해 주셔서|구독)", re.I)


def clean_segments(chunks: list[tuple[float, float, str]], max_consecutive: int = 2,
                   max_total: int = 6) -> tuple[list[tuple[float, float, str]], int]:
    """Drop Whisper hallucinations: credit/subtitle phrases, long runs of one line, lines repeated everywhere."""
    counts: dict[str, int] = {}
    for _, _, text in chunks:
        counts[_norm(text)] = counts.get(_norm(text), 0) + 1
    out, dropped, previous, run = [], 0, None, 0
    for start, end, text in chunks:
        key = _norm(text)
        if not key or HALLUCINATION_RE.search(text) or counts[key] >= max_total:
            dropped += 1
            continue
        run = run + 1 if key == previous else 1
        previous = key
        if run > max_consecutive:
            dropped += 1
            continue
        out.append((start, end, text))
    return out, dropped


def heuristic_sections(chunks: list[tuple[float, float, str]], gap_seconds: float = 2.5,
                       max_lines: int = 6) -> str:
    """Group transcribed lines into blocks; blocks that repeat elsewhere become [Chorus]."""
    if not chunks:
        return ""
    blocks, current, last_end = [], [], None
    for start, end, text in chunks:
        if current and ((last_end is not None and start - last_end > gap_seconds) or len(current) >= max_lines):
            blocks.append(current)
            current = []
        current.append(text)
        last_end = end
    if current:
        blocks.append(current)
    normed = [[_norm(l) for l in b if _norm(l)] for b in blocks]
    kinds = ["Chorus" if any(i != j and _block_overlap(lines, other) >= 0.6 for j, other in enumerate(normed))
             else "Verse" for i, lines in enumerate(normed)]
    out = []
    if chunks[0][0] > 8.0:
        out.append("[Intro]\n")
    previous = None
    for kind, block in zip(kinds, blocks):
        if kind != previous or kind == "Verse":
            out.append(f"[{kind}]")
        out.extend(block)
        out.append("")
        previous = kind
    return "\n".join(out).strip() + "\n"


def plain_to_sections(text: str) -> str:
    """LRCLIB plain lyrics: keep blank-line structure, tag repeated stanzas as chorus."""
    stanzas = [s.strip().splitlines() for s in re.split(r"\n\s*\n", text.strip()) if s.strip()]
    if not stanzas:
        return ""
    normed = [[_norm(l) for l in s if _norm(l)] for s in stanzas]
    out = []
    for i, stanza in enumerate(stanzas):
        repeated = any(i != j and _block_overlap(normed[i], normed[j]) >= 0.6 for j in range(len(stanzas)))
        out.append("[Chorus]" if repeated else "[Verse]")
        out.extend(l.strip() for l in stanza)
        out.append("")
    return "\n".join(out).strip() + "\n"


SECTION_SYSTEM = """You format song lyrics for a music generation model.
Output ONLY the lyrics with section tags such as [Intro], [Verse], [Pre-Chorus], [Chorus], [Bridge], [Outro],
each tag on its own line before its lines, one lyric line per line, a blank line between sections.
Repeated choruses must be written out in full each time they occur.
Fix obvious speech-recognition errors only when the correction is certain; never invent lines, never
add commentary, translations, chords or timestamps. If the text is clearly not lyrics, output an empty response."""


def claude_sections(raw_text: str, model: str = DEFAULT_CLAUDE, style_hint: str = "") -> Optional[str]:
    """Ask Claude to add section tags and fix ASR mistakes. Returns None on any failure."""
    try:
        import anthropic
    except ImportError:
        LOG.warning("anthropic SDK not installed; falling back to heuristic section tags")
        return None
    try:
        client = anthropic.Anthropic()
        hint = f"Style of the song: {style_hint}\n\n" if style_hint else ""
        response = client.beta.messages.create(
            model=model,
            max_tokens=8000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SECTION_SYSTEM,
            messages=[{"role": "user", "content": f"{hint}Raw lyrics:\n\n{raw_text}"}],
        )
        if response.stop_reason == "refusal":
            return None
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        return text + "\n" if text else None
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Claude section tagging failed (%s); using heuristic tags", exc)
        return None


# ── style: CLAP + tempo/key ────────────────────────────────────────────────────

class ClapTagger:
    def __init__(self, model_name: str = DEFAULT_CLAP, device: str = "cuda"):
        from transformers import ClapModel, ClapProcessor
        self.device = torch.device(device)
        LOG.info("loading %s", model_name)
        self.model = ClapModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = ClapProcessor.from_pretrained(model_name)
        self._text_cache: dict[tuple, torch.Tensor] = {}

    def close(self):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _projected(output) -> torch.Tensor:
        # transformers >= 5 returns BaseModelOutputWithPooling (pooler_output = projected embedding)
        return output if torch.is_tensor(output) else output.pooler_output

    @torch.no_grad()
    def text_embeds(self, prompts: list[str]) -> torch.Tensor:
        key = tuple(prompts)
        if key not in self._text_cache:
            inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
            emb = self._projected(self.model.get_text_features(**inputs))
            self._text_cache[key] = torch.nn.functional.normalize(emb, dim=-1)
        return self._text_cache[key]

    @torch.no_grad()
    def audio_embed(self, excerpts48k: list[np.ndarray]) -> torch.Tensor:
        inputs = self.processor(audio=excerpts48k, sampling_rate=48000, return_tensors="pt").to(self.device)
        emb = self._projected(self.model.get_audio_features(**inputs))
        return torch.nn.functional.normalize(emb.mean(0, keepdim=True), dim=-1)

    def rank(self, audio_emb: torch.Tensor, tags: list[str], templates, temperature: float = 50.0) -> list[tuple[str, float]]:
        if isinstance(templates, str):
            templates = [templates]
        text = torch.stack([self.text_embeds([t.format(tag) for tag in tags]) for t in templates]).mean(0)
        text = torch.nn.functional.normalize(text, dim=-1)
        probs = (temperature * audio_emb @ text.T).softmax(-1)[0]
        order = probs.argsort(descending=True)
        return [(tags[i], float(probs[i])) for i in order]


def tempo_and_key(wave48k_mono: np.ndarray) -> tuple[Optional[float], Optional[str]]:
    try:
        import librosa
        y = librosa.resample(wave48k_mono.astype(np.float32), orig_sr=48000, target_sr=22050)
        y = y[: 22050 * 120]
        tempo, _ = librosa.beat.beat_track(y=y, sr=22050)
        tempo = float(np.atleast_1d(tempo)[0])
        chroma = librosa.feature.chroma_cqt(y=y, sr=22050).mean(axis=1)
        best, best_score = None, -2.0
        for i in range(12):
            for name, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
                score = float(np.corrcoef(np.roll(profile, i), chroma)[0, 1])
                if score > best_score:
                    best, best_score = f"{KEYS[i]} {name}", score
        return round(tempo), best
    except Exception as exc:  # noqa: BLE001
        LOG.debug("tempo/key failed: %s", exc)
        return None, None


def excerpts(wave48k_mono: np.ndarray, seconds: float, count: int) -> list[np.ndarray]:
    n = int(seconds * 48000)
    total = len(wave48k_mono)
    if total <= n:
        return [wave48k_mono]
    centres = [total * (i + 1) / (count + 1) for i in range(count)]
    return [wave48k_mono[max(0, int(c - n / 2)): max(0, int(c - n / 2)) + n] for c in centres]


def assemble_style(language: Optional[str], tags: dict, tempo: Optional[float]) -> str:
    parts = []
    if language:
        parts.append(LANGUAGES.get(language, language))
    parts += tags.get("genre", [])
    parts += tags.get("mood", [])
    parts += tags.get("vocal", [])
    parts += tags.get("instruments", [])
    if tempo:
        parts.append(f"{int(tempo)} BPM")
    seen, out = set(), []
    for p in parts:
        if p and p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return ", ".join(out)


def pick_tags(tagger: ClapTagger, audio_emb: torch.Tensor, temperature: float = 50.0) -> dict:
    genre = tagger.rank(audio_emb, GENRES, GENRE_TEMPLATES, temperature)
    mood = tagger.rank(audio_emb, MOODS, MOOD_TEMPLATES, temperature)
    instr = tagger.rank(audio_emb, INSTRUMENTS, INSTRUMENT_TEMPLATES, temperature)
    vocal = tagger.rank(audio_emb, VOCALS, VOCAL_TEMPLATES, temperature * 0.6)
    return {
        "genre": [t for t, p in genre[:2] if p >= 0.1] or [genre[0][0]],
        "mood": [t for t, p in mood[:2] if p >= 0.1] or [mood[0][0]],
        "instruments": [t for t, p in instr[:3] if p >= 0.1],
        "vocal": [vocal[0][0]] if vocal[0][1] >= 0.5 else [],
        "scores": {"genre": genre[:5], "mood": mood[:4], "instruments": instr[:6], "vocal": vocal[:3]},
    }


# ── driver ─────────────────────────────────────────────────────────────────────

def _resolve_device(spec: str) -> str:
    if spec in ("", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return spec


def _resample_mono(wave48k: torch.Tensor, target: int) -> np.ndarray:
    import torchaudio
    mono = wave48k.mean(0, keepdim=True)
    return torchaudio.functional.resample(mono, 48000, target)[0].numpy()


def list_audio(folder: Path, recursive: bool = True) -> list[Path]:
    walker = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in walker if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
                  and not (p.name == "audio.flac" and (p.parent / "request.json").is_file()))


def prepare_folder(folder, cfg: SidecarConfig, recursive: bool = True,
                   progress: Optional[Callable[[int, int, str], None]] = None,
                   interrupt: Optional[Callable[[], None]] = None) -> list[ItemReport]:
    folder = Path(folder)
    files = list_audio(folder, recursive)
    if not files:
        raise ValueError(f"No audio files in {folder}")
    device = _resolve_device(cfg.device)
    need_whisper = cfg.lyrics_source in ("lrclib+whisper", "whisper") or cfg.style_source == "clap"
    whisper = tagger = None
    reports = []
    try:
        for index, path in enumerate(files):
            if interrupt is not None:
                interrupt()
            report = ItemReport(file=str(path))
            style_path = path.with_name(path.stem + ".style.txt")
            lyrics_path = path.with_name(path.stem + ".lyrics.txt")
            want_style = cfg.style_source != "none" and (cfg.overwrite or not style_path.exists())
            want_lyrics = cfg.lyrics_source != "none" and (cfg.overwrite or not lyrics_path.exists())
            if not want_style and not want_lyrics:
                report.notes.append("sidecars exist (skipped)")
                reports.append(report)
                if progress:
                    progress(index + 1, len(files), path.name)
                continue
            wave, sr = load_audio(path)
            wave = to_stereo_48k(wave, sr)
            report.seconds = wave.shape[-1] / 48000
            meta = file_metadata(path)
            mono48 = wave.mean(0).numpy()
            wave16 = None
            language = None

            # ---- lyrics
            lyrics = None
            if want_lyrics and cfg.lyrics_source in ("lrclib", "lrclib+whisper"):
                found = fetch_lrclib(meta["artist"], meta["title"], meta["duration"] or report.seconds)
                if found and len(found) >= cfg.min_lyrics_chars:
                    lyrics = plain_to_sections(found)
                    report.lyrics_source = "lrclib"
            if (want_lyrics and lyrics is None and cfg.lyrics_source in ("whisper", "lrclib+whisper")) or \
                    (want_style and need_whisper and cfg.lyrics_source != "none"):
                if whisper is None:
                    whisper = WhisperTranscriber(cfg.whisper_model, device)
                wave16 = _resample_mono(wave, 16000)
                language = whisper.detect_language(wave16)
                if want_lyrics and lyrics is None:
                    chunks, dropped = clean_segments(whisper.transcribe(wave16, language))
                    if dropped:
                        report.notes.append(f"{dropped} suspected hallucinated lines removed")
                    raw = "\n".join(t for _, _, t in chunks)
                    if len(raw) < cfg.min_lyrics_chars:
                        report.notes.append("little or no speech found; treated as instrumental")
                        lyrics = ""
                        report.lyrics_source = "whisper (instrumental)"
                    else:
                        report.lyrics_source = "whisper"
                        lyrics = None
                        if cfg.section_tags == "claude":
                            lyrics = claude_sections(raw, cfg.claude_model)
                            if lyrics is not None:
                                report.lyrics_source = "whisper+claude"
                        if lyrics is None:
                            lyrics = heuristic_sections(chunks) if cfg.section_tags != "none" else raw + "\n"
            elif want_lyrics and lyrics is None:
                report.notes.append("no lyrics found on LRCLIB")
            if lyrics is not None and cfg.section_tags == "claude" and report.lyrics_source == "lrclib":
                improved = claude_sections(lyrics, cfg.claude_model)
                if improved:
                    lyrics, report.lyrics_source = improved, "lrclib+claude"
            if want_lyrics and lyrics is not None:
                lyrics_path.write_text(lyrics, encoding="utf-8")
                report.lyrics_chars = len(lyrics)
            report.language = language

            # ---- style
            if want_style:
                if cfg.style_source == "clap":
                    if tagger is None:
                        tagger = ClapTagger(cfg.clap_model, device)
                    emb = tagger.audio_embed(excerpts(mono48, cfg.excerpt_seconds, cfg.excerpts))
                    tags = pick_tags(tagger, emb, cfg.temperature)
                    tempo, key = tempo_and_key(mono48)
                    if lyrics == "":
                        tags["vocal"] = ["instrumental"]
                    style = assemble_style(language, tags, tempo)
                    report.tags = {k: v for k, v in tags.items() if k != "scores"}
                    report.tags["scores"] = {k: [(t, round(p, 3)) for t, p in v] for k, v in tags["scores"].items()}
                    report.tempo, report.key = tempo, key
                    report.style_source = "clap"
                else:
                    style = cfg.default_style
                    report.style_source = "default"
                if style:
                    style_path.write_text(style + "\n", encoding="utf-8")
                    report.style = style
            reports.append(report)
            if progress:
                progress(index + 1, len(files), path.name)
    finally:
        if whisper is not None:
            whisper.close()
        if tagger is not None:
            tagger.close()
    (folder / "_prepare_report.json").write_text(
        json.dumps([asdict(r) for r in reports], indent=1, ensure_ascii=False), encoding="utf-8")
    return reports


def summarize(reports: list[ItemReport]) -> str:
    lines = [f"{len(reports)} files"]
    for r in reports:
        name = Path(r.file).name
        bits = []
        if r.style:
            bits.append(f"style[{r.style_source}]: {r.style}")
        if r.lyrics_source:
            bits.append(f"lyrics[{r.lyrics_source}]: {r.lyrics_chars} chars")
        if r.key:
            bits.append(f"key {r.key}")
        bits += r.notes
        lines.append(f"  {name}: " + " | ".join(bits))
    review = [Path(r.file).name for r in reports if r.notes or (r.lyrics_source and "whisper" in r.lyrics_source)]
    if review:
        lines.append("review suggested: " + ", ".join(review))
    return "\n".join(lines)


__all__ = ["SidecarConfig", "ItemReport", "prepare_folder", "summarize", "WHISPER_CHOICES", "file_metadata",
           "parse_stem", "heuristic_sections", "plain_to_sections", "assemble_style", "fetch_lrclib", "clean_segments"]
