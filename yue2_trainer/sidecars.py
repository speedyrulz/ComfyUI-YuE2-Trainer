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

DEFAULT_WHISPER = "openai/whisper-large-v3"
DEFAULT_CLAP = "laion/larger_clap_music_and_speech"
DEFAULT_CLAUDE = "claude-opus-5"
DEFAULT_OMNI = "Qwen/Qwen2.5-Omni-3B"
DEFAULT_MOSS = "OpenMOSS-Team/MOSS-Audio-4B-Instruct"
PRECISIONS = ["bf16", "nf4"]
WHISPER_CHOICES = [DEFAULT_WHISPER, "openai/whisper-large-v3-turbo", "openai/whisper-medium", "openai/whisper-small"]

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
    lyrics_source: str = "lrclib+whisper"   # lrclib+whisper | whisper | lrclib | lrclib+moss-audio | moss-audio | none
    whisper_model: str = DEFAULT_WHISPER
    language: str = "auto"                  # auto or a Whisper language code (en, zh, ja, ...)
    separate_vocals: bool = True            # run demucs and transcribe the vocal stem (much more accurate)
    style_source: str = "moss-audio"        # moss-audio | qwen-omni | clap | none
    clap_model: str = DEFAULT_CLAP
    omni_model: str = DEFAULT_OMNI
    moss_model: str = DEFAULT_MOSS
    precision: str = "bf16"                 # bf16 | nf4  (audio-LLM weight precision; nf4 uses about a third of the VRAM)
    artist: str = ""                        # prepended to every style prompt (also works as a trigger phrase)
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

def fetch_lrclib(artist: Optional[str], title: str, duration: Optional[float], timeout: float = 15.0,
                 tolerance: float = 3.0) -> Optional[dict]:
    """Return {"lyrics", "artist", "title", "duration"} or None.

    Exact artist+title+duration first; otherwise a search whose best hit must be within
    ``tolerance`` seconds of the file's duration (title-only searches are ambiguous, so they
    are only accepted with a duration match). The caller may additionally verify the text
    against a Whisper transcript.
    """
    import requests
    headers = {"User-Agent": "ComfyUI-YuE2-Trainer/0.1 (https://github.com/speedyrulz/ComfyUI-YuE2-Trainer)"}

    def pack(c):
        return {"lyrics": c["plainLyrics"].strip(), "artist": c.get("artistName"), "title": c.get("trackName"),
                "duration": c.get("duration")}
    try:
        if artist and duration:
            r = requests.get("https://lrclib.net/api/get", timeout=timeout, headers=headers,
                             params={"artist_name": artist, "track_name": title, "duration": int(round(duration))})
            if r.status_code == 200 and r.json().get("plainLyrics"):
                return pack(r.json())
        params = {"track_name": title}
        if artist:
            params["artist_name"] = artist
        r = requests.get("https://lrclib.net/api/search", timeout=timeout, headers=headers, params=params)
        if r.status_code != 200:
            return None
        candidates = [c for c in r.json() if c.get("plainLyrics")]
        if not candidates:
            return None
        if duration:
            candidates.sort(key=lambda c: abs(float(c.get("duration") or 0) - duration))
            best = candidates[0]
            if abs(float(best.get("duration") or 0) - duration) <= tolerance:
                return pack(best)
            return None
        return pack(candidates[0]) if artist else None
    except Exception as exc:  # noqa: BLE001
        LOG.warning("LRCLIB lookup failed for %s - %s: %s", artist, title, exc)
        return None


def lyrics_match_transcript(lyrics: str, transcript: str) -> float:
    """Fraction of distinct transcript words (or CJK characters) that occur in the lyrics."""
    def tokens(text):
        text = text.lower()
        words = set(re.findall(r"[a-z0-9']{3,}", text))
        cjk = set(re.findall(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", text))
        return words | cjk
    have, want = tokens(lyrics), tokens(transcript)
    if not want:
        return 0.0
    return len(want & have) / len(want)


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
    def detect_language(self, wave16k: np.ndarray, windows: int = 3) -> Optional[str]:
        """Majority vote over the loudest windows so instrumental passages do not decide the language."""
        try:
            starts = list(range(0, max(1, len(wave16k) - self.CHUNK + 1), self.CHUNK // 2)) or [0]
            starts.sort(key=lambda i: -float(np.abs(wave16k[i:i + self.CHUNK]).mean()))
            votes: dict[str, int] = {}
            for i in starts[:windows]:
                ids = self.model.detect_language(self._features([wave16k[i:i + self.CHUNK]]))
                code = self.processor.tokenizer.decode(ids.reshape(-1)[:1]).strip("<|>")
                if code and code != "nospeech":
                    votes[code] = votes.get(code, 0) + 1
            return max(votes, key=votes.get) if votes else None
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

    @staticmethod
    def _split_sentences(segments: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
        """large-v3 often packs several sung lines into one segment; split on sentence punctuation."""
        out = []
        for start, end, text in segments:
            parts = [p.strip(" ,;") for p in re.split(r"(?<=[.!?。！？])\s+|(?<=[。！？])", text) if p.strip(" ,;")]
            if len(parts) <= 1:
                out.append((start, end, text.strip().rstrip(".。")))
                continue
            step = (end - start) / len(parts)
            for i, part in enumerate(parts):
                out.append((start + i * step, start + (i + 1) * step, part.rstrip(".。")))
        return out

    @torch.no_grad()
    def transcribe(self, wave16k: np.ndarray, language: Optional[str] = None,
                   batch_size: int = 8) -> list[tuple[float, float, str]]:
        pieces = [wave16k[i:i + self.CHUNK] for i in range(0, len(wave16k), self.CHUNK)]
        pieces = [pc for pc in pieces if len(pc) > 16000]
        segments = []
        for b in range(0, len(pieces), batch_size):
            batch = pieces[b:b + batch_size]
            kwargs = {"task": "transcribe", "return_timestamps": True}
            if language:
                kwargs["language"] = language
            ids = self.model.generate(self._features(batch), **kwargs)
            texts = [self.processor.tokenizer.decode(seq.tolist(), skip_special_tokens=True, decode_with_timestamps=True)
                     for seq in ids]
            for j, text in enumerate(texts):
                offset = (b + j) * 30.0
                segments.extend(self._segments(text, offset, len(batch[j]) / 16000))
        return self._split_sentences(segments)


class VocalSeparator:
    """demucs (htdemucs) vocal stem; lazily loaded, optional dependency."""

    def __init__(self, device: str = "cuda"):
        from demucs.pretrained import get_model
        self.device = torch.device(device)
        LOG.info("loading demucs htdemucs")
        self.model = get_model("htdemucs").to(self.device).eval()

    def close(self):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def vocals(self, wave48k: torch.Tensor) -> torch.Tensor:
        """[2, N] 48 kHz -> vocal stem [2, N] 48 kHz."""
        import torchaudio
        from demucs.apply import apply_model
        sr = self.model.samplerate
        mix = torchaudio.functional.resample(wave48k, 48000, sr)
        ref = mix.mean(0)
        mix = (mix - ref.mean()) / (ref.std() + 1e-8)
        stems = apply_model(self.model, mix[None].to(self.device), split=True, overlap=0.25, progress=False)[0]
        vocals = stems[self.model.sources.index("vocals")].cpu() * (ref.std() + 1e-8) + ref.mean()
        return torchaudio.functional.resample(vocals, sr, 48000)


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
    r"(subtitles? by|subscribe|thanks? for watching|we'll be right back|amara\.org|字幕|作曲|作词|作詞|編曲|编曲|词曲|作品|制作|翻译|翻譯|"
    r"転載|ご視聴ありがとう|チャンネル登録|시청해 주셔서|구독)", re.I)


LOOP_RE = re.compile(r"(?i)((?:[^,，、\s]+\s+){0,6}?[^,，、\s]+)(?:[,，、\s]+\1(?=[,，、\s]|$)){2,}")


def collapse_loops(text: str) -> str:
    """'my heart, my heart, my heart, my heart' -> 'my heart' (decoder loops inside one line)."""
    previous = None
    while previous != text:
        previous = text
        text = LOOP_RE.sub(lambda m: m.group(1), text)
    return text.strip(" ,，、")


def clean_segments(chunks: list[tuple[float, float, str]], max_consecutive: int = 2,
                   max_total: int = 8) -> tuple[list[tuple[float, float, str]], int]:
    """Drop transcription hallucinations: credit/subtitle phrases, long runs of one line, and longer lines
    that are repeated an implausible number of times (short refrains such as "yeah" or "passion" are kept)."""
    chunks = [(a, b, collapse_loops(t)) for a, b, t in chunks]
    counts: dict[str, int] = {}
    for _, _, text in chunks:
        counts[_norm(text)] = counts.get(_norm(text), 0) + 1
    out, dropped, previous, run = [], 0, None, 0
    for start, end, text in chunks:
        key = _norm(text)
        if not key or HALLUCINATION_RE.search(text) or (counts[key] >= max_total and len(key) >= 8):
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


# ── style: audio LLM (Qwen2.5-Omni) ───────────────────────────────────────────

OMNI_SYSTEM = ("You are a music tagging assistant for a text-to-music model. Listen to the audio excerpts and describe "
               "the music as ONE line of comma-separated tags. Always include, in this order: 2 genre or subgenre tags; "
               "2 mood or energy tags; the vocal type (male vocal, female vocal, duet, rap, or instrumental) and one tag "
               "for the vocal delivery (e.g. breathy, powerful, smooth, raspy); 3 to 5 tags naming the main instruments "
               "and sounds you hear; 2 production or era descriptors (e.g. lo-fi, polished, 80s synth, live band, "
               "distorted); one tag for the rhythmic feel (e.g. driving, laid-back, syncopated, half-time). "
               "Lowercase, no full sentences, no artist or song names, no BPM or time signature. Output only the tag line.")
OMNI_USER = "Describe this music as tags."


class OmniTagger:
    """Qwen2.5-Omni (thinker only): listens to excerpts and writes a free-form tag line."""

    def __init__(self, model_name: str = DEFAULT_OMNI, device: str = "cuda", precision: str = "bf16"):
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
        self.device = torch.device(device)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        LOG.info("loading %s (%s)", model_name, precision)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
        quant = quantization_config(precision) if self.device.type == "cuda" else None
        if quant is not None:
            self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                model_name, quantization_config=quant, device_map={"": str(self.device)})
        else:
            self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype)
            self.model.to(self.device)
        self.model.eval()

    def close(self):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def describe(self, wave16k: np.ndarray) -> str:
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": OMNI_SYSTEM}]},
            {"role": "user", "content": [{"type": "audio", "audio": wave16k}, {"type": "text", "text": OMNI_USER}]},
        ]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, audio=[wave16k], sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(self.model.dtype)
        out = self.model.generate(**inputs, max_new_tokens=160, do_sample=False)
        generated = out[:, inputs["input_ids"].shape[1]:]
        reply = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return clean_tag_line(reply)


def clean_tag_line(reply: str) -> str:
    """Normalise an LLM answer into 'tag, tag, tag'."""
    line = reply.strip().splitlines()
    line = next((l for l in line if l.strip()), "")
    line = re.sub(r"^(tags?|description)\s*:\s*", "", line, flags=re.I).strip().strip("\"'`.")
    tags = [t.strip(" .;\"'`") for t in re.split(r"[,;\n]+", line)]
    seen, out = set(), []
    for t in tags:
        if t and t.lower() not in seen and len(t) < 48:
            seen.add(t.lower())
            out.append(t)
    return ", ".join(out)


def omni_excerpt(mono48: np.ndarray, seconds: float = 20.0, count: int = 3) -> np.ndarray:
    """Concatenate ``count`` excerpts (start / middle / end) into one 16 kHz clip for the audio LLM."""
    import torchaudio
    pieces = excerpts(mono48, seconds, count)
    joined = np.concatenate(pieces)
    return torchaudio.functional.resample(torch.from_numpy(joined)[None], 48000, 16000)[0].numpy()


# ── style + lyrics: MOSS-Audio (audio-understanding LLM) ───────────────────────

MOSS_TAG_PROMPT = ("Describe this music for a text-to-music model as ONE line of comma-separated tags. Include, in this "
                   "order: two genre or subgenre tags; two mood or energy tags; the vocal type (male vocal, female vocal, "
                   "duet, rap, or instrumental) plus one tag for how the singer sounds; three to five tags naming the "
                   "specific instruments and sounds you hear; two tags for the production style or era; one tag for the "
                   "rhythmic feel. Be specific to this recording. Lowercase, no sentences, no artist or song names, no BPM "
                   "or time signature. Output only the tag line.")
MOSS_LYRICS_PROMPT = ("Transcribe the sung lyrics of this song exactly as sung, in the original language. Write one sung "
                      "line per output line and leave one blank line between sections such as verses and choruses. "
                      "Do not translate, do not add titles, timestamps, commentary or descriptions. If there is no "
                      "singing, output exactly: [instrumental]")


def quantization_config(precision: str):
    if precision in ("", "bf16", "fp16", "none"):
        return None
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # noqa: F401
    except ImportError:
        LOG.warning("bitsandbytes is not installed; loading in bf16 instead of %s", precision)
        return None
    if precision == "int8":  # kept for the CLI; produced empty answers in testing, so it is not offered in the node
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)


class MossAudioTagger:
    """MOSS-Audio-4B-Instruct: music description (tags) and sung-lyrics transcription."""

    def __init__(self, model_name: str = DEFAULT_MOSS, device: str = "cuda", precision: str = "bf16"):
        from .vendor.moss_audio.modeling_moss_audio import MossAudioModel
        from .vendor.moss_audio.processing_moss_audio import MossAudioProcessor
        self.device = torch.device(device)
        LOG.info("loading %s (%s)", model_name, precision)
        kwargs = {"device_map": {"": str(self.device)} if self.device.type == "cuda" else None}
        quant = quantization_config(precision) if self.device.type == "cuda" else None
        if quant is not None:
            kwargs["quantization_config"] = quant
        else:
            kwargs["dtype"] = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = MossAudioModel.from_pretrained(model_name, **kwargs).eval()
        if self.model.device.type != self.device.type:
            self.model.to(self.device)
        self.processor = MossAudioProcessor.from_pretrained(model_name, enable_time_marker=True)
        self.sample_rate = int(self.processor.config.mel_sr)

    def close(self):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def ask(self, wave: np.ndarray, prompt: str, max_new_tokens: int = 256, repetition_penalty: float = 1.0) -> str:
        """``wave`` is mono at ``self.sample_rate``."""
        inputs = self.processor(text=prompt, audios=[wave.astype(np.float32)], return_tensors="pt")
        inputs = inputs.to(self.model.device)
        if inputs.get("audio_data") is not None:
            inputs["audio_data"] = inputs["audio_data"].to(self.model.dtype)
        inputs["audio_input_mask"] = inputs["input_ids"] == self.processor.audio_token_id
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, use_cache=True,
                                  repetition_penalty=repetition_penalty)
        return self.processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def describe(self, wave: np.ndarray) -> str:
        return clean_tag_line(self.ask(wave, MOSS_TAG_PROMPT, max_new_tokens=160))

    def transcribe(self, wave: np.ndarray, chunk_seconds: float = 60.0) -> list[tuple[float, float, str]]:
        """Lyrics as (start, end, line) segments; long songs are transcribed in chunks."""
        sr, n = self.sample_rate, int(chunk_seconds * self.sample_rate)
        segments = []
        for start in range(0, len(wave), n):
            piece = wave[start:start + n]
            if len(piece) < sr:
                continue
            text = self.ask(piece, MOSS_LYRICS_PROMPT, max_new_tokens=700, repetition_penalty=1.1)
            if "[instrumental]" in text.lower():
                continue
            lines = [l.strip() for l in text.splitlines()]
            lines = [l for l in lines if l and not l.startswith(("[", "(", "<"))]
            offset, step = start / sr, (len(piece) / sr) / max(1, len(lines))
            for i, line in enumerate(lines):
                segments.append((offset + i * step, offset + (i + 1) * step, line))
        return segments


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
    for noisy in ("httpx", "huggingface_hub", "urllib3"):  # hub file probes are not useful in the ComfyUI log
        logging.getLogger(noisy).setLevel(logging.WARNING)
    files = list_audio(folder, recursive)
    if not files:
        raise ValueError(f"No audio files in {folder}")
    device = _resolve_device(cfg.device)
    need_whisper = cfg.lyrics_source in ("lrclib+whisper", "whisper")  # language detection rides on Whisper
    whisper = tagger = separator = omni = moss = None
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
            language = None if cfg.language in ("", "auto") else cfg.language
            use_whisper = cfg.lyrics_source in ("whisper", "lrclib+whisper")
            use_moss_lyrics = cfg.lyrics_source in ("moss-audio", "lrclib+moss-audio")

            def moss_tagger():
                nonlocal moss
                if moss is None:
                    moss = MossAudioTagger(cfg.moss_model, device, cfg.precision)
                return moss

            def moss_audio():
                """Vocal stem (if enabled) at MOSS's sample rate."""
                nonlocal separator
                source = wave
                if cfg.separate_vocals:
                    try:
                        if separator is None:
                            separator = VocalSeparator(device)
                        source = separator.vocals(wave)
                    except Exception as exc:  # noqa: BLE001
                        LOG.warning("vocal separation failed on %s: %s", path.name, exc)
                return _resample_mono(source, moss_tagger().sample_rate)

            def whisper_audio():
                nonlocal wave16, separator
                if wave16 is None:
                    source = wave
                    if cfg.separate_vocals:
                        try:
                            if separator is None:
                                separator = VocalSeparator(device)
                            source = separator.vocals(wave)
                            report.notes.append("transcribed the demucs vocal stem")
                        except ImportError:
                            report.notes.append("demucs not installed; transcribed the full mix")
                        except Exception as exc:  # noqa: BLE001
                            LOG.warning("vocal separation failed on %s: %s", path.name, exc)
                            report.notes.append("vocal separation failed; transcribed the full mix")
                    wave16 = _resample_mono(source, 16000)
                return wave16

            # ---- lyrics
            lyrics = None
            if want_lyrics and cfg.lyrics_source in ("lrclib", "lrclib+whisper"):
                if not meta["artist"]:
                    report.notes.append("no artist tag / 'Artist - Title' name; LRCLIB matched by title+duration only")
                found = fetch_lrclib(meta["artist"], meta["title"], meta["duration"] or report.seconds)
                if found and len(found["lyrics"]) >= cfg.min_lyrics_chars:
                    accept = True
                    if use_moss_lyrics:  # cross-check the database hit against what MOSS-Audio hears
                        probe = moss_tagger().transcribe(moss_audio()[: moss_tagger().sample_rate * 90])
                        score = lyrics_match_transcript(found["lyrics"], " ".join(t for _, _, t in probe))
                        accept = score >= 0.35
                        report.notes.append(f"LRCLIB '{found['artist']} - {found['title']}' matched transcript {score:.0%}"
                                            + ("" if accept else " -> rejected"))
                    elif use_whisper:  # cross-check the database hit against what is actually sung
                        if whisper is None:
                            whisper = WhisperTranscriber(cfg.whisper_model, device)
                        probe_audio = whisper_audio()
                        probe = whisper.transcribe(probe_audio[: 16000 * 90], language or whisper.detect_language(probe_audio))
                        score = lyrics_match_transcript(found["lyrics"], " ".join(t for _, _, t in probe))
                        accept = score >= 0.35
                        report.notes.append(f"LRCLIB '{found['artist']} - {found['title']}' matched transcript {score:.0%}"
                                            + ("" if accept else " -> rejected"))
                    if accept:
                        lyrics = plain_to_sections(found["lyrics"])
                        report.lyrics_source = "lrclib"
                elif found is None:
                    report.notes.append(f"LRCLIB: nothing found for '{meta['artist'] or '?'} - {meta['title']}'")
            if use_moss_lyrics and want_lyrics and lyrics is None:
                chunks, dropped = clean_segments(moss_tagger().transcribe(moss_audio()))
                if dropped:
                    report.notes.append(f"{dropped} suspected hallucinated lines removed")
                raw = "\n".join(t for _, _, t in chunks)
                if cfg.separate_vocals:
                    report.notes.append("transcribed the demucs vocal stem with MOSS-Audio")
                if len(raw) < cfg.min_lyrics_chars:
                    report.notes.append("little or no singing found; treated as instrumental")
                    lyrics, report.lyrics_source = "", "moss-audio (instrumental)"
                else:
                    report.lyrics_source = "moss-audio"
                    lyrics = None
                    if cfg.section_tags == "claude":
                        lyrics = claude_sections(raw, cfg.claude_model)
                        if lyrics is not None:
                            report.lyrics_source = "moss-audio+claude"
                    if lyrics is None:
                        lyrics = heuristic_sections(chunks) if cfg.section_tags != "none" else raw + "\n"
            if use_whisper and ((want_lyrics and lyrics is None) or (want_style and language is None)):
                if whisper is None:
                    whisper = WhisperTranscriber(cfg.whisper_model, device)
                audio16 = whisper_audio()
                if language is None:
                    language = whisper.detect_language(audio16)
                    if language:
                        report.notes.append(f"language detected: {LANGUAGES.get(language, language)}")
                if want_lyrics and lyrics is None:
                    chunks, dropped = clean_segments(whisper.transcribe(audio16, language))
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
                report.notes.append("no lyrics written (LRCLIB only)")
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
                if cfg.style_source == "moss-audio":
                    described = moss_tagger().describe(_resample_mono(
                        torch.from_numpy(np.concatenate(excerpts(mono48, 20.0, 3)))[None].repeat(2, 1),
                        moss_tagger().sample_rate))
                    tempo, key = tempo_and_key(mono48)
                    parts = [LANGUAGES.get(language, language)] if language else []
                    parts.append(described)
                    if tempo:
                        parts.append(f"{int(tempo)} BPM")
                    style = ", ".join(p for p in parts if p)
                    report.tags = {"moss-audio": described}
                    report.tempo, report.key = tempo, key
                    report.style_source = "moss-audio"
                elif cfg.style_source == "qwen-omni":
                    if omni is None:
                        omni = OmniTagger(cfg.omni_model, device, cfg.precision)
                    described = omni.describe(omni_excerpt(mono48))
                    tempo, key = tempo_and_key(mono48)
                    parts = [LANGUAGES.get(language, language)] if language else []
                    parts.append(described)
                    if tempo:
                        parts.append(f"{int(tempo)} BPM")
                    style = ", ".join(p for p in parts if p)
                    report.tags = {"omni": described}
                    report.tempo, report.key = tempo, key
                    report.style_source = "qwen-omni"
                elif cfg.style_source == "clap":
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
                artist = cfg.artist.strip().strip(",")
                if artist:
                    style = f"{artist}, {style}" if style else artist
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
        if separator is not None:
            separator.close()
        if omni is not None:
            omni.close()
        if moss is not None:
            moss.close()
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
           "parse_stem", "heuristic_sections", "plain_to_sections", "assemble_style", "fetch_lrclib", "clean_segments",
           "lyrics_match_transcript", "VocalSeparator", "OmniTagger", "clean_tag_line", "DEFAULT_OMNI", "collapse_loops",
           "MossAudioTagger", "DEFAULT_MOSS", "PRECISIONS", "quantization_config"]
