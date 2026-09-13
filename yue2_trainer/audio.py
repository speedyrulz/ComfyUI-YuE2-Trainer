"""Audio loading, resampling and windowed VAE encoding for YuE2 latents."""
from __future__ import annotations

import logging
import math
from pathlib import Path

import torch

from .constants import LATENT_CHANNELS, SAMPLES_PER_FRAME, SAMPLE_RATE


def load_audio(path) -> tuple[torch.Tensor, int]:
    """Return (waveform [channels, samples] float32, sample_rate)."""
    path = str(path)
    errors = []
    try:
        import torchaudio
        wave, sr = torchaudio.load(path)
        return wave.float(), int(sr)
    except Exception as e:  # noqa: BLE001
        errors.append(f"torchaudio: {e}")
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), int(sr)
    except Exception as e:  # noqa: BLE001
        errors.append(f"soundfile: {e}")
    try:
        import av
        with av.open(path) as container:
            stream = container.streams.audio[0]
            sr = stream.rate
            frames = []
            for frame in container.decode(stream):
                frames.append(torch.from_numpy(frame.to_ndarray()))
            if not frames:
                raise ValueError("no audio frames")
            data = torch.cat(frames, dim=-1).float()
            if data.ndim == 1:
                data = data[None]
            # packed formats deliver [1, samples*channels]
            if data.shape[0] == 1 and stream.channels > 1 and data.shape[1] % stream.channels == 0:
                data = data.view(-1, stream.channels).T.contiguous()
            return data, int(sr)
    except Exception as e:  # noqa: BLE001
        errors.append(f"av: {e}")
    raise RuntimeError(f"Could not decode audio {path}: " + "; ".join(errors))


def to_stereo_48k(wave: torch.Tensor, sr: int) -> torch.Tensor:
    """[channels, samples] at any rate -> [2, samples] at 48 kHz."""
    if wave.ndim == 1:
        wave = wave[None]
    if wave.shape[0] == 1:
        wave = wave.repeat(2, 1)
    elif wave.shape[0] > 2:
        wave = wave[:2]
    if sr != SAMPLE_RATE:
        import torchaudio
        wave = torchaudio.functional.resample(wave, sr, SAMPLE_RATE)
    return wave.contiguous()


def frames_for_samples(samples: int) -> int:
    return samples // SAMPLES_PER_FRAME


@torch.no_grad()
def encode_latents(vae, wave: torch.Tensor, window_seconds: float = 60.0, halo_frames: int = 32,
                   progress=None) -> torch.Tensor:
    """Encode a 48 kHz stereo waveform [2, N] into YuE2 latents [64, T] (float32, CPU).

    The song is encoded in overlapping windows so that arbitrarily long audio fits in
    memory; each window keeps a halo on both sides which is cropped away, so the result
    is (up to boundary effects far smaller than a frame) identical to one-shot encoding.
    """
    total_frames = frames_for_samples(wave.shape[-1])
    if total_frames < 1:
        raise ValueError("Audio shorter than one latent frame (40 ms)")
    wave = wave[:, : total_frames * SAMPLES_PER_FRAME]
    window = max(int(window_seconds * SAMPLE_RATE // SAMPLES_PER_FRAME), halo_frames * 2 + 1)
    out = torch.empty((LATENT_CHANNELS, total_frames), dtype=torch.float32)
    starts = list(range(0, total_frames, window))
    for index, a in enumerate(starts):
        b = min(a + window, total_frames)
        wa = max(0, a - halo_frames)
        wb = min(total_frames, b + halo_frames)
        chunk = wave[:, wa * SAMPLES_PER_FRAME: wb * SAMPLES_PER_FRAME]
        latent = vae.encode(chunk[None].movedim(1, -1))  # comfy VAE expects [B, samples, channels]
        latent = latent[0].float().cpu()  # [64, frames]
        expected = wb - wa
        if latent.shape[-1] != expected:
            # Encoder returned a different count; align by centre crop / pad.
            logging.debug("YuE2 trainer: encoder returned %d frames for %d expected", latent.shape[-1], expected)
            if latent.shape[-1] > expected:
                off = (latent.shape[-1] - expected) // 2
                latent = latent[:, off: off + expected]
            else:
                latent = torch.nn.functional.pad(latent, (0, expected - latent.shape[-1]), mode="replicate")
        out[:, a:b] = latent[:, a - wa: a - wa + (b - a)]
        if progress is not None:
            progress(index + 1, len(starts))
    return out


def crop_audio(wave: torch.Tensor, max_seconds: float) -> torch.Tensor:
    if max_seconds and max_seconds > 0:
        n = int(math.floor(max_seconds * SAMPLE_RATE))
        return wave[:, :n]
    return wave


def audio_seconds(path) -> float | None:
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:  # noqa: BLE001
        return None


__all__ = ["load_audio", "to_stereo_48k", "encode_latents", "crop_audio", "audio_seconds", "frames_for_samples"]
