"""Deterministic format checks, preserving all original audio frames."""
import hashlib
import io
import math
import subprocess
from pathlib import Path

from .domain import Failure

MAX_REFERENCE_BYTES = 2 * 1024 * 1024


def pcm16_with_headroom(samples):
    """Quantize unclipped PCM-unit floats with one gain for the whole utterance.

    Called before upstream int16 conversion, never on already clipped audio.
    A 3 dB margin leaves room for the editing copy's sample-rate conversion.
    Quiet takes are not boosted; relative levels across chunks are preserved.
    """
    import numpy as np
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise Failure("invalid_engine_waveform", 500)
    ceiling = 32767 * 10 ** (-3 / 20)
    peak = float(np.abs(values).max())
    gain = min(1.0, ceiling / peak) if peak else 1.0
    return np.rint(values * gain).astype(np.int16)


def inspect_wav(data, reference=False):
    import numpy as np
    import soundfile as sf
    try:
        with sf.SoundFile(io.BytesIO(data)) as wav:
            rate, channels, frames = wav.samplerate, wav.channels, wav.frames
            if wav.format not in ("WAV", "WAVEX") or wav.subtype not in ("PCM_16", "PCM_24") or channels != 1:
                raise Failure("unsupported_audio_format")
            width = 2 if wav.subtype == "PCM_16" else 3
            if rate not in (16000, 22050, 24000, 44100, 48000) or frames <= 0:
                raise Failure("invalid_audio")
            duration = frames / rate
            if reference and (len(data) > MAX_REFERENCE_BYTES or not 1 <= duration <= 15):
                raise Failure("reference_duration_or_size")
            if not reference and duration > 180:
                raise Failure("output_too_long")
            samples = wav.read(frames, dtype="float32")
            if len(samples) != frames or not np.isfinite(samples).all():
                raise Failure("truncated_audio")
    except (sf.LibsndfileError, ValueError):
        raise Failure("invalid_wav") from None
    amplitudes = np.abs(samples)
    peak = float(amplitudes.max())
    if peak == 0:
        raise Failure("silent_audio")
    clipped = int(np.count_nonzero(amplitudes >= 1 - 1 / (2 ** (width * 8 - 1))))
    return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data), "sample_rate_hz": rate,
            "channels": channels, "bit_depth": width * 8, "duration_seconds": duration,
            "peak_dbfs": 20 * math.log10(peak), "clipped_samples": clipped,
            "quiet_sample_fraction": float(np.count_nonzero(amplitudes <= 0.001)) / frames}


def editing_copy(original: Path, output: Path):
    result = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(original), "-map", "0:a:0",
                             "-ac", "1", "-ar", "48000", "-c:a", "pcm_s24le", "-f", "wav", str(output)],
                            capture_output=True, timeout=120)
    if result.returncode:
        raise Failure("audio_conversion_failed", 500)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
