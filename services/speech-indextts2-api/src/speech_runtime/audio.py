"""Deterministic format checks, preserving all original audio frames."""
import functools
import hashlib
import io
import math
import subprocess
from pathlib import Path

from .domain import Failure

MAX_REFERENCE_BYTES = 2 * 1024 * 1024
TRUE_PEAK_CEILING_DBTP = -1.0


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
    leading, trailing = silence_edges(samples, rate)
    return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data), "sample_rate_hz": rate,
            "channels": channels, "bit_depth": width * 8, "duration_seconds": duration,
            "peak_dbfs": 20 * math.log10(peak), "clipped_samples": clipped,
            "quiet_sample_fraction": float(np.count_nonzero(amplitudes <= 0.001)) / frames,
            "true_peak_dbtp": true_peak_dbtp(samples), "integrated_lufs": integrated_lufs(samples, rate),
            "leading_silence_seconds": leading, "trailing_silence_seconds": trailing}


def _fft_filter(signal, kernel, block=1 << 16):
    """Linear convolution by overlap-add, bounded memory for long outputs."""
    import numpy as np
    signal = np.asarray(signal, dtype=np.float64)
    n = 1 << (block + len(kernel) - 2).bit_length()
    spectrum = np.fft.rfft(kernel, n)
    out = np.zeros(len(signal) + len(kernel) - 1)
    for start in range(0, len(signal), block):
        piece = signal[start:start + block]
        filtered = np.fft.irfft(np.fft.rfft(piece, n) * spectrum, n)[:len(piece) + len(kernel) - 1]
        out[start:start + len(filtered)] += filtered
    return out


def true_peak_dbtp(samples, factor=4, taps_per_phase=16):
    """Inter-sample peak by 4x oversampling with a Hann-windowed sinc (BS.1770-4 Annex 2 style)."""
    import numpy as np
    x = np.asarray(samples, dtype=np.float64)
    half = factor * taps_per_phase // 2
    t = np.arange(-half, half + 1)
    kernel = np.sinc(t / factor) * np.hanning(len(t))
    kernel *= factor / kernel.sum()
    # Polyphase: each output phase is a short direct convolution of the original samples.
    peak = max(float(np.abs(np.convolve(x, kernel[p::factor])).max()) for p in range(factor))
    return 20 * math.log10(max(peak, float(np.abs(x).max()), 1e-12))


def _biquad_impulse(b, a, length):
    import numpy as np
    b = [v / a[0] for v in b]
    a1, a2 = a[1] / a[0], a[2] / a[0]
    out = np.zeros(length)
    x1 = x2 = y1 = y2 = 0.0
    for i in range(length):
        x0 = 1.0 if i == 0 else 0.0
        y0 = b[0] * x0 + b[1] * x1 + b[2] * x2 - a1 * y1 - a2 * y2
        out[i], x2, x1, y2, y1 = y0, x1, x0, y1, y0
    return out


@functools.lru_cache(maxsize=8)
def _k_weighting_kernel(rate):
    """K-weighting (BS.1770) as one-second impulse response of shelf + high-pass biquads."""
    import numpy as np
    w0 = 2 * math.pi * 1681.9744509555319 / rate
    gain, q = 10 ** (3.99984385397 / 40), 0.7071752369554193
    cos, root = math.cos(w0), 2 * math.sqrt(gain) * math.sin(w0) / (2 * q)
    shelf_b = [gain * ((gain + 1) + (gain - 1) * cos + root), -2 * gain * ((gain - 1) + (gain + 1) * cos),
               gain * ((gain + 1) + (gain - 1) * cos - root)]
    shelf_a = [(gain + 1) - (gain - 1) * cos + root, 2 * ((gain - 1) - (gain + 1) * cos),
               (gain + 1) - (gain - 1) * cos - root]
    w0 = 2 * math.pi * 38.13547087613982 / rate
    alpha, cos = math.sin(w0) / (2 * 0.5003270373253953), math.cos(w0)
    hp_b = [(1 + cos) / 2, -(1 + cos), (1 + cos) / 2]
    hp_a = [1 + alpha, -2 * cos, 1 - alpha]
    return _fft_filter(_biquad_impulse(shelf_b, shelf_a, rate), _biquad_impulse(hp_b, hp_a, rate))[:rate]


def integrated_lufs(samples, rate):
    """Gated integrated loudness of a mono signal; floors at -70 LUFS when every block is gated."""
    import numpy as np
    y = _fft_filter(np.asarray(samples, dtype=np.float64), _k_weighting_kernel(rate))[:len(samples)]
    block, step = int(0.4 * rate), int(0.1 * rate)
    if len(y) <= block:
        powers = np.array([np.mean(y ** 2)])
    else:
        energy = np.concatenate([[0.0], np.cumsum(y ** 2)])
        starts = np.arange(0, len(y) - block + 1, step)
        powers = (energy[starts + block] - energy[starts]) / block
    loud = lambda p: -0.691 + 10 * math.log10(p) if p > 0 else -math.inf
    kept = powers[[loud(p) > -70 for p in powers]]
    if not len(kept):
        return -70.0
    relative = loud(float(kept.mean())) - 10
    kept = kept[[loud(p) > relative for p in kept]]
    return max(loud(float(kept.mean())), -70.0)


def silence_edges(samples, rate, threshold_db=-50.0, frame_seconds=0.01):
    """Leading and trailing silence measured on 10 ms RMS frames."""
    import numpy as np
    x = np.asarray(samples, dtype=np.float64)
    size = max(1, int(rate * frame_seconds))
    count = len(x) // size
    if count == 0:
        return 0.0, 0.0
    rms = np.sqrt(np.mean(x[:count * size].reshape(count, size) ** 2, axis=1))
    loud = np.nonzero(rms > 10 ** (threshold_db / 20))[0]
    duration = len(x) / rate
    if not len(loud):
        return duration, duration
    return float(loud[0] * size / rate), float(max(0.0, duration - (loud[-1] + 1) * size / rate))


def quality_warnings(native, duration_difference=None):
    """Advisory only: warnings never fail a task or trigger regeneration."""
    warnings = []
    if native["clipped_samples"] > 0:
        warnings.append({"code": "clipping", "level": "error"})
    if native["true_peak_dbtp"] > TRUE_PEAK_CEILING_DBTP:
        warnings.append({"code": "true_peak_over", "level": "error"})
    if duration_difference is not None and duration_difference > 0:
        warnings.append({"code": "window_overrun", "level": "warning"})
    return warnings


def editing_copy(original: Path, output: Path, gain_db=0.0):
    command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(original), "-map", "0:a:0"]
    if gain_db:
        command += ["-af", "volume=%.4fdB" % gain_db]
    result = subprocess.run(command + ["-ac", "1", "-ar", "48000", "-c:a", "pcm_s24le", "-f", "wav", str(output)],
                            capture_output=True, timeout=120)
    if result.returncode:
        raise Failure("audio_conversion_failed", 500)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
