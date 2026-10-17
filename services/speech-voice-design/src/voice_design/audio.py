"""Audio validation and attenuation-only reference derivation."""

import hashlib
import io
import math
import struct

import numpy as np
import soundfile as sf

from .assets import digest
from .request import CandidateError


def inspect_reference(data, expected):
    if len(data) != expected["size_bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
        raise CandidateError("reference_integrity_failed")
    if len(data) > 2097152 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise CandidateError("invalid_reference_container")
    if len(data) < 44 or struct.unpack_from("<I", data, 4)[0] + 8 != len(data):
        raise CandidateError("truncated_reference")
    offset, chunks = 12, {}
    while offset + 8 <= len(data):
        kind, size = struct.unpack_from("<4sI", data, offset)
        end = offset + 8 + size
        if end > len(data) or (kind in (b"fmt ", b"data") and kind in chunks):
            raise CandidateError("invalid_reference_chunks")
        chunks[kind] = data[offset + 8:end]
        offset = end + (size % 2)
    if offset != len(data) or b"fmt " not in chunks or b"data" not in chunks:
        raise CandidateError("invalid_reference_chunks")
    info = sf.info(io.BytesIO(data))
    if info.channels != 1 or info.subtype not in ("PCM_16", "PCM_24") or info.samplerate not in (16000, 22050, 24000, 44100, 48000):
        raise CandidateError("unsupported_reference_format")
    sample_bytes = 2 if info.subtype == "PCM_16" else 3
    if len(chunks[b"data"]) != info.frames * sample_bytes:
        raise CandidateError("invalid_reference_samples")
    if not 1 <= info.duration <= 15:
        raise CandidateError("reference_duration_out_of_range")
    wave, _ = sf.read(io.BytesIO(data), dtype="float32")
    if len(wave) != info.frames or not np.isfinite(wave).all() or not np.any(wave):
        raise CandidateError("invalid_reference_signal")
    peak = float(np.max(np.abs(wave)))
    return {
        "sample_rate_hz": info.samplerate, "samples": info.frames, "duration_seconds": info.duration,
        "channels": 1, "subtype": info.subtype, "peak_dbfs": 20 * math.log10(peak),
        "full_scale_samples": int(np.count_nonzero((wave <= -1) | (wave >= 1 - 2 ** (1 - sample_bytes * 8)))),
    }


def write_audio(directory, waveform, rate, cap_may_have_been_reached):
    waveform = np.asarray(waveform)
    if waveform.ndim != 1 or not waveform.size or rate != 48000 or not np.isfinite(waveform).all():
        raise CandidateError("invalid_generated_audio")
    waveform = waveform.astype(np.float32)
    if not np.isfinite(waveform).all():
        raise CandidateError("generated_audio_overflow")
    peak = float(np.max(np.abs(waveform)))
    if peak == 0:
        raise CandidateError("silent_generated_audio")
    duration = len(waveform) / rate
    original = directory / "original.wav"
    sf.write(original, waveform, rate, format="WAV", subtype="FLOAT")
    reasons = []
    if not 1 <= duration <= 15:
        reasons.append("reference_duration_out_of_range")
    if cap_may_have_been_reached:
        reasons.append("generation_cap_may_have_been_reached")
    outputs = {"original": {"path": original.name, "sha256": digest(original), "size_bytes": original.stat().st_size}}
    gain = min(1.0, 10 ** (-3 / 20) / peak)
    if not reasons:
        reference = directory / "reference.wav"
        encoded = io.BytesIO()
        sf.write(encoded, waveform * gain, rate, format="WAV", subtype="PCM_24")
        if len(encoded.getvalue()) > 2097152:
            reasons.append("reference_size_out_of_range")
        else:
            reference.write_bytes(encoded.getvalue())
            outputs["reference"] = {"path": reference.name, "sha256": digest(reference), "size_bytes": reference.stat().st_size}
    quality = {
        "sample_rate_hz": rate, "channels": 1, "samples": len(waveform), "duration_seconds": duration,
        "original_peak_dbfs": 20 * math.log10(peak),
        "original_at_or_above_full_scale_samples": int(np.count_nonzero(np.abs(waveform) >= 1)),
        "reference_gain_db": 20 * math.log10(gain) if "reference" in outputs else None,
        "reference_technically_eligible": not reasons, "issues": reasons,
        "listening_review": "pending", "selection": "pending", "registration": "pending",
        "transcript_verified": False,
    }
    if "reference" in outputs:
        ref_data = (directory / "reference.wav").read_bytes()
        quality["reference_signal"] = inspect_reference(ref_data, outputs["reference"])
    return outputs, quality
