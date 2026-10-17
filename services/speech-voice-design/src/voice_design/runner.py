"""Immutable one-shot candidate execution with explicit partial-failure state."""

import json
from pathlib import Path
import time

from . import __version__
from .assets import digest, verify_model
from .audio import inspect_reference, write_audio
from .engine import MODEL_REVISION, VoxCPMEngine
from .request import CandidateError, generation_kwargs, parse_request


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")


def run(request, reference_path, model_root, lock_path, output, *, engine_factory=VoxCPMEngine):
    model_text = parse_request(request)
    reference_path = Path(reference_path)
    if reference_path.is_symlink() or not reference_path.is_file() or reference_path.stat().st_size > 2097152:
        raise CandidateError("invalid_reference_file")
    reference_bytes = reference_path.read_bytes()
    reference_info = inspect_reference(reference_bytes, request["reference"])
    model_info = verify_model(model_root, lock_path)
    if model_info["revision"] != MODEL_REVISION:
        raise CandidateError("unsupported_model_revision")
    output = Path(output)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise CandidateError("output_already_exists") from exc
    write_json(output / "request.json", request)
    snapshot = output / "source-reference.wav"
    snapshot.write_bytes(reference_bytes)
    write_json(output / "started.json", {"status": "started", "candidate_id": request["candidate_id"], "auto_retry": False})
    start = time.monotonic()
    try:
        engine = engine_factory(model_root)
        kwargs = generation_kwargs(request, model_text, snapshot)
        waveform, rate, execution = engine.generate(kwargs)
        outputs, quality = write_audio(output, waveform, rate, execution["cap_may_have_been_reached"])
        public_kwargs = {k: v for k, v in kwargs.items() if k != "reference_wav_path"}
        provenance = {
            "schema_version": 1, "candidate_id": request["candidate_id"], "runner_version": __version__,
            "mode": request["mode"], "model": model_info, "execution": execution,
            "request_sha256": digest(output / "request.json"),
            "reference": request["reference"], "reference_signal": reference_info,
            "reference_conditioning": "waveform" if request["mode"] == "clone" else "description_only",
            "model_input": public_kwargs, "elapsed_seconds": time.monotonic() - start,
            "upstream_reference_processing": "resample_to_16000_mono_and_trim_long_silences" if request["mode"] == "clone" else "not_consumed",
            "outputs": outputs, "quality": quality,
        }
        write_json(output / "provenance.json", provenance)
        write_json(output / "completed.json", {"status": "candidate_created", "provenance_sha256": digest(output / "provenance.json")})
        return provenance
    except Exception as exc:
        code = str(exc) if isinstance(exc, CandidateError) else "candidate_execution_failed"
        write_json(output / "failed.json", {"status": "failed", "error_code": code, "auto_retry": False})
        raise CandidateError(code) from exc
