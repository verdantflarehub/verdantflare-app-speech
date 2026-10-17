"""The sole GPU adapter; imports are lazy so preparation/tests remain lightweight."""

import contextlib
import importlib.metadata
import os
from pathlib import Path

from .assets import digest, load_lock
from .request import CandidateError

SOURCE_COMMIT = "cce58a5b59303c9bd63d12f23afe6a49a4a80c59"
MODEL_REVISION = "32279effe8c19989596f05d353d1447f51d9e915"
MIN_FREE_BYTES = 12 * 1024**3


def gpu_probe():
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise CandidateError("one_cuda_device_required")
    free, total = torch.cuda.mem_get_info(0)
    if free < MIN_FREE_BYTES:
        raise CandidateError("insufficient_gpu_memory")
    properties = torch.cuda.get_device_properties(0)
    if not hasattr(properties, "uuid"):
        raise CandidateError("gpu_identity_unavailable")
    sample = torch.ones((16, 16), device="cuda")
    result = sample @ sample
    torch.cuda.synchronize()
    if not torch.all(result == 16).item():
        raise CandidateError("gpu_operator_probe_failed")
    del sample, result
    return {"name": properties.name, "uuid": str(properties.uuid), "free_bytes": free,
            "total_bytes": total, "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda}


def verify_source(root, receipt_path):
    root = Path(root)
    receipt = load_lock(receipt_path)
    if receipt["commit"] != SOURCE_COMMIT:
        raise CandidateError("upstream_commit_mismatch")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*.py")}
    if actual != set(receipt["files"]):
        raise CandidateError("upstream_files_mismatch")
    for relative, expected in receipt["files"].items():
        path = root / relative
        if path.is_symlink() or root.resolve() not in path.resolve().parents or digest(path) != expected:
            raise CandidateError("upstream_source_integrity_failed")


class VoxCPMEngine:
    def __init__(self, model_root):
        # These paths are supplied only by the built image, never by candidate requests.
        verify_source("/opt/voxcpm/src", "/opt/service/source-receipt.json")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        import voxcpm
        from voxcpm import VoxCPM

        if Path(voxcpm.__file__).resolve() != Path("/opt/voxcpm/src/voxcpm/__init__.py"):
            raise CandidateError("unexpected_upstream_import")
        self.gpu = gpu_probe()
        torch.cuda.reset_peak_memory_stats()
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            self.model = VoxCPM.from_pretrained(
                str(model_root), load_denoiser=False, local_files_only=True,
                optimize=False, device="cuda",
            )

    def generate(self, kwargs):
        import torch

        model = self.model.tts_model
        limit = min(int(len(model.text_tokenizer(kwargs["text"])) * 6 + 10), kwargs["max_len"])
        max_samples = limit * model.patch_size * model._decode_chunk_size
        # The adapter never retries or rewrites text; logs from upstream may contain paths.
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            with torch.inference_mode():
                waveform = self.model.generate(**kwargs)
        torch.cuda.synchronize()
        return waveform, model.sample_rate, {
            "source_commit": SOURCE_COMMIT, "gpu": self.gpu,
            "effective_max_tokens": limit,
            "cap_may_have_been_reached": len(waveform) >= max_samples,
            "actual_seed": getattr(model, "last_successful_seed", None),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "packages": {name: importlib.metadata.version(name) for name in (
                "torch", "torchaudio", "transformers", "numpy", "soundfile", "librosa")},
        }
