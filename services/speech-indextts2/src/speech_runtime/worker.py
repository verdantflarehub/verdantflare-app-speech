import contextlib
import gc
import inspect
import json
import os
import threading
import time
from pathlib import Path

from .asr import WhisperSmallASR
from .audio import digest_file, editing_copy, inspect_wav, normalize_editing_wav, quality_warnings
from .domain import Failure, canonical
from .gpu_guard import FencePersistenceError, StaleFenceError
from .store import atomic_write


def _normalize_resolved_vector(values):
    cleaned = [max(0.0, min(0.8, float(x))) for x in values]
    total = sum(cleaned)
    if total > 0.8 and total > 0:
        scale = 0.8 / total
        cleaned = [x * scale for x in cleaned]
    rounded = [int(round(x * 10000)) / 10000.0 for x in cleaned]
    while sum(rounded) > 0.800000001:
        idx = max(range(len(rounded)), key=lambda i: rounded[i])
        rounded[idx] = round(max(0.0, rounded[idx] - 0.0001), 4)
    return rounded


class IndexEngine:
    def __init__(self, model_root, asr_root=None):
        self.root = Path(model_root)
        self.model = None
        self.asr = WhisperSmallASR(asr_root)
        self.load_seconds = 0.0

    def load(self):
        started = time.monotonic()
        from .models import verify_models
        verify_models(self.root)
        os.environ["SPEECH_MODEL_ROOT"] = str(self.root)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["NLTK_DATA"] = str(self.root / "nltk_data")
        from .probe import gpu_probe
        gpu_probe()
        from indextts.infer_v2 import IndexTTS2
        self.model = IndexTTS2(cfg_path=str(self.root / "indextts2/config.yaml"),
                               model_dir=str(self.root / "indextts2"), device="cuda:0",
                               use_fp16=True, use_cuda_kernel=False, use_deepspeed=False,
                               use_accel=False, use_torch_compile=False)
        self.asr.load()
        self.load_seconds = round(time.monotonic() - started, 4)

    def generate(self, reference, inputs, output, emo_reference=None):
        import torch
        options = {"use_random": False, "verbose": False}
        resolved_emotion_vector = None
        if "emotion_vector" in inputs:
            resolved_emotion_vector = _normalize_resolved_vector(inputs["emotion_vector"])
            options["emo_vector"] = resolved_emotion_vector
        elif "emotion_text" in inputs:
            with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), torch.inference_mode():
                raw_dict = self.model.qwen_emo.inference(inputs["emotion_text"])
                norm_vec = self.model.normalize_emo_vec(list(raw_dict.values()), apply_bias=True)
            resolved_emotion_vector = _normalize_resolved_vector(norm_vec)
            options["emo_vector"] = resolved_emotion_vector
        elif emo_reference is not None and "emotion_reference" in inputs:
            options["emo_audio_prompt"] = str(emo_reference)
            options["emo_alpha"] = float(inputs["emotion_reference"]["alpha"])

        torch.cuda.reset_peak_memory_stats()
        # Upstream sometimes prints text or internal filenames even with verbose=False.
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), torch.inference_mode():
            result = self.model.infer(spk_audio_prompt=str(reference), text=inputs["text"],
                                      output_path=str(output), **options)
        if result is None or not output.is_file():
            raise Failure("engine_returned_no_audio", 500)
        torch.cuda.synchronize()
        return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(0),
                "resolved_emotion_vector": resolved_emotion_vector}

    def unload(self):
        self.model = None
        if hasattr(self.asr, "model"):
            self.asr.model = None
        if hasattr(self.asr, "ready"):
            self.asr.ready = False
        gc.collect()
        with contextlib.suppress(Exception):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class Worker:
    def __init__(self, store, engine, converter=editing_copy, fence=None):
        self.store, self.engine, self.converter = store, engine, converter
        self.fence = fence
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.ready = False
        self.error = None
        self.thread = None
        self._gpu_lock = threading.Lock()

    def start(self):
        self.thread = threading.Thread(target=self.run, name="speech-gpu-worker", daemon=True)
        self.thread.start()

    def run(self):
        try:
            self.engine.load()
            self.ready = True
        except Exception:
            self.error = "model_load_failed"
            return
        try:
            while not self.stop.is_set():
                task = self.store.claim()
                if task is None:
                    self.wake.wait(1)
                    self.wake.clear()
                    continue
                self.execute(task)
        except Exception:
            self.error = "worker_failed"
        finally:
            self.ready = False

    def _call_generate(self, reference, inputs, original, emo_reference):
        params = inspect.signature(self.engine.generate).parameters
        if len(params) >= 4 or any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values()):
            return self.engine.generate(reference, inputs, original, emo_reference)
        return self.engine.generate(reference, inputs, original)

    def execute(self, row):
        task_id = row["id"]
        root = self.store.output_dir(task_id)
        inputs = json.loads(row["input_json"])
        reference = self.store.root / "references" / row["scope"] / (inputs["reference_sha256"] + ".wav")
        emo_reference = None
        if "emotion_reference" in inputs:
            emo_hash = inputs["emotion_reference"]["reference_sha256"]
            emo_reference = self.store.root / "references" / row["scope"] / (emo_hash + ".wav")
        started = time.monotonic()
        try:
            root.mkdir(parents=True, exist_ok=False)
            if digest_file(reference) != inputs["reference_sha256"]:
                raise Failure("stored_reference_corrupt", 409)
            if emo_reference is not None and digest_file(emo_reference) != inputs["emotion_reference"]["reference_sha256"]:
                raise Failure("stored_reference_corrupt", 409)
            if self.fence is not None:
                try:
                    self.fence.verify_before_inference(task_id)
                except FencePersistenceError as exc:
                    raise Failure("fence_state_corrupted", 503) from exc
                except StaleFenceError as exc:
                    raise Failure("stale_fencing_token", 409) from exc
            original = root / "original.wav"
            with self._gpu_lock:
                if not self.ready and hasattr(self.engine, "load"):
                    self.engine.load()
                    self.ready = True
                compute = dict(self._call_generate(reference, inputs, original, emo_reference) or {})
            resolved_emotion = compute.pop(
                "resolved_emotion_vector",
                _normalize_resolved_vector(inputs["emotion_vector"]) if "emotion_vector" in inputs else None,
            )
            # Bound before reading, including corrupt/errant engine outputs.
            if original.stat().st_size > 32 * 1024 * 1024:
                raise Failure("output_too_large", 500)
            native = inspect_wav(original.read_bytes())
            editing = root / "editing.wav"
            self.converter(original, editing)
            derived = normalize_editing_wav(editing)
            gain_db = derived["applied_gain_db"]
            if derived["sample_rate_hz"] != 48000 or derived["bit_depth"] != 24:
                raise Failure("invalid_editing_format", 500)
            if abs(native["duration_seconds"] - derived["duration_seconds"]) > 0.001:
                raise Failure("audio_duration_changed", 500)

            asr = getattr(self.engine, "asr", None)
            if asr is not None:
                asr_out = asr.check_audio(original, inputs["text"], source_text=inputs.get("expected_text"))
            else:
                asr_out = {
                    "transcript_check": {"state": "unavailable", "reason": "asr_model_not_loaded"},
                    "asr_metrics": None,
                }
            if asr_out.get("asr_metrics") is not None:
                compute["asr"] = asr_out["asr_metrics"]

            report = {"task_id": task_id, "model_manifest_sha256": row["model_hash"],
                      "input_sha256": row["input_hash"], "reference_sha256": inputs["reference_sha256"],
                      "original": native, "editing": derived, "conversion": "resample_to_48000_pcm_s24le_mono",
                      "elapsed_seconds": time.monotonic() - started, "compute": compute,
                      "creative_review": "not_reviewed", "exact_duration_control": False,
                      "editing_gain_db": gain_db,
                      "resolved_emotion_vector": resolved_emotion,
                      "transcript_check": asr_out["transcript_check"]}
            window = inputs.get("shot_window_seconds", inputs.get("target_duration_seconds"))
            if "shot_window_seconds" in inputs:
                report["shot_window_seconds"] = inputs["shot_window_seconds"]
            if window is not None:
                report["duration_difference_seconds"] = round(native["duration_seconds"] - window, 4)
            report["warnings"] = quality_warnings(
                native,
                report.get("duration_difference_seconds"),
                report["transcript_check"],
            )
            atomic_write(root / "report.json", canonical(report).encode())
            outputs = []
            for name in ("original.wav", "editing.wav", "report.json"):
                path = root / name
                if os.name != "nt":
                    with path.open("rb") as f:
                        os.fsync(f.fileno())
                outputs.append({"name": name, "sha256": digest_file(path), "size": path.stat().st_size,
                                "mime": "application/json" if name.endswith(".json") else "audio/wav"})
            atomic_write(root / "complete.json", canonical({"task_id": task_id, "model_manifest_sha256": row["model_hash"], "outputs": outputs}).encode())
            self.store.finish(task_id)
        except Failure as error:
            self.store.fail(task_id, error.code)
        except Exception:
            self.store.fail(task_id, "inference_failed")
        finally:
            if self.fence is not None and hasattr(self.fence, "seal_task"):
                self.fence.seal_task(task_id)

    def evict_idle_vram(self) -> bool:
        if not self._gpu_lock.acquire(blocking=False):
            return False
        try:
            with self.store.mutex:
                if self.store.active_count() > 0:
                    return False
                if self.fence is not None and hasattr(self.fence, "active_inflight_count") and self.fence.active_inflight_count() > 0:
                    return False
            if hasattr(self.engine, "unload"):
                self.engine.unload()
            self.ready = False
            return True
        finally:
            self._gpu_lock.release()

    def shutdown(self):
        self.stop.set()
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=5)
        # A process termination during a GPU call is recovered from persistent state.
        return self.thread is None or not self.thread.is_alive()
