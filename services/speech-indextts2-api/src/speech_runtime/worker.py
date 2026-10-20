import contextlib
import json
import os
import threading
import time
from pathlib import Path

from .audio import TRUE_PEAK_CEILING_DBTP, digest_file, editing_copy, inspect_wav, quality_warnings
from .domain import Failure, canonical
from .store import atomic_write


class IndexEngine:
    def __init__(self, model_root):
        self.root = Path(model_root)
        self.model = None

    def load(self):
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

    def generate(self, reference, inputs, output):
        import torch
        options = {"use_random": False, "verbose": False}
        if "emotion_vector" in inputs:
            options["emo_vector"] = inputs["emotion_vector"]
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
                "device_name": torch.cuda.get_device_name(0)}


class Worker:
    def __init__(self, store, engine, converter=editing_copy):
        self.store, self.engine, self.converter = store, engine, converter
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.ready = False
        self.error = None
        self.thread = None

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

    def execute(self, row):
        task_id = row["id"]
        root = self.store.output_dir(task_id)
        inputs = json.loads(row["input_json"])
        reference = self.store.root / "references" / row["scope"] / (inputs["reference_sha256"] + ".wav")
        started = time.monotonic()
        try:
            root.mkdir(parents=True, exist_ok=False)
            if digest_file(reference) != inputs["reference_sha256"]:
                raise Failure("stored_reference_corrupt", 409)
            original = root / "original.wav"
            compute = self.engine.generate(reference, inputs, original)
            # Bound before reading, including corrupt/errant engine outputs.
            if original.stat().st_size > 32 * 1024 * 1024:
                raise Failure("output_too_large", 500)
            native = inspect_wav(original.read_bytes())
            editing = root / "editing.wav"
            self.converter(original, editing)
            if editing.stat().st_size > 32 * 1024 * 1024:
                raise Failure("output_too_large", 500)
            derived = inspect_wav(editing.read_bytes())
            gain_db = 0.0
            # Fixed attenuation only; no compression or limiting that would alter the performance.
            if derived["true_peak_dbtp"] > TRUE_PEAK_CEILING_DBTP:
                gain_db = round(TRUE_PEAK_CEILING_DBTP - 0.1 - derived["true_peak_dbtp"], 2)
                editing.unlink()  # ffmpeg with -nostdin refuses to overwrite.
                self.converter(original, editing, gain_db=gain_db)
                derived = inspect_wav(editing.read_bytes())
            if derived["sample_rate_hz"] != 48000 or derived["bit_depth"] != 24:
                raise Failure("invalid_editing_format", 500)
            if abs(native["duration_seconds"] - derived["duration_seconds"]) > 0.001:
                raise Failure("audio_duration_changed", 500)
            report = {"task_id": task_id, "model_manifest_sha256": row["model_hash"],
                      "input_sha256": row["input_hash"], "reference_sha256": inputs["reference_sha256"],
                      "original": native, "editing": derived, "conversion": "resample_to_48000_pcm_s24le_mono",
                      "elapsed_seconds": time.monotonic() - started, "compute": compute,
                      "creative_review": "not_reviewed", "exact_duration_control": False,
                      "editing_gain_db": gain_db}
            if "target_duration_seconds" in inputs:
                report["duration_difference_seconds"] = native["duration_seconds"] - inputs["target_duration_seconds"]
            report["warnings"] = quality_warnings(native, report.get("duration_difference_seconds"))
            atomic_write(root / "report.json", canonical(report).encode())
            outputs = []
            for name in ("original.wav", "editing.wav", "report.json"):
                path = root / name
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

    def shutdown(self):
        self.stop.set()
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=5)
        # A process termination during a GPU call is recovered from persistent state.
        return self.thread is None or not self.thread.is_alive()
