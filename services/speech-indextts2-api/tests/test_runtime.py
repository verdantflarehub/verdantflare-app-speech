import concurrent.futures
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from speech_runtime.audio import inspect_wav
from speech_runtime.domain import Failure, Scope, new_id, validate_create
from speech_runtime.models import verify_models, safe_target
from speech_runtime.server import create_app
from speech_runtime.store import Store
from speech_runtime.worker import Worker

MODEL = "a" * 64
TOKEN = "test-only-runtime-credential-123456789"


def wav(seconds=2, rate=22050, width=16):
    buffer = io.BytesIO()
    samples = 0.15 * np.sin(np.arange(int(rate * seconds)) * 2 * np.pi * 220 / rate)
    sf.write(buffer, samples, rate, format="WAV", subtype="PCM_" + str(width))
    return buffer.getvalue()


class Fixture(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name).resolve()
        self.scope = Scope(new_id(), new_id(), new_id())
        self.reference = wav()
        self.hash = hashlib.sha256(self.reference).hexdigest()
        self.inputs = {"idempotency_key": "utterance-take-001", "text": "这是一句新的台词。", "language": "zh-CN", "reference_sha256": self.hash}

    def tearDown(self):
        self.directory.cleanup()

    def assertFailure(self, code, fn):
        with self.assertRaises(Failure) as error:
            fn()
        self.assertEqual(error.exception.code, code)


class Inputs(Fixture):
    def test_reject_unimplemented_controls_and_bad_emotion(self):
        for change in ({"exact_duration": 3}, {"reference_url": "http://localhost/private"}, {"text": "  "},
                       {"emotion_vector": [0.2] * 8}, {"emotion_vector": [True] * 8}, {"language": "xx"}):
            with self.subTest(change=change), self.assertRaises(Failure):
                validate_create(self.inputs | change)

    def test_reference_duration_format_and_silence(self):
        self.assertFailure("reference_duration_or_size", lambda: inspect_wav(wav(16), reference=True))
        b = io.BytesIO()
        sf.write(b, np.zeros(22050), 22050, format="WAV", subtype="PCM_16")
        self.assertFailure("silent_audio", lambda: inspect_wav(b.getvalue(), reference=True))
        report = inspect_wav(wav(2, 48000, 24), reference=True)
        self.assertEqual((report["bit_depth"], report["sample_rate_hz"]), (24, 48000))


class DurableQueue(Fixture):
    def setUp(self):
        super().setUp()
        self.store = Store(self.root, MODEL)
        self.store.put_reference(self.scope, self.hash, self.reference)

    def tearDown(self):
        self.store.close()
        super().tearDown()

    def test_idempotency_conflict_and_simultaneous_submission(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            a, b = list(pool.map(lambda _: self.store.create(self.scope, self.inputs), range(2)))
        self.assertEqual(a["task_id"], b["task_id"])
        self.assertFailure("idempotency_conflict", lambda: self.store.create(self.scope, self.inputs | {"text": "另一句话。"}))
        self.assertEqual(self.store.claim()["id"], a["task_id"])
        self.assertIsNone(self.store.claim())

    def test_project_and_subject_isolation(self):
        task = self.store.create(self.scope, self.inputs)
        for scope in (Scope(self.scope.organization_id, self.scope.subject_id, new_id()),
                      Scope(self.scope.organization_id, new_id(), self.scope.project_id)):
            self.assertFailure("task_not_found", lambda: self.store.get(scope, task["task_id"]))
            self.assertFailure("reference_not_found", lambda: self.store.create(scope, self.inputs))

    def test_checksum_failure_does_not_register_reference(self):
        bad = "b" * 64
        self.assertFailure("reference_hash_mismatch", lambda: self.store.put_reference(self.scope, bad, self.reference))
        self.assertFalse(self.store.reference_path(self.scope, bad).exists())

    def test_restart_does_not_repeat_uncertain_inference(self):
        task = self.store.create(self.scope, self.inputs)
        self.store.claim()
        self.store.close()
        self.store = Store(self.root, MODEL)
        recovered = self.store.get(self.scope, task["task_id"])
        self.assertEqual(recovered["status"], "failed")
        self.assertTrue(recovered["error"]["recovery_required"])
        self.assertEqual(self.store.create(self.scope, self.inputs)["task_id"], task["task_id"])
        self.assertIsNone(self.store.claim())

    def test_pending_task_cannot_silently_switch_model(self):
        task = self.store.create(self.scope, self.inputs)
        self.store.close()
        self.store = Store(self.root, "b" * 64)
        self.assertIsNone(self.store.claim())
        self.assertEqual(self.store.get(self.scope, task["task_id"])["error"]["code"], "model_version_changed")

    def test_process_lock_prevents_two_gpu_consumers(self):
        with self.assertRaisesRegex(RuntimeError, "runtime_already_active"):
            Store(self.root, MODEL)

    def test_completed_outputs_recover_after_commit_interruption(self):
        task = self.store.create(self.scope, self.inputs)
        row = self.store.claim()
        class Engine:
            def generate(inner, reference, inputs, output):
                output.write_bytes(wav())
                return {"test_engine": True}
        def convert(original, output):
            output.write_bytes(wav(rate=48000, width=24))
        worker = Worker(self.store, Engine(), convert)
        original_finish = self.store.finish
        self.store.finish = lambda _: None  # Simulate death after the durable completion marker.
        worker.execute(row)
        self.store.finish = original_finish
        self.store.close()
        self.store = Store(self.root, MODEL)
        result = self.store.get(self.scope, task["task_id"])
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(len(result["outputs"]), 3)
        (self.store.output_dir(task["task_id"]) / "original.wav").write_bytes(b"corruption")
        self.assertFailure("output_integrity_failed", lambda: self.store.get(self.scope, task["task_id"]))


class InternalAPI(Fixture):
    def setUp(self):
        super().setUp()
        self.app = create_app(root=self.root, token=TOKEN, model_hash=MODEL, start_worker=False)
        self.headers = {"Authorization": "Bearer " + TOKEN, "X-Organization-Id": self.scope.organization_id,
                        "X-User-Id": self.scope.subject_id, "X-Project-Id": self.scope.project_id, "X-Request-Id": new_id()}
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.app.state.store.close()
        super().tearDown()

    def test_authentication_and_no_model_readiness(self):
        self.assertEqual(self.client.get("/live").status_code, 200)
        self.assertEqual(self.client.get("/health").status_code, 503)
        self.assertEqual(self.client.get("/v1/capabilities").status_code, 401)
        self.assertEqual(self.client.get("/v1/capabilities", headers={"Authorization": "Bearer " + TOKEN}).status_code, 403)
        caps = self.client.get("/v1/capabilities", headers=self.headers).json()
        self.assertFalse(caps["ready"])
        self.assertFalse(caps["exact_duration"])
        response = self.client.post("/v1/tasks", json=self.inputs, headers=self.headers)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "model_not_ready")

    def test_upload_create_replay_and_scope_query(self):
        response = self.client.put("/v1/references/" + self.hash, content=self.reference,
                                   headers=self.headers | {"Content-Type": "audio/wav"})
        self.assertEqual(response.status_code, 200)
        self.app.state.worker.ready = True
        result = self.client.post("/v1/tasks", json=self.inputs, headers=self.headers)
        self.assertEqual(result.status_code, 202)
        task_id = result.json()["task_id"]
        self.app.state.worker.ready = False
        replay = self.client.post("/v1/tasks", json=self.inputs, headers=self.headers)
        self.assertEqual(replay.json()["task_id"], task_id)
        wrong = self.headers | {"X-Project-Id": new_id()}
        self.assertEqual(self.client.get("/v1/tasks/" + task_id, headers=wrong).status_code, 404)
        self.assertEqual(self.client.get("/v1/tasks/" + task_id + "/outputs/original.wav", headers=self.headers).status_code, 409)

    def test_duplicate_json_and_encoded_upload_rejected(self):
        response = self.client.post("/v1/tasks", content=b'{"text":"one","text":"two"}',
                                    headers=self.headers | {"Content-Type": "application/json"})
        self.assertEqual(response.json()["error"]["code"], "duplicate_json_key")
        response = self.client.put("/v1/references/" + self.hash, content=self.reference,
                                   headers=self.headers | {"Content-Type": "audio/wav", "Content-Encoding": "gzip"})
        self.assertEqual(response.status_code, 400)


class ModelIntegrity(Fixture):
    def test_fixed_manifest_and_file_integrity(self):
        payload = b"small-test-model"
        manifest = {"files": [{"path": "model/test.bin", "sha256": hashlib.sha256(payload).hexdigest(),
                                "size": len(payload), "url": "https://huggingface.co/test/model/resolve/fixed/test.bin"}]}
        source = self.root / "source.json"
        source.write_text(json.dumps(manifest))
        root = self.root / "installed"
        (root / "model").mkdir(parents=True)
        (root / "manifest.lock.json").write_bytes(source.read_bytes())
        (root / "model/test.bin").write_bytes(payload)
        self.assertEqual(verify_models(root, source)["files"], 1)
        (root / "model/test.bin").write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "model_integrity_failed"):
            verify_models(root, source)
        with self.assertRaisesRegex(ValueError, "invalid_model_path"):
            safe_target(root, "../outside")


if __name__ == "__main__":
    unittest.main()
