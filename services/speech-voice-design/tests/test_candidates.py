import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from voice_design.assets import prepare_model, verify_model
from voice_design.engine import MODEL_REVISION, SOURCE_COMMIT, VoxCPMEngine, verify_source
from voice_design.request import CandidateError, parse_request
from voice_design.runner import run


class FakeEngine:
    calls = []
    waveform = None
    capped = False

    def __init__(self, root):
        pass

    def generate(self, kwargs):
        self.calls.append(kwargs)
        wave = self.waveform if self.waveform is not None else np.sin(np.arange(96000) / 20) * 0.2
        return wave, 48000, {"cap_may_have_been_reached": self.capped, "actual_seed": kwargs["seed"]}


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = self.root / "source.wav"
        sf.write(self.reference, np.sin(np.arange(48000) / 10) * 0.1, 48000, subtype="PCM_16")
        data = self.reference.read_bytes()
        self.request = {"schema_version": 1, "candidate_id": "test-voice", "mode": "clone",
                        "text": "下一段旅程即将开始。", "seed": 42,
                        "reference": {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}}
        self.models = self.root / "models"
        self.models.mkdir()
        self.config = self.models / "config.json"
        self.config.write_text('{"architecture":"voxcpm2"}')
        self.lock = self.root / "lock.json"
        self.lock.write_text(json.dumps({"repository": "openbmb/VoxCPM2", "revision": MODEL_REVISION,
            "files": [{"path": "config.json", "size": self.config.stat().st_size,
                       "sha256": hashlib.sha256(self.config.read_bytes()).hexdigest(), "url": "https://example.invalid/model"}]}))
        self.out = self.root / "out"
        FakeEngine.calls, FakeEngine.waveform, FakeEngine.capped = [], None, False

    def execute(self, engine=FakeEngine):
        return run(self.request, self.reference, self.models, self.lock, self.out, engine_factory=engine)

    def design(self):
        self.request.update(mode="design", reference_traits="沉稳、低沉", change_request="更年轻、更清亮",
                            target_description="年轻男性，清亮而沉稳的声音")

    def test_clone_uses_verified_snapshot_once_without_retries(self):
        self.request["style_instruction"] = "平静地讲述"
        report = self.execute()
        self.assertEqual(len(FakeEngine.calls), 1)
        call = FakeEngine.calls[0]
        self.assertEqual(Path(call["reference_wav_path"]).read_bytes(), self.reference.read_bytes())
        self.assertFalse(call["retry_badcase"])
        self.assertEqual(call["retry_badcase_max_times"], 1)
        self.assertFalse(call["normalize"])
        self.assertFalse(call["denoise"])
        self.assertEqual(report["reference_conditioning"], "waveform")
        self.assertTrue(report["quality"]["reference_technically_eligible"])
        self.assertEqual(report["quality"]["listening_review"], "pending")
        self.assertNotIn(str(self.root), (self.out / "provenance.json").read_text())

    def test_design_records_reference_but_does_not_condition_on_audio(self):
        self.design()
        report = self.execute()
        self.assertNotIn("reference_wav_path", FakeEngine.calls[0])
        self.assertEqual(report["reference_conditioning"], "description_only")
        self.assertEqual(report["reference"], self.request["reference"])
        self.assertTrue(FakeEngine.calls[0]["text"].startswith("(年轻男性，清亮而沉稳的声音)"))

    def test_invalid_mode_combinations_rejected_before_generation(self):
        for updates in ({"target_description": "年轻声音"}, {"mode": "design"}, {"seed": True},
                        {"speaker_age": 20}, {"style_instruction": "开心(重复台词)"},
                        {"text": "正文<|ref_start|>"}, {"text": "正文\u200b"}):
            request = copy.deepcopy(self.request)
            request.update(updates)
            with self.subTest(updates=updates), self.assertRaises(CandidateError):
                parse_request(request)
        self.assertEqual(FakeEngine.calls, [])

    def test_original_text_and_actual_whitespace_handling_preserved(self):
        self.request["text"] = "下一段\n\t旅程即将开始。"
        report = self.execute()
        self.assertEqual(json.loads((self.out / "request.json").read_text())["text"], self.request["text"])
        self.assertEqual(report["model_input"]["text"], "下一段 旅程即将开始。")

    def test_hash_mismatch_refuses_execution(self):
        self.reference.write_bytes(self.reference.read_bytes()[:-2])
        with self.assertRaisesRegex(CandidateError, "reference_integrity_failed"):
            self.execute()
        self.assertFalse(self.out.exists())
        self.assertFalse(FakeEngine.calls)

    def test_malformed_wav_rejected_even_with_matching_hash(self):
        data = self.reference.read_bytes()[:-2]
        self.reference.write_bytes(data)
        self.request["reference"] = {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
        with self.assertRaisesRegex(CandidateError, "truncated_reference"):
            self.execute()

    def test_unverified_fallback_weights_and_symlinks_rejected(self):
        extra = self.models / "audiovae.safetensors"
        extra.write_bytes(b"unverified")
        with self.assertRaisesRegex(CandidateError, "unexpected_model_asset"):
            self.execute()
        extra.unlink()
        self.config.unlink()
        self.config.symlink_to(self.lock)
        with self.assertRaisesRegex(CandidateError, "asset_symlink"):
            self.execute()
        self.assertFalse(FakeEngine.calls)

    def test_output_never_overwritten_or_reexecuted(self):
        self.execute()
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.out.iterdir()}
        with self.assertRaisesRegex(CandidateError, "output_already_exists"):
            self.execute()
        self.assertEqual(len(FakeEngine.calls), 1)
        self.assertEqual(hashes, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.out.iterdir()})

    def test_signal_over_full_scale_preserved_and_reference_only_attenuated(self):
        FakeEngine.waveform = np.sin(np.arange(96000) / 20) * 1.5
        report = self.execute()
        original, _ = sf.read(self.out / "original.wav")
        reference, _ = sf.read(self.out / "reference.wav")
        self.assertGreater(np.max(np.abs(original)), 1)
        self.assertLessEqual(np.max(np.abs(reference)), 10 ** (-3 / 20) + 2 ** -23)
        self.assertGreater(report["quality"]["original_at_or_above_full_scale_samples"], 0)
        self.assertEqual(len(reference), len(original))
        self.assertLess(report["quality"]["reference_gain_db"], 0)

    def test_long_candidate_is_preserved_without_clipping_or_padding(self):
        FakeEngine.waveform = np.ones(48000 * 16) * 0.1
        report = self.execute()
        self.assertFalse(report["quality"]["reference_technically_eligible"])
        self.assertEqual(sf.info(self.out / "original.wav").frames, 48000 * 16)
        self.assertFalse((self.out / "reference.wav").exists())

    def test_fifteen_second_pcm24_over_byte_limit_is_not_registered(self):
        FakeEngine.waveform = np.ones(48000 * 15) * 0.1
        report = self.execute()
        self.assertIn("reference_size_out_of_range", report["quality"]["issues"])
        self.assertFalse((self.out / "reference.wav").exists())
        self.assertTrue((self.out / "completed.json").exists())

    def test_possible_token_cap_does_not_create_ready_reference(self):
        FakeEngine.capped = True
        report = self.execute()
        self.assertIn("generation_cap_may_have_been_reached", report["quality"]["issues"])
        self.assertFalse((self.out / "reference.wav").exists())

    def test_failed_inference_has_no_retry_and_no_private_exception_text(self):
        class Broken(FakeEngine):
            def generate(self, kwargs):
                self.calls.append(kwargs)
                raise RuntimeError("private path or upstream request detail")
        with self.assertRaisesRegex(CandidateError, "candidate_execution_failed"):
            self.execute(Broken)
        self.assertEqual(len(FakeEngine.calls), 1)
        self.assertNotIn("private", (self.out / "failed.json").read_text())
        self.assertFalse((self.out / "completed.json").exists())

    def test_nonfinite_audio_is_failed_not_registered(self):
        FakeEngine.waveform = np.array([float("nan"), 0.1])
        with self.assertRaisesRegex(CandidateError, "invalid_generated_audio"):
            self.execute()
        self.assertFalse((self.out / "reference.wav").exists())

    def test_verified_existing_model_is_reused_without_network(self):
        with patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
            result = prepare_model(self.models, self.lock)
        self.assertEqual(result["revision"], MODEL_REVISION)

    def test_differing_existing_model_is_never_replaced(self):
        self.config.write_text("different")
        with patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
            with self.assertRaisesRegex(CandidateError, "existing_model_asset_mismatch"):
                prepare_model(self.models, self.lock)
        self.assertEqual(self.config.read_text(), "different")

    def test_zero_length_partial_download_can_resume(self):
        data = self.config.read_bytes()
        self.config.unlink()
        (self.models / "config.json.part").write_bytes(b"")
        with patch("urllib.request.urlopen", return_value=io.BytesIO(data)):
            prepare_model(self.models, self.lock)
        self.assertEqual(self.config.read_bytes(), data)

    def test_ignored_range_request_preserves_partial_bytes(self):
        data = self.config.read_bytes()
        self.config.unlink()
        part = self.models / "config.json.part"
        part.write_bytes(data[:5])
        response = io.BytesIO(data)
        response.status = 200
        response.headers = {}
        with patch("urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(CandidateError, "model_resume_not_supported"):
                prepare_model(self.models, self.lock)
        self.assertEqual(part.read_bytes(), data[:5])

    def test_fixed_official_modelscope_source_preserves_content_identity(self):
        data = self.config.read_bytes()
        self.config.unlink()
        lock = json.loads(self.lock.read_text())
        lock["files"][0]["modelscope_revision"] = "a" * 40
        self.lock.write_text(json.dumps(lock))
        with patch("urllib.request.urlopen", return_value=io.BytesIO(data)) as fetch:
            result = prepare_model(self.models, self.lock)
        request = fetch.call_args.args[0]
        self.assertEqual(request.full_url, "https://www.modelscope.cn/models/OpenBMB/VoxCPM2/resolve/" + "a" * 40 + "/config.json")
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(self.config.read_bytes(), data)
        self.assertEqual(result["revision"], MODEL_REVISION)

    def test_moving_modelscope_branch_is_rejected_before_network(self):
        lock = json.loads(self.lock.read_text())
        lock["files"][0]["modelscope_revision"] = "master"
        self.lock.write_text(json.dumps(lock))
        with patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
            with self.assertRaisesRegex(CandidateError, "invalid_modelscope_revision"):
                prepare_model(self.models, self.lock)

    def test_source_receipt_rejects_modified_or_added_code(self):
        source = self.root / "src"
        source.mkdir()
        code = source / "a.py"
        code.write_text("pass\n")
        receipt = self.root / "receipt.json"
        receipt.write_text(json.dumps({"commit": SOURCE_COMMIT, "files": {"a.py": hashlib.sha256(code.read_bytes()).hexdigest()}}))
        verify_source(source, receipt)
        code.write_text("raise Exception()\n")
        with self.assertRaisesRegex(CandidateError, "upstream_source_integrity_failed"):
            verify_source(source, receipt)
        (source / "extra.py").write_text("pass\n")
        with self.assertRaisesRegex(CandidateError, "upstream_files_mismatch"):
            verify_source(source, receipt)

    def test_adapter_loads_only_local_cuda_model_without_hidden_warmup(self):
        calls = []
        fake = types.ModuleType("voxcpm")
        fake.__file__ = "/opt/voxcpm/src/voxcpm/__init__.py"
        class Loader:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                calls.append((args, kwargs))
                return object()
        fake.VoxCPM = Loader
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(reset_peak_memory_stats=lambda: None)
        with patch.dict("sys.modules", {"torch": torch, "voxcpm": fake}), \
             patch("voice_design.engine.verify_source"), patch("voice_design.engine.gpu_probe", return_value={}):
            VoxCPMEngine(self.models)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], {"load_denoiser": False, "local_files_only": True,
                                      "optimize": False, "device": "cuda"})


if __name__ == "__main__":
    unittest.main()
