import io
import math
import unittest

import numpy as np
import soundfile as sf

from speech_runtime.asr import WhisperSmallASR, compare_transcript
from speech_runtime.audio import inspect_wav, integrated_lufs, quality_warnings, silence_edges, true_peak_dbtp
from speech_runtime.domain import Failure, validate_create


def sine(freq, seconds, rate, amplitude, phase=0.0):
    t = np.arange(int(rate * seconds)) / rate
    return amplitude * np.sin(2 * np.pi * freq * t + phase)


class Loudness(unittest.TestCase):
    def test_reference_tone_reads_bs1770_level(self):
        # BS.1770: a 1 kHz sine at -20 dBFS on one channel reads about -23.0 LUFS.
        for rate in (22050, 48000):
            with self.subTest(rate=rate):
                self.assertAlmostEqual(integrated_lufs(sine(1000, 3, rate, 0.1), rate), -23.01, delta=0.3)

    def test_gating_ignores_silence_and_floors_quiet_audio(self):
        rate = 48000
        tone = sine(1000, 2, rate, 0.1)
        padded = np.concatenate([np.zeros(rate * 2), tone, np.zeros(rate * 2)])
        # Edge blocks partly overlapping the tone still pass the relative gate, as in BS.1770 meters.
        gated, ungated = integrated_lufs(padded, rate), integrated_lufs(tone, rate) + 10 * math.log10(2 / 6)
        self.assertAlmostEqual(gated, integrated_lufs(tone, rate), delta=1.0)
        self.assertGreater(gated, ungated + 3)
        self.assertEqual(integrated_lufs(sine(1000, 1, rate, 1e-5), rate), -70.0)

    def test_utterance_shorter_than_one_block_is_measured(self):
        self.assertAlmostEqual(integrated_lufs(sine(1000, 0.3, 48000, 0.1), 48000), -23.0, delta=0.6)


class TruePeak(unittest.TestCase):
    def test_inter_sample_peak_exceeds_sample_peak(self):
        # A quarter-rate sine sampled 45 degrees off its crest hides 3 dB between samples.
        samples = sine(48000 / 4, 1, 48000, 0.5, phase=math.pi / 4)
        sample_peak = 20 * math.log10(np.abs(samples).max())
        measured = true_peak_dbtp(samples)
        self.assertAlmostEqual(sample_peak, -9.03, delta=0.05)
        self.assertAlmostEqual(measured, -6.02, delta=0.3)

    def test_never_below_sample_peak(self):
        samples = sine(440, 1, 22050, 0.25)
        self.assertGreaterEqual(true_peak_dbtp(samples), 20 * math.log10(np.abs(samples).max()) - 1e-9)


class Silence(unittest.TestCase):
    def test_edges(self):
        rate = 22050
        x = np.concatenate([np.zeros(int(rate * 0.5)), sine(300, 1, rate, 0.2), np.zeros(int(rate * 0.25))])
        leading, trailing = silence_edges(x, rate)
        self.assertAlmostEqual(leading, 0.5, delta=0.011)
        self.assertAlmostEqual(trailing, 0.25, delta=0.011)

    def test_no_silence(self):
        self.assertEqual(silence_edges(sine(300, 1, 48000, 0.2), 48000), (0.0, 0.0))


class Performance(unittest.TestCase):
    def test_longest_output_is_measured_quickly(self):
        import time
        rate, x = 48000, sine(220, 180, 48000, 0.1)
        started = time.monotonic()
        true_peak_dbtp(x), integrated_lufs(x, rate), silence_edges(x, rate)
        self.assertLess(time.monotonic() - started, 20)


class Report(unittest.TestCase):
    def test_inspect_wav_reports_new_metrics(self):
        buffer = io.BytesIO()
        sf.write(buffer, sine(1000, 2, 48000, 0.1), 48000, format="WAV", subtype="PCM_24")
        report = inspect_wav(buffer.getvalue())
        for key in ("true_peak_dbtp", "integrated_lufs", "leading_silence_seconds", "trailing_silence_seconds"):
            self.assertIn(key, report)
        self.assertAlmostEqual(report["integrated_lufs"], -23.0, delta=0.3)
        self.assertAlmostEqual(report["true_peak_dbtp"], -20.0, delta=0.2)

    def test_report_matches_runtime_contract(self):
        import json
        from pathlib import Path
        from jsonschema import Draft202012Validator
        import speech_runtime.audio as audio
        contract = json.loads(Path(audio.__file__).with_name("contract.json").read_text())
        schema = dict(contract, **{"$ref": "#/$defs/audio_report"})
        schema.pop("oneOf", None)
        buffer = io.BytesIO()
        sf.write(buffer, sine(1000, 2, 22050, 0.1), 22050, format="WAV", subtype="PCM_16")
        Draft202012Validator(schema).validate(inspect_wav(buffer.getvalue(), reference=True))

    def test_warnings_are_advisory_and_specific(self):
        clean = {"clipped_samples": 0, "true_peak_dbtp": -3.0}
        self.assertEqual(quality_warnings(clean), [])
        self.assertEqual(quality_warnings(clean, -0.2), [])
        hot = {"clipped_samples": 37, "true_peak_dbtp": 0.6}
        self.assertEqual([w["code"] for w in quality_warnings(hot, 0.3)], ["clipping", "true_peak_over", "window_overrun"])
        self.assertEqual({w["level"] for w in quality_warnings(hot)}, {"error"})
        mismatch = {"state": "checked", "cer": 8.3}
        self.assertEqual(
            [w["code"] for w in quality_warnings(hot, 0.3, mismatch)],
            ["clipping", "true_peak_over", "transcript_mismatch", "window_overrun"],
        )
        self.assertEqual(quality_warnings(clean, None, {"state": "unavailable", "reason": "asr_model_not_loaded"}), [])


class TranscriptCheck(unittest.TestCase):
    def test_exact_match_and_punctuation_invariance(self):
        res = compare_transcript("娥儿，船上风大，把披风系紧些。", "娥儿, 船上风大! 把披风系紧些")
        self.assertEqual(res["cer"], 0.0)
        self.assertEqual(res["differences"], [])

    def test_substituted_missing_and_extra_positions(self):
        sub = compare_transcript("封住渡口，一个不许放走。", "封住路口，一个不许放走。")
        self.assertGreater(sub["cer"], 0)
        self.assertEqual(sub["differences"], [{"kind": "substituted", "index": 2, "expected": "渡", "heard": "路"}])

        miss = compare_transcript("快走！别回头！", "快走！回头！")
        self.assertEqual(miss["differences"], [{"kind": "missing", "index": 3, "expected": "别"}])

        extra = compare_transcript("快走！", "快快走！")
        self.assertEqual(len(extra["differences"]), 1)
        self.assertEqual(extra["differences"][0]["kind"], "extra")
        self.assertEqual(extra["differences"][0]["heard"], "快")

    def test_lexicon_pinyin_uses_source_text_fallback(self):
        res = compare_transcript(
            "娥儿，船上风大，把披风ji4紧些。",
            "娥儿，船上风大，把披风系紧些。",
            source_text="娥儿，船上风大，把披风系紧些。",
        )
        self.assertEqual(res["cer"], 0.0)
        self.assertEqual(res["differences"], [])

    def test_unavailable_asr_reports_state_unavailable(self):
        asr = WhisperSmallASR("/nonexistent-asr-dir")
        self.assertFalse(asr.load())
        out = asr.check_audio("/nonexistent.wav", "测试")
        self.assertEqual(out["transcript_check"]["state"], "unavailable")

    def test_mutually_exclusive_emotion_and_shot_window_inputs(self):
        base = {
            "idempotency_key": "E01-S03-L01.take3",
            "text": "娥儿，船上风大。",
            "language": "zh-CN",
            "reference_sha256": "a" * 64,
        }
        self.assertEqual(validate_create(base | {"emotion_text": "温和叮嘱", "shot_window_seconds": 3.5})["emotion_text"], "温和叮嘱")
        self.assertEqual(
            validate_create(base | {"emotion_reference": {"reference_sha256": "b" * 64, "alpha": 0.6}})["emotion_reference"]["alpha"],
            0.6,
        )
        with self.assertRaises(Failure):
            validate_create(base | {"emotion_vector": [0] * 8, "emotion_text": "平静"})
        with self.assertRaises(Failure):
            validate_create(base | {"emotion_text": "平静", "emotion_reference": {"reference_sha256": "b" * 64, "alpha": 0.5}})


if __name__ == "__main__":
    unittest.main()
