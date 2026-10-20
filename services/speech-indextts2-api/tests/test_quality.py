import io
import math
import unittest

import numpy as np
import soundfile as sf

from speech_runtime.audio import inspect_wav, integrated_lufs, quality_warnings, silence_edges, true_peak_dbtp


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
        contract = json.loads((Path(__file__).parents[1] / "src/speech_runtime/contract.json").read_text())
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


if __name__ == "__main__":
    unittest.main()
