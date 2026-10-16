import unittest

import numpy as np

from speech_runtime.audio import pcm16_with_headroom
from speech_runtime.domain import Failure


class Headroom(unittest.TestCase):
    def test_over_range_peaks_keep_shape_and_chunk_levels(self):
        # Distinct peaks above full scale must remain distinct after conversion.
        source = np.array([[0, 20000, 40000, 60000, -40000, -60000, 1000.]])
        result = pcm16_with_headroom(source)
        self.assertEqual(result.dtype, np.int16)
        self.assertEqual(result.shape, source.shape)
        self.assertLess(np.max(np.abs(result.astype(float))), 23200)
        self.assertLess(result[0, 2], result[0, 3])
        np.testing.assert_allclose(result, source * (32767 * 10 ** (-3 / 20) / 60000), atol=.5)

    def test_quiet_audio_is_not_boosted_or_lengthened(self):
        source = np.array([[0., 100., -100., 23000.]])
        np.testing.assert_array_equal(pcm16_with_headroom(source), source)
        np.testing.assert_array_equal(pcm16_with_headroom(np.zeros((1, 10))), np.zeros((1, 10)))

    def test_invalid_engine_values_fail_before_integer_conversion(self):
        for source in ([], [np.nan], [np.inf], [-np.inf]):
            with self.subTest(source=source), self.assertRaises(Failure):
                pcm16_with_headroom(source)
