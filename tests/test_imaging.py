"""Reproducibility, analytical noise statistics, and fixed-window checks."""

import unittest
import base64
import io

import numpy as np
from PIL import Image

from simulator.imaging import display_image, display_png_data_uri, make_phantom, sample_image


class ImagingTests(unittest.TestCase):
    def test_phantom_is_deterministic_bounded_and_has_a_uniform_reference(self):
        phantom = make_phantom()
        np.testing.assert_array_equal(phantom, make_phantom())
        self.assertEqual(phantom.shape, (256, 256))
        self.assertGreaterEqual(phantom.min(), 0.25)
        self.assertLessEqual(phantom.max(), 1.0)
        np.testing.assert_array_equal(phantom[:20, :20], np.ones((20, 20)))

    def test_seed_reproduces_noise_and_alternative_seed_changes_it(self):
        first = sample_image(100.0, seed=42)
        np.testing.assert_array_equal(first, sample_image(100.0, seed=42))
        self.assertFalse(np.array_equal(first, sample_image(100.0, seed=43)))
        # An intervening different realization must not change a saved case.
        np.testing.assert_array_equal(first, sample_image(100.0, seed=42))

    def test_local_poisson_mean_and_variance_match_theory(self):
        phantom = make_phantom()
        image = sample_image(20.0, seed=2026)
        # The returned image is I = C/N; C itself is a nonnegative count.
        counts = 20.0 * image
        np.testing.assert_allclose(counts, np.rint(counts), rtol=0, atol=1e-12)
        self.assertGreaterEqual(counts.min(), 0)
        for signal in (0.25, 0.72, 1.0):
            samples = image[phantom == signal]
            standard_error = np.sqrt(signal / 20.0 / samples.size)
            self.assertAlmostEqual(samples.mean(), signal, delta=5.0 * standard_error)
            # Sample-variance uncertainty is approximately sqrt(2 / n).
            self.assertAlmostEqual(samples.var(), signal / 20.0, delta=0.08 * signal / 20.0)
            # sqrt(N) applies only to S=1, not to every phantom region.
            expected_snr = np.sqrt(20.0 * signal)
            self.assertAlmostEqual(samples.mean() / samples.std(), expected_snr, delta=0.05 * expected_snr)

    def test_fourfold_counts_halve_relative_reference_noise(self):
        reference = make_phantom() == 1.0
        low_signal = sample_image(100.0, seed=17)[reference]
        high_signal = sample_image(400.0, seed=18)[reference]
        low_cv = low_signal.std() / low_signal.mean()
        high_cv = high_signal.std() / high_signal.mean()
        self.assertAlmostEqual(low_cv, 0.1, delta=0.003)
        self.assertAlmostEqual(high_cv, 0.05, delta=0.002)
        self.assertAlmostEqual(high_cv / low_cv, 0.5, delta=0.025)
        # With normalized brightness held fixed, SD(I) halves while the
        # absolute count SD(C) doubles. These are different quantities.
        self.assertAlmostEqual(high_signal.mean() / low_signal.mean(), 1, delta=0.005)
        self.assertAlmostEqual(high_signal.std() / low_signal.std(), 0.5, delta=0.025)
        low_counts, high_counts = 100.0 * low_signal, 400.0 * high_signal
        self.assertAlmostEqual(high_counts.std() / low_counts.std(), 2, delta=0.10)

    def test_display_uses_fixed_window_and_does_not_modify_raw_signal(self):
        raw = np.array([[-0.1, 0.0, 0.6, 1.2, 1.8]])
        copy = raw.copy()
        np.testing.assert_allclose(display_image(raw), [[0.0, 0.0, 0.5, 1.0, 1.0]])
        np.testing.assert_array_equal(raw, copy)
        np.testing.assert_allclose(display_image(np.array([[0.6, 0.9]])), [[0.5, 0.75]])

    def test_sampling_preserves_unclipped_outliers(self):
        raw = sample_image(0.1, seed=7)
        self.assertGreater(raw.max(), 1.2)
        self.assertLessEqual(display_image(raw).max(), 1.0)

    def test_embedded_png_keeps_fixed_window_and_leaves_raw_signal_unchanged(self):
        raw = np.array([[-0.1, 0.0, 0.6, 1.2, 1.8]])
        original = raw.copy()
        uri = display_png_data_uri(raw)
        self.assertTrue(uri.startswith("data:image/png;base64,"))
        with Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))) as decoded:
            self.assertEqual(decoded.mode, "L")
            np.testing.assert_array_equal(np.asarray(decoded), [[0, 0, 127, 255, 255]])
        np.testing.assert_array_equal(raw, original)

    def test_valid_numerical_limits_are_finite(self):
        for counts in (1e-9, 1e12):
            self.assertTrue(np.isfinite(sample_image(counts, size=32)).all())

    def test_invalid_counts_sizes_seeds_and_windows_are_rejected(self):
        for counts in (0.0, -1.0, float("nan"), float("inf"), 1e13, 1e-10, True, "100"):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                sample_image(counts)
        for size in (0, 31, 1025, 64.0, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                make_phantom(size)
        for seed in (-1, 2**64, 1.5, True):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                sample_image(100, seed=seed)
        for low, high in ((1, 1), (2, 1), (float("nan"), 1), (0, float("inf"))):
            with self.subTest(window=(low, high)), self.assertRaises(ValueError):
                display_image(np.ones((2, 2)), low, high)
        for image in (np.array([]), np.ones(4), np.array([[float("nan")]])):
            with self.subTest(shape=image.shape), self.assertRaises(ValueError):
                display_image(image)


if __name__ == "__main__":
    unittest.main()
