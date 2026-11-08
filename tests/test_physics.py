"""Analytic reference tests; clinical validity is deliberately not asserted."""

from dataclasses import FrozenInstanceError, replace
import math
import unittest

from simulator.physics import (
    ExposureSettings, ModelParameters, calculate_exposure,
    deviation_index, exposure_index,
)


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.settings = ExposureSettings()
        self.parameters = ModelParameters()

    def test_iec_calibration_known_value(self):
        self.assertEqual(exposure_index(2.5), 250.0)

    def test_di_target_double_and_half(self):
        self.assertEqual(deviation_index(250, 250), 0.0)
        self.assertAlmostEqual(deviation_index(500, 250), 3.010299956639812, places=12)
        self.assertAlmostEqual(deviation_index(125, 250), -3.010299956639812, places=12)

    def test_known_default_solution(self):
        result = calculate_exposure(self.settings, self.parameters)
        # Independently evaluated reference: 50 × 5 × exp(-0.18 × 15).
        self.assertAlmostEqual(result.detector_kerma_ugy, 16.80137818493744, places=11)
        self.assertAlmostEqual(result.ei, 1680.137818493744, places=9)
        self.assertAlmostEqual(result.effective_mu_per_cm, 0.18, places=15)
        self.assertAlmostEqual(result.transmission, math.exp(-2.7), places=15)

    def test_mas_double_scales_kerma_ei_noise(self):
        baseline = calculate_exposure(self.settings, self.parameters)
        changed = calculate_exposure(replace(self.settings, mas=10), self.parameters)
        self.assertAlmostEqual(changed.detector_kerma_ugy / baseline.detector_kerma_ugy, 2)
        self.assertAlmostEqual(changed.ei / baseline.ei, 2)
        self.assertAlmostEqual(changed.di - baseline.di, 10 * math.log10(2))
        self.assertAlmostEqual(changed.relative_noise / baseline.relative_noise, 1 / math.sqrt(2))

    def test_mas_quadruple_halves_relative_noise(self):
        baseline = calculate_exposure(self.settings, self.parameters)
        changed = calculate_exposure(replace(self.settings, mas=20), self.parameters)
        self.assertAlmostEqual(changed.relative_noise / baseline.relative_noise, 0.5)
        self.assertAlmostEqual(changed.snr / baseline.snr, 2)

    def test_inverse_square_distance(self):
        baseline = calculate_exposure(self.settings, self.parameters)
        changed = calculate_exposure(replace(self.settings, sid_cm=200), self.parameters)
        self.assertAlmostEqual(changed.detector_kerma_ugy / baseline.detector_kerma_ugy, 0.25)
        self.assertAlmostEqual(changed.relative_noise / baseline.relative_noise, 2)

    def test_half_value_thickness(self):
        baseline = calculate_exposure(replace(self.settings, thickness_cm=0), self.parameters)
        half_value = math.log(2) / self.parameters.mu_ref_per_cm
        changed = calculate_exposure(replace(self.settings, thickness_cm=half_value), self.parameters)
        self.assertAlmostEqual(changed.detector_kerma_ugy / baseline.detector_kerma_ugy, 0.5)

    def test_open_beam_at_reference(self):
        result = calculate_exposure(replace(self.settings, thickness_cm=0), self.parameters)
        self.assertAlmostEqual(result.detector_kerma_ugy, 250)
        self.assertEqual(result.transmission, 1)

    def test_kvp_power_without_attenuation(self):
        baseline = calculate_exposure(replace(self.settings, kvp=60, thickness_cm=0), self.parameters)
        changed = calculate_exposure(replace(self.settings, kvp=120, thickness_cm=0), self.parameters)
        self.assertAlmostEqual(changed.detector_kerma_ugy / baseline.detector_kerma_ugy, 4)
        self.assertAlmostEqual(changed.effective_mu_per_cm / baseline.effective_mu_per_cm, 0.5)

    def test_kvp_change_with_thickness_combines_output_and_transmission(self):
        low = calculate_exposure(replace(self.settings, kvp=60), self.parameters)
        high = calculate_exposure(replace(self.settings, kvp=120), self.parameters)
        # At 15 cm, mu changes from 0.24 to 0.12 cm^-1. The output
        # factor is 4, but transmission also increases by exp(1.8).
        # This is a reference for the stated hypothetical model, not real kVp response.
        expected_ratio = 4 * math.exp(1.8)
        self.assertAlmostEqual(high.detector_kerma_ugy / low.detector_kerma_ugy, expected_ratio)
        self.assertAlmostEqual(high.ei / low.ei, expected_ratio)
        self.assertAlmostEqual(high.di - low.di, 10 * math.log10(expected_ratio))
        self.assertAlmostEqual(high.relative_noise / low.relative_noise, 1 / math.sqrt(expected_ratio))

    def test_zero_attenuation(self):
        result = calculate_exposure(self.settings, replace(self.parameters, mu_ref_per_cm=0))
        self.assertEqual(result.transmission, 1)
        self.assertAlmostEqual(result.detector_kerma_ugy, 250)

    def test_zero_mu_kvp_exponent_keeps_attenuation_constant(self):
        parameters = replace(self.parameters, mu_kvp_exponent=0)
        low = calculate_exposure(replace(self.settings, kvp=40), parameters)
        high = calculate_exposure(replace(self.settings, kvp=120), parameters)
        self.assertEqual(low.effective_mu_per_cm, self.parameters.mu_ref_per_cm)
        self.assertEqual(low.effective_mu_per_cm, high.effective_mu_per_cm)
        self.assertEqual(low.transmission, high.transmission)
        self.assertAlmostEqual(high.detector_kerma_ugy / low.detector_kerma_ugy, 9)

    def test_grid_transmission_scales_kerma(self):
        baseline = calculate_exposure(self.settings, self.parameters)
        changed = calculate_exposure(self.settings, replace(self.parameters, grid_transmission=0.5))
        self.assertAlmostEqual(changed.detector_kerma_ugy / baseline.detector_kerma_ugy, 0.5)

    def test_target_changes_only_di(self):
        baseline = calculate_exposure(self.settings, self.parameters)
        changed = calculate_exposure(self.settings, replace(self.parameters, target_ei=500))
        self.assertEqual(changed.detector_kerma_ugy, baseline.detector_kerma_ugy)
        self.assertEqual(changed.ei, baseline.ei)
        self.assertEqual(changed.relative_noise, baseline.relative_noise)
        self.assertAlmostEqual(changed.di - baseline.di, -10 * math.log10(2))

    def test_quanta_gain_changes_noise_not_kerma_or_ei(self):
        baseline = calculate_exposure(self.settings, self.parameters)
        changed = calculate_exposure(self.settings, replace(self.parameters, quanta_per_ugy=4000))
        self.assertEqual(changed.ei, baseline.ei)
        self.assertAlmostEqual(changed.relative_noise / baseline.relative_noise, 0.5)
        self.assertAlmostEqual(changed.mean_quanta / baseline.mean_quanta, 4)

    def test_invalid_settings(self):
        for name in ("kvp", "mas", "sid_cm"):
            for value in (0, -1, math.nan, math.inf, -math.inf, True, "80"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    replace(self.settings, **{name: value})
        for value in (-1, math.nan, math.inf, True):
            with self.subTest(thickness=value), self.assertRaises(ValueError):
                replace(self.settings, thickness_cm=value)
        with self.assertRaises(ValueError):
            replace(self.settings, thickness_cm=100)

    def test_invalid_parameters(self):
        for name in (
            "output_ugy_per_mas_at_reference", "reference_kvp", "reference_sid_cm",
            "kvp_exponent", "target_ei", "quanta_per_ugy",
            "grid_transmission",
        ):
            for value in (0, -1, math.nan, math.inf):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    replace(self.parameters, **{name: value})
        for value in (-1, math.nan, math.inf):
            with self.subTest(mu=value), self.assertRaises(ValueError):
                replace(self.parameters, mu_ref_per_cm=value)
            with self.subTest(p=value), self.assertRaises(ValueError):
                replace(self.parameters, mu_kvp_exponent=value)
        with self.assertRaises(ValueError):
            replace(self.parameters, grid_transmission=1.01)

    def test_invalid_ei_di_inputs(self):
        for value in (0, -1, math.nan, math.inf, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    exposure_index(value)
                with self.assertRaises(ValueError):
                    deviation_index(value, 250)
                with self.assertRaises(ValueError):
                    deviation_index(250, value)

    def test_numeric_extremes_fail_explicitly(self):
        with self.assertRaises(ValueError):
            exposure_index(1e308)
        with self.assertRaises(ValueError):
            calculate_exposure(replace(self.settings, mas=1e308), self.parameters)
        with self.assertRaises(ValueError):
            calculate_exposure(self.settings, replace(self.parameters, mu_ref_per_cm=1e308))
        # Log-difference remains finite even if EI/EIT cannot be represented.
        self.assertAlmostEqual(deviation_index(1e300, 1e-300), 6000)

    def test_ui_domain_corners_are_finite(self):
        for kvp in (40, 120):
            for mas in (0.1, 20):
                for sid in (50, 200):
                    for thickness in (0, 35):
                        result = calculate_exposure(ExposureSettings(kvp, mas, sid, thickness))
                        for value in vars(result).values():
                            self.assertTrue(math.isfinite(value))
                        self.assertGreater(result.detector_kerma_ugy, 0)

    def test_settings_and_parameters_are_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            self.settings.mas = 10
        with self.assertRaises(FrozenInstanceError):
            self.parameters.target_ei = 500


if __name__ == "__main__":
    unittest.main()
