"""Extended teaching signal geometry and unchanged per-pixel Poisson physics."""

from dataclasses import asdict
import json
import math

import numpy as np
import pytest

from simulator.imaging import make_phantom, sample_image
from simulator.patterns import (
    DOG_ROI,
    LEGACY_ROI,
    LINE_PAIR_GROUPS,
    PATTERN_SPECS,
    PATTERN_VERSION,
    make_extended_phantom,
    sample_extended_image,
)
from simulator.physics import ExposureSettings, ModelParameters, calculate_exposure


def test_extended_signal_is_reproducible_and_preserves_legacy_pixels_exactly():
    ideal = make_extended_phantom()
    assert ideal.shape == (512, 512)
    assert ideal.dtype == np.float64
    assert np.isfinite(ideal).all()
    assert 0 < ideal.min() < ideal.max() <= 1
    assert LEGACY_ROI == (0, 256, 0, 256)
    y0, y1, x0, x1 = LEGACY_ROI
    np.testing.assert_array_equal(ideal[y0:y1, x0:x1], make_phantom())
    np.testing.assert_array_equal(ideal, make_extended_phantom())
    ideal[0, 0] = -99
    assert make_extended_phantom()[0, 0] == 1


def test_pattern_specs_identify_versions_shapes_and_pixel_only_dimensions():
    assert set(PATTERN_SPECS) == {"legacy", "extended"}
    assert tuple(PATTERN_SPECS["legacy"]["shape"]) == (256, 256)
    assert tuple(PATTERN_SPECS["extended"]["shape"]) == (512, 512)
    assert PATTERN_VERSION == "extended-pomeranian-linepairs-v1"
    assert PATTERN_SPECS["extended"]["version"] == PATTERN_VERSION
    assert PATTERN_SPECS["legacy"]["version"]
    assert PATTERN_SPECS["legacy"]["label"]
    assert PATTERN_SPECS["extended"]["label"]
    # These metadata are teaching-pixel dimensions, not an inferred mm scale.
    assert [group["line_width_px"] for group in LINE_PAIR_GROUPS] == [8, 6, 4, 3, 2, 1]
    json.dumps(PATTERN_SPECS)
    json.dumps(LINE_PAIR_GROUPS)


@pytest.mark.parametrize("group", LINE_PAIR_GROUPS, ids=lambda group: f"width_{group['line_width_px']}px")
def test_line_and_gap_widths_exactly_match_unblurred_signal_and_metadata(group):
    ideal = make_extended_phantom()
    y0, y1, x0, x1 = group["bbox"]
    width = group["line_width_px"]
    gap = group["gap_width_px"]
    assert width == gap
    assert (y1 - y0, x1 - x0) == (80, 144)
    assert group["dark_signal"] == 0.30
    assert group["bright_signal"] == 0.90
    assert group["periods"] * (width + gap) == x1 - x0
    assert group["cycles_per_pixel"] == pytest.approx(1 / (width + gap))
    unit = np.array([0.30] * width + [0.90] * gap)
    expected_row = np.tile(unit, group["periods"])
    expected = np.tile(expected_row, (y1 - y0, 1))
    np.testing.assert_array_equal(ideal[y0:y1, x0:x1], expected)
    # The finest pair is exactly two samples per cycle; this is a discrete
    # pattern identity, not evidence of a detector's measured MTF.
    if width == 1:
        assert group["cycles_per_pixel"] == 0.5


def test_dog_signal_contains_multiple_gray_levels_and_native_pixel_detail():
    assert DOG_ROI == (0, 256, 256, 512)
    y0, y1, x0, x1 = DOG_ROI
    dog = make_extended_phantom()[y0:y1, x0:x1]
    assert dog.shape == (256, 256)
    assert np.unique(dog).size >= 10
    assert np.count_nonzero((dog > 0.2) & (dog < 0.9)) > 1000
    # A native-resolution target must not be a uniformly repeated 2x2 image.
    # This catches loss of single-pixel structure during premature reduction.
    blocks = dog.reshape(128, 2, 128, 2)
    assert np.count_nonzero(np.ptp(blocks, axis=(1, 3)) > 0) > 100
    # A documented native 1px fur stroke remains between equal mid-gray
    # neighbors; smoothing or downsampling must not erase this teaching target.
    np.testing.assert_array_equal(dog[146, 174:177], [0.74, 0.79, 0.74])


def test_extended_sampling_is_reproducible_and_matches_local_poisson_statistics():
    ideal = make_extended_phantom()
    first = sample_extended_image(20.0, seed=2026)
    assert first.shape == ideal.shape
    assert first.dtype == np.float64
    np.testing.assert_array_equal(first, sample_extended_image(20.0, seed=2026))
    assert not np.array_equal(first, sample_extended_image(20.0, seed=2027))
    np.testing.assert_array_equal(first, sample_extended_image(20.0, seed=2026))
    np.testing.assert_allclose(20 * first, np.rint(20 * first), rtol=0, atol=1e-12)
    assert first.min() >= 0
    for signal in (0.30, 0.90, 1.0):
        values = first[ideal == signal]
        assert values.size > 1000
        standard_error = math.sqrt(signal / (20 * values.size))
        assert values.mean() == pytest.approx(signal, abs=5 * standard_error)
        assert values.var() == pytest.approx(signal / 20, rel=0.08)
        assert values.mean() / values.std() == pytest.approx(math.sqrt(20 * signal), rel=0.05)
    # Sampling remains a linear signal; display clipping is a separate step.
    assert sample_extended_image(0.1, seed=7).max() > 1.2


def test_pattern_size_keeps_the_reference_count_per_pixel_and_physics_unchanged():
    settings, parameters = ExposureSettings(), ModelParameters()
    result = calculate_exposure(settings, parameters)
    before = asdict(result)
    for ideal, sampled in (
        (make_phantom(), sample_image(result.mean_quanta, seed=17)),
        (make_extended_phantom(), sample_extended_image(result.mean_quanta, seed=17)),
    ):
        values = sampled[ideal == 1]
        assert values.mean() == pytest.approx(1, abs=5 / math.sqrt(result.mean_quanta * values.size))
        assert values.std() / values.mean() == pytest.approx(result.relative_noise, rel=0.03)
        assert values.mean() / values.std() == pytest.approx(result.snr, rel=0.03)
    assert asdict(calculate_exposure(settings, parameters)) == before


def test_extended_sampling_keeps_existing_count_and_seed_domain():
    for count in (1e-9, 1e12):
        assert np.isfinite(sample_extended_image(count, seed=42)).all()
    for count in (0, -1, float("nan"), float("inf"), 1e-10, 1e13, True, "100"):
        with pytest.raises(ValueError):
            sample_extended_image(count)
    for seed in (-1, 2**64, 1.5, True):
        with pytest.raises(ValueError):
            sample_extended_image(100, seed=seed)
