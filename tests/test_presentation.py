"""Display invariance, residual scaling, clipping, and pixel magnification."""

import base64
from dataclasses import FrozenInstanceError
import io

import numpy as np
from PIL import Image
import pytest

from simulator.imaging import display_image, display_png_data_uri, make_phantom, sample_image
from simulator.presentation import (
    ENHANCEMENT_FACTORS,
    ZOOM_FACTORS,
    amplify_noise,
    prepare_display,
)


def decode_preview(uri):
    assert uri.startswith("data:image/png;base64,")
    with Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1], validate=True))) as image:
        assert image.format == "PNG"
        assert image.mode == "L"
        return np.asarray(image).copy()


def test_factor_one_is_an_exact_independent_copy_including_dtype():
    raw = np.array([[1e-12, 0.25], [0.72, 1.0]], dtype=np.float32)
    ideal = np.full_like(raw, 1e10)
    preview_signal = amplify_noise(raw, ideal, 1)
    np.testing.assert_array_equal(preview_signal, raw)
    assert preview_signal.dtype == raw.dtype
    assert not np.shares_memory(preview_signal, raw)
    preview_signal[0, 0] = 99
    assert raw[0, 0] == np.float32(1e-12)


def test_amplification_has_known_unclipped_values_and_never_mutates_inputs():
    raw = np.array([[0.6, 0.8], [0.4, 0.2]])
    ideal = np.full((2, 2), 0.5)
    before_raw, before_ideal = raw.copy(), ideal.copy()
    amplified = amplify_noise(raw, ideal, 3)
    np.testing.assert_allclose(amplified, [[0.8, 1.4], [0.2, -0.4]], atol=1e-15)
    prepare_display(raw, ideal, factor=5, zoom=4)
    np.testing.assert_array_equal(raw, before_raw)
    np.testing.assert_array_equal(ideal, before_ideal)
    assert not np.shares_memory(amplified, raw)
    assert not np.shares_memory(amplified, ideal)


@pytest.mark.parametrize("factor", [2, 3, 5])
def test_residual_amplitude_and_standard_deviation_scale_before_clipping(factor):
    ideal = make_phantom()
    raw = sample_image(100, seed=17)
    residual = raw - ideal
    amplified_residual = amplify_noise(raw, ideal, factor) - ideal
    np.testing.assert_allclose(amplified_residual, factor * residual, rtol=1e-13, atol=1e-15)
    assert amplified_residual.std() == pytest.approx(factor * residual.std(), rel=1e-13)
    assert amplified_residual.var() == pytest.approx(factor**2 * residual.var(), rel=1e-13)


def test_standard_preview_is_identical_to_existing_png_display():
    raw = sample_image(100, seed=17)
    result = prepare_display(raw, make_phantom())
    assert result.uri == display_png_data_uri(raw)
    assert result.enhancement_factor == result.zoom == 1
    assert result.source_shape == (256, 256)
    assert result.crop_bounds == (0, 256, 0, 256)
    assert decode_preview(result.uri).dtype == np.uint8
    with pytest.raises(FrozenInstanceError):
        result.zoom = 2


def test_clipping_fraction_counts_only_values_strictly_outside_fixed_window():
    raw = np.array([[-0.1, 0.0, 0.6, 1.2, 1.3]])
    result = prepare_display(raw, np.zeros_like(raw))
    assert result.clipping_fraction == 2 / 5
    np.testing.assert_array_equal(decode_preview(result.uri), [[0, 0, 127, 255, 255]])
    # Display conversion uses float64, so a float32 approximation above the
    # exact 1.2 endpoint must count as clipped as well as being displayed white.
    above_endpoint = np.array([[1.2]], dtype=np.float32)
    assert prepare_display(above_endpoint, above_endpoint).clipping_fraction == 1.0


def test_clipping_makes_displayed_amplitude_nonlinear():
    ideal = np.array([[0.6, 0.6]])
    raw = np.array([[0.3, 0.9]])
    amplified = amplify_noise(raw, ideal, 3)
    np.testing.assert_allclose(amplified, [[-0.3, 1.5]], atol=1e-15)
    # Without clipping, the displayed residual would be ±0.75. The fixed
    # display bounds saturate it at ±0.5; it no longer has threefold amplitude.
    np.testing.assert_allclose(display_image(amplified) - display_image(ideal), [[-0.5, 0.5]])
    result = prepare_display(raw, ideal, factor=3)
    assert result.clipping_fraction == 1.0
    np.testing.assert_array_equal(decode_preview(result.uri), [[0, 255]])


@pytest.mark.parametrize("zoom, bounds", [(2, (2, 6, 3, 9)), (4, (3, 5, 4, 7))])
def test_zoom_repeats_central_roi_pixels_at_the_original_display_size(zoom, bounds):
    raw = np.linspace(-0.15, 1.35, 96).reshape(8, 12)
    ideal = np.full_like(raw, 0.6)
    standard = prepare_display(raw, ideal, factor=2)
    magnified = prepare_display(raw, ideal, factor=2, zoom=zoom)
    original_pixels = decode_preview(standard.uri)
    zoomed_pixels = decode_preview(magnified.uri)
    assert magnified.crop_bounds == bounds
    y0, y1, x0, x1 = bounds
    assert zoomed_pixels.shape == raw.shape
    for row_offset in range(zoom):
        for column_offset in range(zoom):
            np.testing.assert_array_equal(
                zoomed_pixels[row_offset::zoom, column_offset::zoom],
                original_pixels[y0:y1, x0:x1],
            )
    assert magnified.source_shape == standard.source_shape == raw.shape
    assert magnified.enhancement_factor == standard.enhancement_factor == 2
    assert magnified.zoom == zoom


def test_clipping_fraction_is_measured_within_the_shown_roi_only():
    raw = np.full((8, 8), 0.6)
    raw[0, 0], raw[1, 1], raw[2, 2], raw[3, 3] = -0.1, 1.3, 1.3, 1.3
    raw[3, 4], raw[4, 3] = 1.2, 0.0
    ideal = np.full_like(raw, 0.6)
    assert prepare_display(raw, ideal, zoom=1).clipping_fraction == 4 / 64
    assert prepare_display(raw, ideal, zoom=2).clipping_fraction == 2 / 16
    magnified = prepare_display(raw, ideal, zoom=4)
    assert magnified.clipping_fraction == 1 / 4
    assert magnified.crop_bounds == (3, 5, 3, 5)
    pixels = decode_preview(magnified.uri)
    np.testing.assert_array_equal(pixels[::4, ::4], [[255, 255], [0, 127]])


def test_non_multiple_sizes_keep_original_shape_with_defined_central_crop():
    raw = np.arange(15, dtype=float).reshape(3, 5) / 15
    ideal = np.full_like(raw, 0.5)
    standard = decode_preview(prepare_display(raw, ideal).uri)
    zoomed = prepare_display(raw, ideal, zoom=2)
    assert zoomed.crop_bounds == (1, 2, 1, 3)
    first, second = standard[1, 1], standard[1, 2]
    expected_row = [first, first, second, second, second]
    np.testing.assert_array_equal(decode_preview(zoomed.uri), [expected_row] * 3)


def test_only_explicit_integer_enhancement_and_zoom_choices_are_accepted():
    raw = np.ones((2, 2))
    assert ENHANCEMENT_FACTORS == (1, 2, 3, 5)
    assert ZOOM_FACTORS == (1, 2, 4)
    assert prepare_display(raw, raw, factor=np.int64(2), zoom=np.int64(4)).zoom == 4
    for factor in (0, -1, 4, 6, 1.0, 2.5, True, np.bool_(True), "2", None, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            amplify_noise(raw, raw, factor)
        with pytest.raises(ValueError):
            prepare_display(raw, raw, factor=factor)
    for zoom in (0, -1, 3, 5, 1.0, 2.5, True, np.bool_(True), "2", None, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            prepare_display(raw, raw, zoom=zoom)


def test_mismatched_empty_nonfinite_and_non_numeric_arrays_are_rejected():
    good = np.ones((2, 2))
    bad_arrays = (
        np.ones((1, 2)), np.array([]), np.zeros((0, 2)), np.ones(4), np.ones((2, 2, 1)),
        np.array([[float("nan"), 1], [1, 1]]), np.full((2, 2), float("inf")),
        np.full((2, 2), True), np.full((2, 2), "1"), np.full((2, 2), 1 + 0j),
        np.full((2, 2), 1, dtype=object), [[1], [1, 2]],
    )
    for bad in bad_arrays:
        for raw, ideal in ((bad, good), (good, bad)):
            with pytest.raises(ValueError):
                amplify_noise(raw, ideal, 1)
            with pytest.raises(ValueError):
                prepare_display(raw, ideal)


def test_arithmetic_overflow_is_rejected_without_clipping_or_changing_raw_data():
    raw = np.array([[1e308]])
    ideal = np.array([[-1e308]])
    with pytest.raises(ValueError, match="finite float64"):
        amplify_noise(raw, ideal, 5)
    with pytest.raises(ValueError, match="finite float64"):
        prepare_display(raw, ideal, factor=5)
    np.testing.assert_array_equal(raw, [[1e308]])
    np.testing.assert_array_equal(amplify_noise(raw, ideal, 1), raw)
