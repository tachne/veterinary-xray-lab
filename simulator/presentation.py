"""Display-only noise amplification and pixel-preserving magnification.

Neither operation changes the exposure model, raw synthetic signals, EI, DI,
or theoretical noise/SNR. Amplified previews are teaching displays, not new
detector measurements. Their residuals are multiplied before a shared fixed
window is applied; clipping and 8-bit quantization are nonlinear afterwards.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import io
import numbers

import numpy as np
from PIL import Image

from simulator.imaging import display_image


ENHANCEMENT_FACTORS = (1, 2, 3, 5)
ZOOM_FACTORS = (1, 2, 4)


@dataclass(frozen=True)
class DisplayPreview:
    """An encoded display and its display-only settings and clipping fraction.

    ``clipping_fraction`` counts pixels below 0 or above 1.2 within the shown
    central ROI before display clipping. Boundary values are not counted.
    ``source_shape`` remains the original (rows, columns); ``crop_bounds`` is
    the half-open source ROI (y0, y1, x0, x1), before nearest-neighbor zoom.
    """

    uri: str
    clipping_fraction: float
    enhancement_factor: int
    zoom: int
    source_shape: tuple[int, int]
    crop_bounds: tuple[int, int, int, int]


def _validate_choice(value: int, choices: tuple[int, ...], name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be one of the integers {choices}")
    if value not in choices:
        raise ValueError(f"{name} must be one of the integers {choices}")
    return int(value)


def _signal_array(value: np.ndarray, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite, non-empty, real numeric 2D array") from None
    if (
        array.ndim != 2
        or array.size == 0
        or array.dtype.kind not in "fiu"
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be a finite, non-empty, real numeric 2D array")
    return array


def amplify_noise(raw: np.ndarray, ideal: np.ndarray, factor: int) -> np.ndarray:
    """Return J = ideal + factor * (raw - ideal), without clipping or mutation.

    A factor of 1 returns an independent, exactly equal copy, including dtype,
    to avoid cancellation in the algebraically equivalent expression. For
    other factors, calculations use float64; overflowing results are rejected.
    Accepted factors are the integers 1, 2, 3, and 5, excluding booleans.
    """
    factor = _validate_choice(factor, ENHANCEMENT_FACTORS, "factor")
    raw_array = _signal_array(raw, "raw")
    ideal_array = _signal_array(ideal, "ideal")
    if raw_array.shape != ideal_array.shape:
        raise ValueError("raw and ideal must have the same shape")
    if factor == 1:
        return raw_array.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        raw_values = raw_array.astype(np.float64)
        ideal_values = ideal_array.astype(np.float64)
        amplified = ideal_values + factor * (raw_values - ideal_values)
    if not np.isfinite(amplified).all():
        raise ValueError("noise amplification exceeded the finite float64 range")
    return amplified


def prepare_display(
    raw: np.ndarray,
    ideal: np.ndarray,
    factor: int = 1,
    zoom: int = 1,
) -> DisplayPreview:
    """Encode a central-ROI display with a shared 0–1.2 window and zoom.

    Order: residual amplification, central ROI selection, fixed-window
    clipping, multiplication by 255 and uint8 truncation, nearest-neighbor
    enlargement back to the source dimensions, then PNG encoding.
    This preserves the standard display's quantization at factor=zoom=1.
    The PNG data URI avoids a transient HTTP request for the small preview.
    At zoom=1 the whole image is displayed. For zoom=2/4 the ROI dimensions
    are floor(source dimensions / zoom), with a minimum of one pixel; a
    one-pixel centering remainder is left on the bottom/right. Source sizes
    divisible by zoom, including 256x256, give exact integer magnification.
    Otherwise nearest-neighbor resizing retains the original display shape.
    Clipping is reported within this ROI, not the entire source image.
    No smoothing, adaptive windowing, or numerical measurement is applied.
    """
    zoom = _validate_choice(zoom, ZOOM_FACTORS, "zoom")
    amplified = amplify_noise(raw, ideal, factor)
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.asarray(amplified, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("display signal exceeded the finite float64 range")
    rows, columns = values.shape
    crop_rows, crop_columns = max(1, rows // zoom), max(1, columns // zoom)
    y0, x0 = (rows - crop_rows) // 2, (columns - crop_columns) // 2
    y1, x1 = y0 + crop_rows, x0 + crop_columns
    roi = values[y0:y1, x0:x1]
    clipping_fraction = float(np.count_nonzero((roi < 0.0) | (roi > 1.2)) / roi.size)
    pixels = (255 * display_image(roi, low=0.0, high=1.2)).astype(np.uint8)
    preview_image = Image.fromarray(pixels)
    if zoom > 1:
        preview_image = preview_image.resize((columns, rows), resample=Image.Resampling.NEAREST)
    output = io.BytesIO()
    preview_image.save(output, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")
    return DisplayPreview(
        uri=uri,
        clipping_fraction=clipping_fraction,
        enhancement_factor=int(factor),
        zoom=zoom,
        source_shape=(int(values.shape[0]), int(values.shape[1])),
        crop_bounds=(y0, y1, x0, x1),
    )
