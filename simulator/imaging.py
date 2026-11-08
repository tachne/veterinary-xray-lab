"""Geometric teaching phantom with ideal, independent Poisson quantum noise.

These arrays are synthetic signals, not measured radiographs or a calibrated
detector model. No detector blur, scatter, electronic noise, image processing,
anatomical contrast, DQE, or MTF is simulated. Signal 1.0 corresponds to the
reference count ``mean_quanta``; local counts scale with the phantom signal.
"""

from __future__ import annotations

import math
import numbers
import base64
import io

import numpy as np
from PIL import Image


def _validate_size(size: int) -> None:
    if isinstance(size, bool) or not isinstance(size, numbers.Integral):
        raise ValueError("size must be an integer between 32 and 1024")
    if not 32 <= size <= 1024:
        raise ValueError("size must be an integer between 32 and 1024")


def make_phantom(size: int = 256) -> np.ndarray:
    """Return a fixed geometric phantom with dimensionless signals 0.25–1.0.

    The outer margin is a homogeneous 1.0 reference field. Disks, bars, and
    low-contrast squares are visual teaching targets, not an MTF test object.
    Pixel size has no physical unit in this phase of the application.
    """
    _validate_size(size)
    coordinates = (np.arange(size, dtype=np.float64) + 0.5) / size
    x, y = np.meshgrid(coordinates, coordinates)
    phantom = np.ones((size, size), dtype=np.float64)
    field = (x > 0.12) & (x < 0.88) & (y > 0.12) & (y < 0.88)
    phantom[field] = 0.72

    for center_x, radius, signal in (
        (0.29, 0.105, 0.25),
        (0.53, 0.085, 0.50),
        (0.75, 0.065, 0.85),
    ):
        disk = (x - center_x) ** 2 + (y - 0.33) ** 2 <= radius**2
        phantom[disk] = signal

    for left, signal in ((0.18, 0.67), (0.43, 0.70), (0.68, 0.75)):
        patch = (x >= left) & (x < left + 0.14) & (y >= 0.50) & (y < 0.61)
        phantom[patch] = signal

    for left, width, period in ((0.18, 0.17, 0.06), (0.42, 0.17, 0.03), (0.66, 0.17, 0.015)):
        region = (x >= left) & (x < left + width) & (y >= 0.68) & (y < 0.81)
        dark_bars = np.remainder(x - left, period) < period / 2
        phantom[region & dark_bars] = 0.36
    return phantom


def sample_image(mean_quanta: float, seed: int = 42, size: int = 256) -> np.ndarray:
    """Sample an un-clipped linear signal image at a fixed exposure scale.

    Each pixel is Poisson(N * phantom) / N, where N = mean_quanta.
    Thus E[signal] = phantom and Var[signal] = phantom / N. Relative noise
    is 1/sqrt(N) in the outer reference field; it is 1/sqrt(N * phantom)
    elsewhere. Changing N leaves the expected displayed signal unchanged.

    Bounds on N and image size prevent excessive memory use and numerical
    overflow; they are software limits, not validated detector operating limits.
    A fixed seed gives reproducible examples, not identical noise realizations
    at different exposures.
    """
    if isinstance(mean_quanta, bool) or not isinstance(mean_quanta, numbers.Real):
        raise ValueError("mean_quanta must be a finite number in [1e-9, 1e12]")
    try:
        count = float(mean_quanta)
    except (ValueError, OverflowError):
        raise ValueError("mean_quanta must be a finite number in [1e-9, 1e12]") from None
    if not math.isfinite(count) or not 1e-9 <= count <= 1e12:
        raise ValueError("mean_quanta must be a finite number in [1e-9, 1e12]")
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an integer in [0, 2**64)")
    phantom = make_phantom(size)
    rng = np.random.default_rng(int(seed))
    return rng.poisson(count * phantom).astype(np.float64) / count


def display_image(image: np.ndarray, low: float = 0.0, high: float = 1.2) -> np.ndarray:
    """Map a linear signal to [0, 1] using one explicit, fixed display window.

    This clips only the display, not the values returned by ``sample_image``.
    Do not independently normalize each image when comparing exposures.
    """
    try:
        width = high - low
        window_valid = math.isfinite(low) and math.isfinite(high) and math.isfinite(width) and width > 0
    except (TypeError, ValueError, OverflowError):
        window_valid = False
    if not window_valid:
        raise ValueError("display window must have finite bounds with high > low")
    values = np.asarray(image, dtype=np.float64)
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("image must be a finite, non-empty 2D array")
    return np.clip((values - low) / width, 0.0, 1.0)


def display_png_data_uri(image: np.ndarray) -> str:
    """Embed the small display PNG so reruns need no transient /media request.

    Preserve Streamlit's float-image quantization (multiply by 255 then
    truncate). This affects display transport only, never the source signal.
    """
    pixels = (255 * display_image(image)).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")
