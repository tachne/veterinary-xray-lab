"""Extended-pattern UI integration without changing the legacy regression set."""

import base64
from copy import deepcopy
import csv
from dataclasses import asdict
import io
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from streamlit.testing.v1 import AppTest

from simulator import comparison
from simulator.patterns import make_extended_phantom, sample_extended_image
from simulator.physics import ExposureSettings, ModelParameters, calculate_exposure


APP = str(Path(__file__).resolve().parents[1] / "app.py")


def start_extended():
    app = AppTest.from_file(APP, default_timeout=20).run()
    assert not app.exception
    assert app.selectbox(key="test_pattern").value == "extended"
    return app


def png_pixels(element):
    assert len(element.proto.imgs) == 1
    uri = element.proto.imgs[0].url
    assert uri.startswith("data:image/png;base64,")
    with Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1], validate=True))) as image:
        assert image.format == "PNG"
        assert image.mode == "L"
        return np.asarray(image).copy()


def main_comparison_pixels(app):
    tab = next(tab for tab in app.tabs if tab.label == "模擬画像")
    columns = tab.get("column")
    assert len(columns) == 2
    assert all(len(column.get("image")) == 1 for column in columns)
    return [png_pixels(column.get("image")[0]) for column in columns]


def expected_extended_pixels(result, seed, gain=1, zoom=1):
    raw = sample_extended_image(result["mean_quanta"], seed=seed)
    ideal = make_extended_phantom()
    signal = raw if gain == 1 else ideal + gain * (raw - ideal)
    side = 512 // zoom
    start = (512 - side) // 2
    roi = signal[start:start + side, start:start + side]
    pixels = (255 * np.clip(roi / 1.2, 0.0, 1.0)).astype(np.uint8)
    return np.repeat(np.repeat(pixels, zoom, axis=0), zoom, axis=1)


@pytest.mark.parametrize("mode", ["standard_emphasis", "reference_current"])
def test_default_extended_comparison_panes_share_512_source_window_and_zoom(mode):
    app = start_extended()
    reference = deepcopy(app.session_state["reference_snapshot"])
    assert reference["image_pattern"] == "extended"
    assert reference["image_size_px"] == 512
    initial = expected_extended_pixels(reference["result"], 42)
    for pixels in main_comparison_pixels(app):
        assert pixels.shape == (512, 512)
        np.testing.assert_array_equal(pixels, initial)

    app.slider(key="mas").set_value(10.0)
    app.selectbox(key="display_gain").set_value(5)
    app.selectbox(key="display_zoom").set_value(4)
    app.radio(key="image_comparison").set_value(mode)
    app.run()
    assert not app.exception
    current = asdict(calculate_exposure(ExposureSettings(mas=10.0), ModelParameters()))
    left, right = main_comparison_pixels(app)
    if mode == "standard_emphasis":
        expected_left = expected_extended_pixels(current, 42, gain=1, zoom=4)
    else:
        expected_left = expected_extended_pixels(reference["result"], 42, gain=5, zoom=4)
    np.testing.assert_array_equal(left, expected_left)
    np.testing.assert_array_equal(right, expected_extended_pixels(current, 42, gain=5, zoom=4))

    # Saved-case comparisons must also use the selected common 512px pattern.
    app.button(key="save_case").click().run()
    app.slider(key="mas").set_value(5.0).run()
    app.button(key="save_case").click().run()
    assert not app.exception
    saved_tab = next(tab for tab in app.tabs if tab.label.startswith("条件を比較"))
    saved_images = saved_tab.get("image")
    assert len(saved_images) == 2
    for image, record in zip(saved_images, app.session_state["snapshots"]):
        pixels = png_pixels(image)
        assert pixels.shape == (512, 512)
        np.testing.assert_array_equal(
            pixels, expected_extended_pixels(record["result"], record["seed"], gain=5, zoom=4),
        )


def test_pattern_display_and_new_noise_choices_do_not_change_physical_values():
    app = start_extended()
    app.button(key="save_case").click().run()
    baseline = deepcopy(app.session_state["snapshots"][0])
    metrics = [metric.value for metric in app.metric]
    app.radio(key="seed_mode").set_value("new_on_conditions").run()
    app.selectbox(key="test_pattern").set_value("legacy")
    app.selectbox(key="display_gain").set_value(5)
    app.selectbox(key="display_zoom").set_value(4)
    app.run()
    app.button(key="save_case").click().run()
    changed = app.session_state["snapshots"][1]
    assert changed["image_pattern"] == "legacy"
    assert changed["image_size_px"] == 256
    assert changed["seed"] == app.session_state["noise_seed"] == baseline["seed"]
    for field in ("settings", "parameters", "result"):
        assert changed[field] == baseline[field]
    assert [metric.value for metric in app.metric] == metrics

    app.selectbox(key="test_pattern").set_value("extended").run()
    assert app.session_state["noise_seed"] == baseline["seed"]
    app.button(key="resample").click().run()
    app.button(key="save_case").click().run()
    assert not app.exception
    resampled = app.session_state["snapshots"][2]
    assert resampled["image_pattern"] == "extended"
    assert resampled["seed"] == baseline["seed"] + 1
    assert resampled["result"] == baseline["result"]
    assert [metric.value for metric in app.metric] == metrics
    assert app.session_state["snapshots"][0] == baseline


def test_csv_distinguishes_saved_pattern_from_current_common_view(monkeypatch):
    exports = []
    original_csv = comparison.snapshots_csv

    def capture_csv(records):
        data = original_csv(records)
        exports.append(data)
        return data

    monkeypatch.setattr(comparison, "snapshots_csv", capture_csv)
    app = start_extended()
    app.selectbox(key="display_gain").set_value(3).run()
    app.button(key="save_case").click().run()
    app.selectbox(key="test_pattern").set_value("legacy")
    app.selectbox(key="display_gain").set_value(2)
    app.run()
    app.button(key="save_case").click().run()
    originals = deepcopy(app.session_state["snapshots"])
    app.selectbox(key="test_pattern").set_value("extended")
    app.selectbox(key="display_gain").set_value(5)
    app.selectbox(key="display_zoom").set_value(4)
    app.run()
    assert not app.exception
    assert exports
    rows = list(csv.DictReader(io.StringIO(exports[-1].decode("utf-8-sig"))))
    assert len(rows) == 2
    for row, record in zip(rows, originals):
        assert row["image_pattern"] == record["image_pattern"]
        assert row["pattern_version"] == record["pattern_version"]
        assert int(row["image_size_px"]) == record["image_size_px"]
        assert float(row["enhancement_factor"]) == record["enhancement_factor"]
        assert row["view_pattern"] == "extended"
        assert float(row["view_enhancement_factor"]) == 5
        assert float(row["view_zoom"]) == 4
        assert float(row["view_window_low"]) == 0
        assert float(row["view_window_high"]) == 1.2
        assert row["view_roi"] == "center"
        assert row["view_interpolation"] == "nearest"
        assert int(row["seed"]) == record["seed"]
        for field, value in record["result"].items():
            assert float(row[f"result.{field}"]) == value
    assert app.session_state["snapshots"] == originals
