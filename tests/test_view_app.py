"""v0.1.2 UI contracts: display independence, seed control, and frozen references."""

import base64
from copy import deepcopy
from dataclasses import asdict
import io
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from streamlit.testing.v1 import AppTest

from simulator.imaging import make_phantom, sample_image
from simulator.physics import ExposureSettings, ModelParameters, calculate_exposure


APP = str(Path(__file__).resolve().parents[1] / "app.py")


def start():
    app = AppTest.from_file(APP, default_timeout=20).run()
    assert not app.exception
    # Preserve every existing 256px pixel oracle while the default is extended.
    app.selectbox(key="test_pattern").set_value("legacy").run()
    assert not app.exception
    return app


def comparison_pixels(app):
    """Read only the two main columns, excluding ideal-signal expanders."""
    image_tabs = [tab for tab in app.tabs if tab.label == "模擬画像"]
    assert len(image_tabs) == 1
    columns = image_tabs[0].get("column")
    assert len(columns) == 2
    images = []
    # Streamlit 1.67 exposes the ImageList proto through the 'image' element.
    for column in columns:
        elements = column.get("image")
        assert len(elements) == 1
        assert len(elements[0].proto.imgs) == 1
        entry = elements[0].proto.imgs[0]
        assert entry.url.startswith("data:image/png;base64,")
        encoded = entry.url.split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
            assert image.format == "PNG"
            assert image.mode == "L"
            images.append(np.asarray(image).copy())
    assert len(images) == 2
    return images


def expected_pixels(result, seed, factor, zoom):
    """Independent fixed-window oracle for the 256-pixel UI comparison.

    The presentation helper is deliberately not used here: this verifies that
    the UI wires both panes to the specified factor, central ROI and window.
    """
    raw = sample_image(result["mean_quanta"], seed=seed)
    ideal = make_phantom()
    signal = raw if factor == 1 else ideal + factor * (raw - ideal)
    side = 256 // zoom
    start = (256 - side) // 2
    roi = signal[start:start + side, start:start + side]
    pixels = (255 * np.clip(roi / 1.2, 0.0, 1.0)).astype(np.uint8)
    return np.repeat(np.repeat(pixels, zoom, axis=0), zoom, axis=1)


@pytest.mark.parametrize("gain", [1, 2, 3, 5])
@pytest.mark.parametrize("zoom", [1, 2, 4])
def test_all_display_choices_leave_saved_physics_and_seed_unchanged(gain, zoom):
    app = start()
    app.button(key="save_case").click().run()
    baseline = deepcopy(app.session_state["snapshots"][0])
    metrics = [metric.value for metric in app.metric]
    app.selectbox(key="display_gain").set_value(gain)
    app.selectbox(key="display_zoom").set_value(zoom)
    app.run()
    app.button(key="save_case").click().run()
    assert not app.exception
    changed = app.session_state["snapshots"][1]
    for field in ("settings", "parameters", "result", "seed"):
        assert changed[field] == baseline[field]
    assert changed["enhancement_factor"] == gain
    assert changed["zoom"] == zoom
    assert changed["seed_mode"] == "fixed"
    assert changed["seed"] == app.session_state["noise_seed"] == 42
    assert [metric.value for metric in app.metric] == metrics


def test_seed_42_reproduces_both_displayed_images_after_another_seed():
    app = start()
    app.selectbox(key="display_gain").set_value(5)
    app.selectbox(key="display_zoom").set_value(2)
    app.run()
    first = comparison_pixels(app)
    app.number_input(key="fixed_seed").set_value(43).run()
    assert not app.exception
    assert app.session_state["noise_seed"] == 43
    second = comparison_pixels(app)
    assert all(not np.array_equal(before, after) for before, after in zip(first, second))
    app.number_input(key="fixed_seed").set_value(42).run()
    assert not app.exception
    assert app.session_state["noise_seed"] == 42
    for original, reproduced in zip(first, comparison_pixels(app)):
        np.testing.assert_array_equal(reproduced, original)


def test_fixed_mode_keeps_seed_on_condition_changes_and_resample_updates_seed_control():
    app = start()
    assert app.radio(key="seed_mode").value == "fixed"
    app.slider(key="mas").set_value(10.0).run()
    app.number_input(key="output").set_value(60.0).run()
    assert not app.exception
    assert app.session_state["noise_seed"] == 42
    assert app.number_input(key="fixed_seed").value == 42
    app.button(key="resample").click().run()
    assert not app.exception
    assert app.session_state["noise_seed"] == 43
    assert app.number_input(key="fixed_seed").value == 43


def test_new_sample_mode_changes_seed_only_for_physical_conditions():
    app = start()
    app.radio(key="seed_mode").set_value("new_on_conditions").run()
    assert app.session_state["noise_seed"] == 42
    app.slider(key="mas").set_value(10.0).run()
    assert app.session_state["noise_seed"] == 43
    app.number_input(key="output").set_value(60.0).run()
    assert app.session_state["noise_seed"] == 44
    app.selectbox(key="display_gain").set_value(5)
    app.selectbox(key="display_zoom").set_value(4)
    app.run()
    assert app.session_state["noise_seed"] == 44
    app.number_input(key="target_ei").set_value(500.0).run()
    assert app.session_state["noise_seed"] == 44
    app.button(key="save_case").click().run()
    assert app.session_state["noise_seed"] == 44
    assert app.session_state["snapshots"][0]["seed_mode"] == "new_on_conditions"
    app.radio(key="image_comparison").set_value("reference_current").run()
    assert app.session_state["noise_seed"] == 44
    app.multiselect(key="image_cases").set_value([]).run()
    assert app.session_state["noise_seed"] == 44
    app.run()
    assert not app.exception
    assert app.session_state["noise_seed"] == 44


def test_reference_snapshot_is_frozen_until_explicitly_pinned():
    app = start()
    initial = deepcopy(app.session_state["reference_snapshot"])
    assert initial["settings"] == asdict(ExposureSettings())
    assert initial["parameters"] == asdict(ModelParameters())
    assert initial["result"] == asdict(calculate_exposure(ExposureSettings(), ModelParameters()))
    assert initial["seed"] == 42
    app.slider(key="mas").set_value(10.0).run()
    assert app.session_state["reference_snapshot"] == initial
    app.button(key="pin_reference").click().run()
    frozen = deepcopy(app.session_state["reference_snapshot"])
    assert frozen["settings"]["mas"] == 10.0
    assert frozen["seed"] == 42
    assert frozen["result"] == asdict(calculate_exposure(ExposureSettings(mas=10.0), ModelParameters()))
    app.slider(key="mas").set_value(15.0).run()
    app.button(key="resample").click().run()
    assert not app.exception
    assert app.session_state["noise_seed"] == 43
    assert app.session_state["reference_snapshot"] == frozen


@pytest.mark.parametrize("mode", ["standard_emphasis", "reference_current"])
def test_comparison_pngs_use_a_shared_fixed_window_and_central_zoom(mode):
    app = start()
    reference = deepcopy(app.session_state["reference_snapshot"])
    app.slider(key="mas").set_value(10.0)
    app.selectbox(key="display_gain").set_value(5)
    app.radio(key="image_comparison").set_value(mode)
    app.run()
    current = asdict(calculate_exposure(ExposureSettings(mas=10.0), ModelParameters()))
    for zoom in (1, 2, 4):
        app.selectbox(key="display_zoom").set_value(zoom).run()
        assert not app.exception
        assert app.session_state["noise_seed"] == 42
        left, right = comparison_pixels(app)
        if mode == "standard_emphasis":
            expected_left = expected_pixels(current, 42, factor=1, zoom=zoom)
        else:
            expected_left = expected_pixels(reference["result"], reference["seed"], factor=5, zoom=zoom)
        expected_right = expected_pixels(current, 42, factor=5, zoom=zoom)
        assert left.shape == right.shape == (256, 256)
        np.testing.assert_array_equal(left, expected_left)
        np.testing.assert_array_equal(right, expected_right)
