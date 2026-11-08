from copy import deepcopy
import math
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def start():
    app = AppTest.from_file(APP, default_timeout=20).run()
    assert not app.exception
    # Keep the established v0.1.2 regression scenarios on their 256px source.
    app.selectbox(key="test_pattern").set_value("legacy").run()
    assert not app.exception
    return app


def test_start_and_mas_doubling_changes_exposure_noise_and_di():
    app = start()
    assert len(app.metric) == 4
    app.button(key="save_case").click().run()
    base = deepcopy(app.session_state["snapshots"][0])
    app.slider(key="mas").set_value(10.0).run()
    app.button(key="save_case").click().run()
    assert not app.exception
    second = app.session_state["snapshots"][1]
    assert list(app.multiselect(key="image_cases").value) == [1, 2]
    assert second["result"]["ei"] / base["result"]["ei"] == pytest.approx(2)
    assert second["result"]["di"] - base["result"]["di"] == pytest.approx(10*math.log10(2))
    assert second["result"]["relative_noise"] / base["result"]["relative_noise"] == pytest.approx(1/math.sqrt(2))
    assert app.session_state["snapshots"][0] == base


def test_target_only_changes_di_and_noise_resample_only_changes_seed():
    app = start()
    app.button(key="save_case").click().run()
    baseline = deepcopy(app.session_state["snapshots"][0])
    app.number_input(key="target_ei").set_value(500.0).run()
    app.button(key="resample").click().run()
    app.button(key="save_case").click().run()
    assert not app.exception
    new = app.session_state["snapshots"][1]
    assert new["result"]["detector_kerma_ugy"] == baseline["result"]["detector_kerma_ugy"]
    assert new["result"]["mean_quanta"] == baseline["result"]["mean_quanta"]
    assert new["result"]["di"] - baseline["result"]["di"] == pytest.approx(-10*math.log10(2))
    assert new["seed"] == baseline["seed"] + 1
    assert any("モデル係数または目標EIが異なります" in e.value for e in app.warning)


def test_ui_calibration_reference_and_target_only_change():
    app = start()
    app.slider(key="mas").set_value(0.1)
    app.slider(key="thickness").set_value(0.0)
    app.number_input(key="output").set_value(25.0)
    app.run()
    app.button(key="save_case").click().run()
    baseline = deepcopy(app.session_state["snapshots"][0])
    assert baseline["result"]["detector_kerma_ugy"] == pytest.approx(2.5)
    assert app.metric[1].value == "250"
    assert float(app.metric[2].value) == pytest.approx(0.0)
    app.number_input(key="target_ei").set_value(500.0).run()
    app.button(key="save_case").click().run()
    assert not app.exception
    changed = app.session_state["snapshots"][1]
    for key in ("detector_kerma_ugy", "ei", "mean_quanta", "relative_noise", "snr"):
        assert changed["result"][key] == baseline["result"][key]
    assert changed["seed"] == baseline["seed"]
    assert changed["settings"] == baseline["settings"]
    assert app.metric[1].value == "250"
    assert float(app.metric[2].value) == pytest.approx(-3.01)
    assert baseline["parameters"]["target_ei"] == 250.0
    assert changed["parameters"]["target_ei"] == 500.0


def test_snapshots_limit_deletion_and_selection_cleanup():
    app = start()
    for _ in range(8):
        app.button(key="save_case").click().run()
    assert not app.exception
    assert len(app.session_state["snapshots"]) == 8
    assert app.button(key="save_case").disabled
    app.multiselect(key="image_cases").set_value([1, 8]).run()
    app.multiselect(key="delete_ids").set_value([1, 8]).run()
    app.button(key="delete_cases").click().run()
    assert not app.exception
    assert len(app.session_state["snapshots"]) == 6
    assert not app.button(key="save_case").disabled
    assert {s["id"] for s in app.session_state["snapshots"]} == set(range(2, 8))
    app.multiselect(key="delete_ids").set_value(list(range(2, 8))).run()
    app.button(key="delete_cases").click().run()
    assert not app.exception
    assert not app.session_state["snapshots"]


@pytest.mark.parametrize("axis", ["mAs", "kVp", "SID", "体厚"])
def test_each_curve_axis_renders(axis):
    app = start()
    app.selectbox(key="curve_axis").set_value(axis).run()
    assert not app.exception
    assert len(app.get("vega_lite_chart")) == 4


def test_ui_extreme_low_counts_show_explanation_and_keep_comparisons_working():
    app = start()
    for key, value in {"kvp":40.0, "mas":0.1, "sid":200.0, "thickness":35.0}.items():
        app.slider(key=key).set_value(value)
    for key, value in {"output":1.0, "n":3.0, "mu":0.30, "p":2.0, "grid":0.1, "quanta":100.0}.items():
        app.number_input(key=key).set_value(value)
    app.run()
    assert not app.exception
    assert any("画像生成の計算範囲" in item.value for item in app.warning)
    app.button(key="save_case").click().run()
    assert not app.exception
    assert app.session_state["snapshots"][0]["result"]["mean_quanta"] < 1e-9


def test_ui_maximum_exposure_and_zero_attenuation_exponent():
    app = start()
    for key, value in {"kvp":120.0, "mas":20.0, "sid":50.0, "thickness":0.0}.items():
        app.slider(key=key).set_value(value)
    for key, value in {"output":100.0, "n":3.0, "mu":0.01, "p":0.0, "grid":1.0, "quanta":10000.0}.items():
        app.number_input(key=key).set_value(value)
    app.run()
    assert not app.exception
    assert not app.error
    app.button(key="save_case").click().run()
    assert app.session_state["snapshots"][0]["result"]["detector_kerma_ugy"] == pytest.approx(27000)
