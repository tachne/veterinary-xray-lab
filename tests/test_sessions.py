"""Session-state isolation, parallel pure calculations, and portable resource paths.

AppTest instances are deliberately interleaved sequentially: its process-wide
test machinery and the CSV observer are not used from concurrent threads.
This is a functional session-isolation regression, not a simultaneous browser
WebSocket, authentication, privacy, or deployment load test. No assertion here
claims that Streamlit or Python never writes runtime files or emits logs.
"""

import base64
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import csv
from dataclasses import asdict
import io
from pathlib import Path

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

from simulator import comparison
from simulator.imaging import sample_image
from simulator.patterns import sample_extended_image
from simulator.physics import ExposureSettings, ModelParameters, calculate_exposure


ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")


def _current_image_size(app):
    tab = next(tab for tab in app.tabs if tab.label == "模擬画像")
    columns = tab.get("column")
    assert len(columns) == 2
    image = columns[1].get("image")[0]
    uri = image.proto.imgs[0].url
    assert uri.startswith("data:image/png;base64,")
    with Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1], validate=True))) as decoded:
        return decoded.size


def test_interleaved_sessions_keep_inputs_records_references_seeds_and_csv_separate(monkeypatch):
    exports = defaultdict(list)
    active_session = [None]
    original_csv = comparison.snapshots_csv

    def observe_csv(records):
        # Preserve the real exporter and inspect its real result. This observer
        # does not replace persistence, filesystem, or network behavior.
        data = original_csv(records)
        assert active_session[0] is not None
        exports[active_session[0]].append(data)
        return data

    monkeypatch.setattr(comparison, "snapshots_csv", observe_csv)

    def run(label, app):
        active_session[0] = label
        try:
            app.run()
        finally:
            active_session[0] = None
        assert not app.exception
        return app

    def rows(label):
        return list(csv.DictReader(io.StringIO(exports[label][-1].decode("utf-8-sig"))))

    student_a = run("a", AppTest.from_file(APP, default_timeout=20))
    student_b = run("b", AppTest.from_file(APP, default_timeout=20))
    assert student_a.session_state["snapshots"] == student_b.session_state["snapshots"] == []
    default_b_reference = deepcopy(student_b.session_state["reference_snapshot"])

    student_a.slider(key="mas").set_value(2.0)
    student_a.slider(key="kvp").set_value(70.0)
    student_a.slider(key="thickness").set_value(10.0)
    student_a.number_input(key="target_ei").set_value(300.0)
    student_a.number_input(key="fixed_seed").set_value(111)
    student_a.selectbox(key="test_pattern").set_value("legacy")
    student_a.selectbox(key="display_gain").set_value(3)
    student_a.selectbox(key="display_zoom").set_value(2)
    student_a.text_input(key="case_name").set_value("student-a-first")
    run("a", student_a)
    student_a.button(key="save_case").click()
    run("a", student_a)
    student_a.button(key="pin_reference").click()
    run("a", student_a)
    a_reference = deepcopy(student_a.session_state["reference_snapshot"])
    a_first = deepcopy(student_a.session_state["snapshots"][0])
    assert a_first["result"] == asdict(calculate_exposure(
        ExposureSettings(kvp=70, mas=2, thickness_cm=10), ModelParameters(target_ei=300),
    ))

    # A's previous reruns must not supply B's widgets, reference, or history.
    run("b", student_b)
    assert student_b.slider(key="mas").value == 5.0
    assert student_b.number_input(key="target_ei").value == 250.0
    assert student_b.text_input(key="case_name").value == ""
    assert student_b.session_state["snapshots"] == []
    assert student_b.session_state["reference_snapshot"] == default_b_reference
    assert student_b.session_state["noise_seed"] == 42

    student_b.slider(key="mas").set_value(12.0)
    student_b.slider(key="kvp").set_value(100.0)
    student_b.slider(key="thickness").set_value(20.0)
    student_b.number_input(key="target_ei").set_value(500.0)
    student_b.number_input(key="fixed_seed").set_value(222)
    student_b.selectbox(key="display_gain").set_value(5)
    student_b.selectbox(key="display_zoom").set_value(4)
    student_b.text_input(key="case_name").set_value("student-b-only")
    run("b", student_b)
    student_b.radio(key="seed_mode").set_value("new_on_conditions")
    run("b", student_b)
    student_b.button(key="resample").click()
    run("b", student_b)
    student_b.button(key="save_case").click()
    run("b", student_b)
    student_b.button(key="pin_reference").click()
    run("b", student_b)
    b_reference = deepcopy(student_b.session_state["reference_snapshot"])
    b_records = deepcopy(student_b.session_state["snapshots"])
    assert b_records[0]["seed"] == 223
    assert b_records[0]["result"] == asdict(calculate_exposure(
        ExposureSettings(kvp=100, mas=12, thickness_cm=20), ModelParameters(target_ei=500),
    ))

    run("a", student_a)
    assert student_a.session_state["snapshots"] == [a_first]
    assert student_a.session_state["reference_snapshot"] == a_reference
    assert student_a.radio(key="seed_mode").value == "fixed"
    assert student_a.session_state["noise_seed"] == 111
    assert student_a.slider(key="mas").value == 2.0
    assert _current_image_size(student_a) == (256, 256)
    student_a.button(key="resample").click()
    run("a", student_a)
    student_a.slider(key="mas").set_value(4.0)
    student_a.text_input(key="case_name").set_value("student-a-second")
    run("a", student_a)
    student_a.button(key="save_case").click()
    run("a", student_a)
    a_second = deepcopy(student_a.session_state["snapshots"][1])
    assert a_second["seed"] == 112
    assert student_a.session_state["reference_snapshot"] == a_reference

    # B's automatic seed advances only for B's own physical-condition change.
    student_b.slider(key="mas").set_value(13.0)
    run("b", student_b)
    assert student_b.session_state["noise_seed"] == 224
    assert student_b.session_state["snapshots"] == b_records
    assert student_b.session_state["reference_snapshot"] == b_reference
    assert _current_image_size(student_b) == (512, 512)

    student_a.multiselect(key="delete_ids").set_value([1])
    run("a", student_a)
    student_a.button(key="delete_cases").click()
    run("a", student_a)
    assert student_a.session_state["snapshots"] == [a_second]
    assert student_b.session_state["snapshots"] == b_records
    assert student_a.session_state["snapshots"] is not student_b.session_state["snapshots"]

    # A third new client starts from defaults, even after both other histories
    # have been populated and one has deleted a record with the same ID.
    student_c = run("c", AppTest.from_file(APP, default_timeout=20))
    assert student_c.session_state["snapshots"] == []
    assert student_c.session_state["next_snapshot_id"] == 1
    assert student_c.session_state["noise_seed"] == 42
    assert student_c.radio(key="seed_mode").value == "fixed"
    assert student_c.selectbox(key="test_pattern").value == "extended"
    assert student_c.text_input(key="case_name").value == ""
    assert student_c.session_state["reference_snapshot"]["settings"] == asdict(ExposureSettings())
    assert student_c.session_state["reference_snapshot"]["parameters"] == asdict(ModelParameters())
    assert not exports["c"]

    run("a", student_a)
    run("b", student_b)
    assert student_a.session_state["noise_seed"] == 112
    assert student_b.session_state["noise_seed"] == 224
    assert student_a.session_state["reference_snapshot"] == a_reference
    assert student_b.session_state["reference_snapshot"] == b_reference
    assert student_a.text_input(key="case_name").value == "student-a-second"
    assert student_b.text_input(key="case_name").value == "student-b-only"
    for label, record, pattern, size in (
        ("a", a_second, "legacy", 256), ("b", b_records[0], "extended", 512),
    ):
        exported = rows(label)
        assert len(exported) == 1
        row = exported[0]
        assert row["name"] == record["name"]
        assert int(row["id"]) == record["id"]
        assert int(row["seed"]) == record["seed"]
        assert row["seed_mode"] == record["seed_mode"]
        assert row["image_pattern"] == row["view_pattern"] == pattern
        assert int(row["image_size_px"]) == int(row["view_image_size_px"]) == size
        for field, value in record["result"].items():
            assert float(row[f"result.{field}"]) == value


def test_parallel_pure_calculations_use_independent_rng_and_fresh_arrays():
    jobs = [
        ("legacy", ExposureSettings(mas=0.2, thickness_cm=25), 42),
        ("extended", ExposureSettings(kvp=60, mas=1, thickness_cm=20), 43),
        ("legacy", ExposureSettings(kvp=100, mas=12, thickness_cm=10), 2026),
        ("extended", ExposureSettings(kvp=120, mas=20, sid_cm=50, thickness_cm=0), 7),
    ] * 2

    def simulate(job):
        pattern, settings, seed = job
        result = calculate_exposure(settings, ModelParameters())
        sample = sample_extended_image if pattern == "extended" else sample_image
        return asdict(result), sample(result.mean_quanta, seed=seed)

    global_rng_before = np.random.get_state()
    expected = [simulate(job) for job in jobs]
    with ThreadPoolExecutor(max_workers=4) as executor:
        concurrent = list(executor.map(simulate, jobs))
    for (result, pixels), (expected_result, expected_pixels) in zip(concurrent, expected):
        assert result == expected_result
        np.testing.assert_array_equal(pixels, expected_pixels)
    global_rng_after = np.random.get_state()
    assert global_rng_after[0] == global_rng_before[0]
    np.testing.assert_array_equal(global_rng_after[1], global_rng_before[1])
    assert global_rng_after[2:] == global_rng_before[2:]

    # Duplicate requests produce equal but separately owned arrays. Mutating a
    # returned array cannot contaminate another user or a subsequent request.
    assert not np.shares_memory(concurrent[0][1], concurrent[4][1])
    concurrent[0][1][0, 0] = -999
    np.testing.assert_array_equal(concurrent[4][1], expected[4][1])
    np.testing.assert_array_equal(simulate(jobs[0])[1], expected[0][1])


def test_app_loads_assets_and_documents_from_an_unrelated_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(ROOT))
    app = AppTest.from_file(APP, default_timeout=20).run()
    assert not app.exception
    assert not app.error
    assert app.selectbox(key="test_pattern").value == "extended"
    assert _current_image_size(app) == (512, 512)
    svg_urls = [entry.url for image in app.get("image") for entry in image.proto.imgs
                if entry.url.startswith("data:image/svg+xml")]
    assert svg_urls, "The geometry SVG must be loaded, not omitted from the application."
    for label in ("計算式・物理モデル説明書", "未実装機能・開発順序"):
        expander = next(expander for expander in app.expander if expander.label == label)
        assert expander.markdown
        assert expander.markdown[0].value.strip()
    assert Path.cwd() == tmp_path
