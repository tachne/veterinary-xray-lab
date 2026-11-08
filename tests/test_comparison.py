import csv
import io
from dataclasses import replace

import pytest
import numpy as np

from simulator import APP_VERSION
from simulator.comparison import make_snapshot, restore_inputs, snapshots_csv
from simulator.physics import ExposureSettings, ModelParameters, calculate_exposure


def test_snapshot_retains_all_inputs_and_can_reproduce_results():
    settings = ExposureSettings(mas=2.0)
    params = ModelParameters(target_ei=400.0, grid_transmission=0.5)
    record = make_snapshot(1, "演習A", settings, params, 43)
    new_settings = replace(settings, mas=4.0)
    assert record["settings"]["mas"] == 2.0
    assert new_settings.mas == 4.0
    assert record["parameters"]["target_ei"] == 400.0
    assert record["seed"] == 43
    assert record["saved_at"].endswith("+09:00")
    restored = calculate_exposure(*restore_inputs(record))
    assert restored.ei == record["result"]["ei"]


def test_csv_retains_unrounded_units_model_provenance_and_seed():
    record = make_snapshot(1, "記録,日本語", ExposureSettings(), ModelParameters(), 42)
    data = snapshots_csv([record])
    assert data.startswith(b"\xef\xbb\xbf")
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
    assert rows[0]["name"] == "記録,日本語"
    assert float(rows[0]["result.detector_kerma_ugy"]) == record["result"]["detector_kerma_ugy"]
    assert float(rows[0]["parameters.target_ei"]) == 250.0
    assert rows[0]["seed"] == "42"
    assert "実測値なし" in rows[0]["provenance"]
    assert rows[0]["model_version"] == "0.1.0"
    assert rows[0]["app_version"] == APP_VERSION
    assert rows[0]["numpy_version"] == np.__version__
    assert "I=C/N" in rows[0]["image_signal"]
    assert "fraction" in rows[0]["noise_reference"]


@pytest.mark.parametrize("legacy_first", [True, False])
def test_csv_keeps_legacy_records_without_inventing_missing_metadata(legacy_first):
    old = make_snapshot(1, "既存記録", ExposureSettings(), ModelParameters(), 42)
    for key in ("app_version", "numpy_version", "image_signal", "noise_reference"):
        del old[key]
    new = make_snapshot(2, "新しい記録", ExposureSettings(mas=10), ModelParameters(), 43)
    records = [old, new] if legacy_first else [new, old]
    rows = list(csv.DictReader(io.StringIO(snapshots_csv(records).decode("utf-8-sig"))))
    by_id = {row["id"]: row for row in rows}
    assert by_id["1"]["app_version"] == ""
    assert by_id["1"]["numpy_version"] == ""
    assert by_id["2"]["app_version"] == APP_VERSION
    for record in records:
        row = by_id[str(record["id"])]
        assert float(row["result.ei"]) == record["result"]["ei"]
        assert float(row["settings.mas"]) == record["settings"]["mas"]
    assert "app_version" not in old


@pytest.mark.parametrize("name", ["=1+1", " +SUM(1)", "-test", "@example"])
def test_csv_protects_learner_supplied_formula_names(name):
    record = make_snapshot(1, name, ExposureSettings(), ModelParameters(), 42)
    rows = list(csv.DictReader(io.StringIO(snapshots_csv([record]).decode("utf-8-sig"))))
    assert rows[0]["name"].startswith("'")
    assert float(rows[0]["result.di"]) == record["result"]["di"]


def test_empty_export_and_default_label():
    assert snapshots_csv([]) == b""
    assert make_snapshot(3, "  ", ExposureSettings(), ModelParameters(), 42)["name"] == "記録 03"

