"""Display metadata must not become new physical exposure conditions."""

import copy
import csv
from dataclasses import asdict, replace
import io
import math

import numpy as np
import pytest

from simulator.comparison import make_snapshot, snapshots_csv
from simulator.imaging import sample_image
from simulator.physics import ExposureSettings, ModelParameters, calculate_exposure


DISPLAY_FIELDS = (
    "enhancement_factor", "zoom", "seed_mode", "display_formula",
    "display_interpretation",
)


def test_display_options_leave_calculation_inputs_results_and_seed_unchanged():
    settings = ExposureSettings(mas=2.0, thickness_cm=25.0)
    parameters = ModelParameters(target_ei=400.0)
    baseline = make_snapshot(1, "標準表示", settings, parameters, 42)
    enhanced = make_snapshot(
        2, "強調・拡大", settings, parameters, 42,
        enhancement_factor=5.0, zoom=4.0, seed_mode="new_on_conditions",
    )
    assert baseline["enhancement_factor"] == 1.0
    assert baseline["zoom"] == 1.0
    assert baseline["seed_mode"] == "fixed"
    assert enhanced["enhancement_factor"] == 5.0
    assert enhanced["zoom"] == 4.0
    assert enhanced["seed_mode"] == "new_on_conditions"
    assert enhanced["result"] == asdict(calculate_exposure(settings, parameters))
    for field in ("settings", "parameters", "result", "seed"):
        assert enhanced[field] == baseline[field]
    np.testing.assert_array_equal(
        sample_image(baseline["result"]["mean_quanta"], baseline["seed"]),
        sample_image(enhanced["result"]["mean_quanta"], enhanced["seed"]),
    )


@pytest.mark.parametrize("legacy_first", [True, False])
def test_csv_tracks_display_conditions_and_preserves_unknown_legacy_values(legacy_first):
    legacy = make_snapshot(1, "旧記録", ExposureSettings(), ModelParameters(), 42)
    for field in DISPLAY_FIELDS:
        del legacy[field]
    current = make_snapshot(
        2, "新記録", ExposureSettings(mas=10.0), ModelParameters(), 43,
        enhancement_factor=3.0, zoom=2.0, seed_mode="fixed",
    )
    records = [legacy, current] if legacy_first else [current, legacy]
    originals = copy.deepcopy(records)
    rows = list(csv.DictReader(io.StringIO(snapshots_csv(records).decode("utf-8-sig"))))
    by_id = {row["id"]: row for row in rows}
    for field in DISPLAY_FIELDS:
        assert by_id["1"][field] == ""
    assert float(by_id["2"]["enhancement_factor"]) == 3.0
    assert float(by_id["2"]["zoom"]) == 2.0
    assert by_id["2"]["seed_mode"] == "fixed"
    assert by_id["2"]["display_formula"] == "S+g*(I-S)"
    assert "生の信号I" in by_id["2"]["display_interpretation"]
    assert "不変" in by_id["2"]["display_interpretation"]
    for record in records:
        row = by_id[str(record["id"])]
        assert int(row["seed"]) == record["seed"]
        for field, value in record["result"].items():
            assert float(row[f"result.{field}"]) == value
    assert records == originals


@pytest.mark.parametrize(
    ("changed_settings", "expected_kerma_ratio"),
    [
        (ExposureSettings(mas=20.0), 4.0),
        (ExposureSettings(thickness_cm=20.0), math.exp(-0.18 * 5.0)),
    ],
    ids=["fourfold_mas_reduces_noise", "greater_thickness_reduces_kerma_increases_noise"],
)
def test_display_metadata_preserves_physical_exposure_and_noise_relations(
    changed_settings, expected_kerma_ratio,
):
    parameters = ModelParameters()
    baseline = make_snapshot(1, "基準", ExposureSettings(), parameters, 42)
    changed = make_snapshot(
        2, "条件変更", changed_settings, parameters, 42,
        enhancement_factor=5.0, zoom=4.0,
    )
    first, second = baseline["result"], changed["result"]
    for field in ("detector_kerma_ugy", "ei", "mean_quanta"):
        assert second[field] / first[field] == pytest.approx(expected_kerma_ratio)
    assert second["relative_noise"] / first["relative_noise"] == pytest.approx(
        1.0 / math.sqrt(expected_kerma_ratio)
    )
    assert second["snr"] / first["snr"] == pytest.approx(math.sqrt(expected_kerma_ratio))
    assert changed["seed"] == baseline["seed"]
