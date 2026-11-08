"""Reproducible session snapshots and a self-describing UTF-8 CSV export."""

from dataclasses import asdict
from datetime import datetime
import csv
import io
from zoneinfo import ZoneInfo

import numpy as np

from simulator import APP_VERSION
from simulator.physics import ExposureSettings, ModelParameters, calculate_exposure
from simulator.patterns import PATTERN_SPECS

# The exposure equations remain unchanged; pattern versions are tracked separately.
MODEL_VERSION = "0.1.0"
MAX_SNAPSHOTS = 8


def make_snapshot(
    identifier, name, settings, parameters, seed, *,
    enhancement_factor=1.0, zoom=1.0, seed_mode="fixed", image_pattern="legacy",
):
    """Freeze calculation inputs and independent, display-only metadata.

    Values are assumptions, never measurements. The display options do not
    alter the exposure calculation, the raw signal I, or the supplied seed.
    ``seed_mode`` records the UI choice; this function never generates a seed.
    """
    if image_pattern not in PATTERN_SPECS:
        raise ValueError("Unknown image pattern")
    pattern = PATTERN_SPECS[image_pattern]
    return {
        "id": identifier,
        "name": name.strip()[:50] or f"記録 {identifier:02d}",
        "saved_at": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "app_version": APP_VERSION,
        "model_version": MODEL_VERSION,
        "provenance": "教育用仮定値・物理近似（実測値なし）",
        "ei_interpretation": "校正関係を仮定したEI・患者線量ではない",
        "seed": seed,
        "numpy_version": np.__version__,
        "image_signal": "I=C/N; dimensionless, before display clipping",
        "noise_reference": "S=1; relative_noise=SD/mean (fraction); SNR=sqrt(N)",
        "image_size_px": 512 if image_pattern == "extended" else 256,
        "image_pattern": image_pattern,
        "pattern_version": pattern["version"],
        "display_low": 0.0,
        "display_high": 1.2,
        "enhancement_factor": enhancement_factor,
        "zoom": zoom,
        "seed_mode": seed_mode,
        "display_formula": "S+g*(I-S)",
        "display_interpretation": (
            "表示専用：既知の無雑音信号Sからの偏差I-Sをg倍。"
            "生の信号I・検出器カーマ・EI・DI・理論ノイズ・SNRは不変。"
            "zoomは中央ROIの表示拡大倍率。"
        ),
        "settings": asdict(settings),
        "parameters": asdict(parameters),
        "result": asdict(calculate_exposure(settings, parameters)),
    }


def _csv_safe(value):
    # Learner-provided names may be opened by a spreadsheet as formulas.
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def snapshots_csv(snapshots):
    """Return UTF-8 BOM bytes, including units in all physical field names."""
    if not snapshots:
        return b""
    rows = []
    for record in snapshots:
        row = {k: v for k, v in record.items() if k not in ("settings", "parameters", "result")}
        for group in ("settings", "parameters", "result"):
            row.update({f"{group}.{k}": v for k, v in record[group].items()})
        rows.append({k: _csv_safe(v) for k, v in row.items()})
    output = io.StringIO(newline="")
    # A live session may contain records made before an application update.
    # Keep their original metadata and leave unknown fields blank, not inferred.
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def restore_inputs(snapshot):
    return ExposureSettings(**snapshot["settings"]), ModelParameters(**snapshot["parameters"])

