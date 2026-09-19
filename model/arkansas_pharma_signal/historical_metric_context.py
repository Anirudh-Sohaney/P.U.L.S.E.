"""Leakage-safe historical context assembly for modular training.

Only sources with an event or archived-capture date are admitted here. The
builder uses the latest observation at or before a quarterly feature origin;
it never consumes the current qualified forecast artifact. Unmatched NDCs
remain missing rather than receiving a national signal by broadcast.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .recall_pressure import load_recall_events
from .respnet import load_respnet_rsv_weekly


METRIC_CONTEXT_FEATURES = (
    "metric__ndc_monthly_shortage_supplier_count",
    "metric__nadac_next_observed_price",
    "metric__arkansas_region_annual_demand_five_state",
    "metric__ndc_monthly_recall_severity",
    "metric__supplier_ndc_monthly_recall_severity",
    "metric__arcos_zip3_drug_next_quarter_distribution_state",
    "metric__arkansas_weekly_nssp_influenza_ed_pressure_state",
    "metric__national_weekly_respnet_rsv_hospitalization_pressure_state",
)
METRIC_CONTEXT_WIDTH = len(METRIC_CONTEXT_FEATURES) * 2


def _ndc9(value: object) -> str:
    digits = "".join(character for character in str(value) if character.isdigit())
    if len(digits) == 9:
        return digits
    return digits[:9] if len(digits) >= 11 else ""


def _quarter_end(year: int, quarter: int) -> pd.Period:
    return pd.Period(f"{int(year)}Q{int(quarter)}", freq="Q").asfreq("M", "end")


def _state_context(value: object) -> float:
    """Return a finite numeric metric value, or explicit missingness."""
    if pd.isna(value):
        return np.nan
    numeric = float(value)
    return numeric if np.isfinite(numeric) and numeric >= 0.0 else np.nan


def _index_period_values(frame: pd.DataFrame, key_columns: list[str],
                         value_column: str) -> dict[str, list[tuple[pd.Period, float]]]:
    indexed: dict[str, list[tuple[pd.Period, float]]] = {}
    for row in frame.itertuples(index=False):
        key = "|".join(str(getattr(row, column)) for column in key_columns)
        indexed.setdefault(key, []).append(
            (getattr(row, "month"), float(getattr(row, value_column))))
    for values in indexed.values():
        values.sort(key=lambda item: item[0])
    return indexed


def _latest_before(index: dict[str, list[tuple[pd.Period, float]]], key: str,
                   period: pd.Period) -> float:
    values = index.get(key, [])
    eligible = [value for observed, value in values if observed <= period]
    return eligible[-1] if eligible else np.nan


@lru_cache(maxsize=4)
def _shortage_lookup(path_string: str) -> dict[str, list[tuple[pd.Period, float]]]:
    path = Path(path_string)
    if not path.exists():
        return {}
    frame = pd.read_csv(path, usecols=["ndc9", "month", "supplier", "shortage_active"],
                        low_memory=False)
    frame["ndc9"] = frame["ndc9"].map(_ndc9)
    frame["month"] = pd.PeriodIndex(frame["month"].astype(str), freq="M")
    frame["shortage_active"] = pd.to_numeric(frame["shortage_active"], errors="coerce")
    frame = frame.dropna(subset=["ndc9", "month", "shortage_active"])
    monthly = (frame.groupby(["ndc9", "month"], as_index=False)
               .agg(active_supplier_count=("shortage_active", "sum")))
    return _index_period_values(monthly, ["ndc9"], "active_supplier_count")


@lru_cache(maxsize=4)
def _nadac_lookup(path_string: str) -> dict[str, list[tuple[pd.Period, float]]]:
    """Index historical NADAC prices without using observations after origin."""
    path = Path(path_string)
    if not path.exists():
        return {}
    frame = pd.read_csv(path, usecols=["ndc", "nadac_per_unit", "as_of_date"])
    frame["ndc9"] = frame["ndc"].map(_ndc9)
    frame["month"] = pd.to_datetime(frame["as_of_date"], errors="coerce").dt.to_period("M")
    frame["price"] = pd.to_numeric(frame["nadac_per_unit"], errors="coerce")
    frame = frame[frame["ndc9"].ne("") & frame["month"].notna() & frame["price"].gt(0)]
    monthly = (frame.groupby(["ndc9", "month"], as_index=False)["price"].mean())
    return _index_period_values(monthly, ["ndc9"], "price")


@lru_cache(maxsize=4)
def _recall_lookups(path_string: str) -> dict[str, list[tuple[pd.Period, pd.Period, pd.Period, float]]]:
    path = Path(path_string)
    if not path.exists():
        return {}
    events = load_recall_events([path])
    indexed: dict[str, list[tuple[pd.Period, pd.Period, pd.Period, float]]] = {}
    for row in events.itertuples(index=False):
        indexed.setdefault(str(row.ndc), []).append(
            (row.start, row.end, row.reported, float(row.severity)))
    return indexed


def _recall_state_before(index: dict[str, list[tuple[pd.Period, pd.Period, pd.Period, float]]],
                         ndc: str, origin: pd.Period) -> float:
    events = index.get(ndc, [])
    seen = False
    state = 0.0
    for start, end, reported, severity in events:
        if reported <= origin:
            seen = True
            if start <= origin and (pd.isna(end) or end > origin):
                state = max(state, severity)
    return state if seen else np.nan


@lru_cache(maxsize=4)
def _nssp_lookup(path_string: str) -> dict[str, list[tuple[pd.Period, float]]]:
    """Build point-in-time five-state NSSP values from prior weekly history."""
    path = Path(path_string)
    if not path.exists():
        return {}
    frame = pd.read_json(path)
    required = {"week_end", "county", "percent_visits_influenza"}
    if not required.issubset(frame.columns):
        return {}
    frame = frame[frame["county"].astype(str).eq("All")].copy()
    frame["week"] = pd.to_datetime(frame["week_end"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["percent_visits_influenza"], errors="coerce")
    frame = frame.dropna(subset=["week", "value"]).sort_values("week")
    states = []
    history: list[float] = []
    for row in frame.itertuples(index=False):
        state = np.nan
        if len(history) >= 52:
            thresholds = np.quantile(history, np.arange(1, 5) / 5)
            if len(np.unique(thresholds)) == 4:
                state = float(np.digitize(float(row.value), thresholds))
        states.append((pd.Timestamp(row.week).to_period("M"), state))
        history.append(float(row.value))
    return {"state": states}


@lru_cache(maxsize=4)
def _respnet_lookup(path_string: str) -> dict[str, list[tuple[pd.Period, float]]]:
    """Build point-in-time national RSV states from prior weekly history."""
    path = Path(path_string)
    if not path.exists():
        return {}
    try:
        frame = load_respnet_rsv_weekly(path)
    except (TypeError, ValueError):
        return {}
    frame = frame.sort_values("week_end")
    states = []
    history: list[float] = []
    for row in frame.itertuples(index=False):
        value = float(row.current_value)
        state = np.nan
        if len(history) >= 52:
            thresholds = np.quantile(history, np.arange(1, 5) / 5)
            if len(np.unique(thresholds)) == 4:
                state = float(np.digitize(value, thresholds))
        states.append((pd.Timestamp(row.week_end).to_period("M"), state))
        history.append(value)
    return {"state": states}


def build_historical_metric_context(root: Path, panel: pd.DataFrame) -> tuple[dict[tuple[int, int, str], np.ndarray], dict]:
    """Build per-training-row context and a source coverage report.

    The returned vectors use the same value-then-missingness ordering accepted
    by the modular temporal layer. FDA shortage, NADAC, and recall sources are
    exact NDC joins. The regional and ARCOS metrics remain missing until safe
    historical adapters are implemented; ARCOS uses ZIP3/controlled-drug keys
    rather than NDCs. No current forecast output is used as a training feature.
    """
    shortage_path = root / "data/targeted_additions/fda_shortage_archive/data/fda_shortage_monthly.csv"
    nadac_path = root / "data/targeted_additions/cms_nadac_historical/data/nadac_arkansas_exposed_2021_2025.csv.gz"
    recall_path = root / "data/demand_price/data/openfda_enforcement_2012_2022.csv.gz"
    nssp_path = root / "data/targeted_additions/cdc_nssp_ar_current/data/cdc_nssp_ar_weekly.json"
    respnet_path = root / "data/targeted_additions/cdc_respnet_rsv_current/data/respnet_rsv_overall_weekly.json"
    shortage = _shortage_lookup(str(shortage_path))
    nadac = _nadac_lookup(str(nadac_path))
    recall = _recall_lookups(str(recall_path))
    nssp = _nssp_lookup(str(nssp_path))
    respnet = _respnet_lookup(str(respnet_path))
    result: dict[tuple[int, int, str], np.ndarray] = {}
    coverage = {name: 0 for name in METRIC_CONTEXT_FEATURES}
    for row in panel.itertuples(index=False):
        key = (int(row.year), int(row.quarter), str(row.drug))
        end = _quarter_end(row.year, row.quarter)
        ndc = _ndc9(getattr(row, "ndc", ""))
        values = np.full(len(METRIC_CONTEXT_FEATURES), np.nan, dtype=np.float32)
        if ndc:
            values[0] = _state_context(_latest_before(shortage, ndc, end))
            values[1] = _state_context(_latest_before(nadac, ndc, end))
            values[3] = _state_context(_recall_state_before(recall, ndc, end))
        values[6] = _state_context(_latest_before(nssp, "state", end))
        values[7] = _state_context(_latest_before(respnet, "state", end))
        missing = np.isnan(values).astype(np.float32)
        filled = np.nan_to_num(values, nan=0.0)
        vector = np.asarray([*filled, *missing], dtype=np.float32)
        result[key] = vector
        for index, name in enumerate(METRIC_CONTEXT_FEATURES):
            coverage[name] += int(not missing[index])
    return result, {
        "protocol": "historical_metric_context_v1",
        "feature_order": [*METRIC_CONTEXT_FEATURES,
                          *[f"{name}__missing" for name in METRIC_CONTEXT_FEATURES]],
        "feature_width": METRIC_CONTEXT_WIDTH,
        "sources": {
            "fda_shortage_archive": {
                "path": str(shortage_path), "exists": shortage_path.exists(),
                "point_in_time_basis": "dated Wayback captures and event dates",
            },
            "cms_nadac_historical": {
                "path": str(nadac_path), "exists": nadac_path.exists(),
                "point_in_time_basis": "NADAC as-of date, monthly mean for duplicate observations",
            },
            "fda_enforcement_2012_2022": {
                "path": str(recall_path), "exists": recall_path.exists(),
                "point_in_time_basis": "recall initiation and termination dates",
            },
            "cdc_nssp_ar_current": {
                "path": str(nssp_path), "exists": nssp_path.exists(),
                "point_in_time_basis": "weekly influenza ED percentage; five-state thresholds fit on prior weeks only",
            },
            "cdc_respnet_rsv_current": {
                "path": str(respnet_path), "exists": respnet_path.exists(),
                "point_in_time_basis": "weekly RSV rate; five-state thresholds fit on prior weeks only",
            },
        },
        "coverage_rows": coverage,
        "unavailable_features_remain_missing": True,
        "current_forecast_artifact_used": False,
        "supplier_recall_lookup_rows": 0,
    }
