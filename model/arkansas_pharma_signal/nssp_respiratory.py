"""CDC NSSP Arkansas statewide pathogen-pressure evaluation adapter."""

from __future__ import annotations

import pandas as pd

from .wastewater_pressure import (
    PATHOGENS, build_wastewater_weekly_view, evaluate_wastewater_state_rolling,
)

NSSP_COLUMNS = {
    "covid": "percent_visits_covid",
    "influenza": "percent_visits_influenza",
    "rsv": "percent_visits_rsv",
}
NSSP_PATHOGENS = {
    "covid": "SARS-CoV-2",
    "influenza": "Influenza A virus",
    "rsv": "RSV",
}


def build_nssp_weekly_view(source: pd.DataFrame, *, pathogen: str) -> pd.DataFrame:
    """Convert statewide NSSP ED percentages to the shared weekly view."""
    if pathogen not in NSSP_COLUMNS:
        raise ValueError(f"unsupported NSSP pathogen: {pathogen}")
    required = {"week_end", "county", NSSP_COLUMNS[pathogen]}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"NSSP data missing columns: {sorted(missing)}")
    statewide = source[source["county"].astype(str).eq("All")].copy()
    statewide = statewide[["week_end", NSSP_COLUMNS[pathogen]]].rename(
        columns={NSSP_COLUMNS[pathogen]: "site_wval"})
    statewide["pathogen_target"] = NSSP_PATHOGENS[pathogen]
    shared_pathogen = "sars_cov_2" if pathogen == "covid" else pathogen
    return build_wastewater_weekly_view(statewide, pathogen=shared_pathogen)


def evaluate_nssp_state_rolling(source: pd.DataFrame, *, pathogen: str) -> dict:
    """Evaluate next-week five-state NSSP pressure with fit-only thresholds."""
    view = build_nssp_weekly_view(source, pathogen=pathogen)
    result = evaluate_wastewater_state_rolling(view)
    result.update({
        "protocol": "rolling_origin_next_week_arkansas_nssp_pathogen_five_state",
        "target": f"arkansas_weekly_nssp_{pathogen}_ed_pressure_state",
        "pathogen": pathogen,
        "region": "AR statewide",
        "scope": "Arkansas statewide NSSP ED pathogen percentage proxy; not pharmacy dispensing",
        "view_rows": int(len(view)),
        "first_week": str(view["week"].min()) if not view.empty else None,
        "last_target_week": str(view["next_week"].max()) if not view.empty else None,
    })
    return result
