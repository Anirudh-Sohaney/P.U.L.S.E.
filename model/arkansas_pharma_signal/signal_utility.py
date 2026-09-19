"""Cross-signal incremental-utility experiment for public health covariates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .regression import LogisticRidge
from .hospital_respiratory import load_hospital_respiratory_weekly
from .weekly_health_proxy import FEATURE_COLUMNS, load_fluview_weekly_proxy


BASE_FEATURES = list(FEATURE_COLUMNS)
AUGMENTED_FEATURES = [*BASE_FEATURES, "wastewater_influenza_mean"]
HOSPITAL_AUGMENTED_FEATURES = [*BASE_FEATURES, "hospital_influenza_admissions"]
WEATHER_FEATURES = [
    "weather_temperature_mean", "weather_temperature_min",
    "weather_temperature_max", "weather_precipitation", "weather_snowfall",
    "weather_coverage_ratio",
]
WEATHER_AUGMENTED_FEATURES = [*BASE_FEATURES, *WEATHER_FEATURES]
DISASTER_FEATURES = ["disaster_active_count", "disaster_severity_max", "disaster_counties"]
DISASTER_AUGMENTED_FEATURES = [*BASE_FEATURES, *DISASTER_FEATURES]


def build_fluview_wastewater_view(fluview: pd.DataFrame,
                                  wastewater: pd.DataFrame) -> pd.DataFrame:
    """Align current-week wastewater with next-week FluView targets."""
    flu = (load_fluview_weekly_proxy(fluview)
           if {"wili", "epiweek"}.issubset(fluview.columns) else fluview.copy())
    if "region" in flu.columns:
        flu = flu[flu["region"].astype(str).str.lower().eq("ar")].copy()
    water = wastewater.copy()
    required = {"week_end", "pathogen_target", "site_wval"}
    missing = required.difference(water.columns)
    if missing:
        raise ValueError(f"wastewater data missing columns: {sorted(missing)}")
    water = water[water["pathogen_target"].eq("Influenza A virus")].copy()
    water["week_start"] = (pd.to_datetime(water["week_end"], errors="coerce")
                            - pd.Timedelta(days=5))
    water["site_wval"] = pd.to_numeric(water["site_wval"], errors="coerce")
    water = (water.dropna(subset=["week_start", "site_wval"])
             .groupby("week_start", as_index=False)["site_wval"].mean()
             .rename(columns={"site_wval": "wastewater_influenza_mean"}))
    result = flu.merge(water, on="week_start", how="inner")
    return result.sort_values("week_start").reset_index(drop=True)


def _state(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.digitize(np.asarray(values, dtype=float), [lower, upper]).astype(int)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=int); predicted = np.asarray(predicted, dtype=int)
    recalls = []
    for state in range(3):
        mask = actual == state
        recalls.append(float(np.mean(predicted[mask] == state)) if mask.any() else 0.0)
    return {"accuracy": float(np.mean(actual == predicted)),
            "balanced_accuracy": float(np.mean(recalls)),
            "state_counts": {str(x): int(np.sum(actual == x)) for x in range(3)}}


def _predict(train: pd.DataFrame, target: pd.DataFrame, features: list[str],
             alpha: float, lower: float, upper: float) -> np.ndarray:
    train_state = _state(train["target"].to_numpy(float), lower, upper)
    probabilities = []
    for state in range(3):
        model = LogisticRidge(alpha=alpha).fit(
            train[features].fillna(0).to_numpy(float),
            (train_state == state).astype(float), features)
        probabilities.append(model.predict_proba(
            target[features].fillna(0).to_numpy(float)))
    return np.column_stack(probabilities).argmax(axis=1)


def evaluate_wastewater_incremental_utility(view: pd.DataFrame) -> dict[str, Any]:
    """Compare FluView state prediction with and without wastewater features."""
    folds = []
    for train_year, validation_year, test_year in (
            (2022, 2023, 2024), (2023, 2024, 2025), (2024, 2025, 2026)):
        train = view[view.year <= train_year]
        validation = view[view.year == validation_year]
        test = view[view.year == test_year]
        if train.empty or validation.empty or test.empty:
            continue
        lower, upper = np.quantile(train["target"].to_numpy(float), [1 / 3, 2 / 3])
        val_state = _state(validation["target"], lower, upper)
        candidates = {}
        for features in (BASE_FEATURES, AUGMENTED_FEATURES):
            scores = {}
            for alpha in (0.1, 1.0, 10.0, 100.0):
                scores[alpha] = _metrics(
                    val_state, _predict(train, validation, features, alpha, lower, upper)
                )["balanced_accuracy"]
            alpha = max(scores, key=scores.get)
            candidates[tuple(features)] = alpha
        actual = _state(test["target"], lower, upper)
        results = {}
        for features in (BASE_FEATURES, AUGMENTED_FEATURES):
            results["augmented" if features == AUGMENTED_FEATURES else "base"] = _metrics(
                actual, _predict(pd.concat([train, validation]), test, features,
                                 candidates[tuple(features)], lower, upper))
        folds.append({"train_year": train_year, "validation_year": validation_year,
                      "test_year": test_year, "test_rows": int(len(test)),
                      "base": results["base"], "augmented": results["augmented"],
                      "balanced_accuracy_delta": results["augmented"]["balanced_accuracy"]
                      - results["base"]["balanced_accuracy"]})
    if not folds:
        return {"fold_count": 0, "publishable_candidate": False, "folds": []}
    deltas = [fold["balanced_accuracy_delta"] for fold in folds]
    return {
        "protocol": "rolling_origin_fluview_state_wastewater_incremental_utility",
        "fold_count": len(folds), "test_rows": int(sum(x["test_rows"] for x in folds)),
        "mean_base_accuracy": float(np.mean([x["base"]["accuracy"] for x in folds])),
        "mean_augmented_accuracy": float(np.mean([x["augmented"]["accuracy"] for x in folds])),
        "mean_base_balanced_accuracy": float(np.mean([
            x["base"]["balanced_accuracy"] for x in folds])),
        "mean_augmented_balanced_accuracy": float(np.mean([
            x["augmented"]["balanced_accuracy"] for x in folds])),
        "mean_balanced_accuracy_delta": float(np.mean(deltas)),
        "all_folds_improve": bool(all(delta > 0 for delta in deltas)),
        "publishable_candidate": False,
        "scope": "utility experiment only; neither target is pharmacy dispensing truth",
        "folds": folds,
    }


def build_fluview_hospital_view(
    fluview: pd.DataFrame, hospital: pd.DataFrame
) -> pd.DataFrame:
    """Align current-week hospital influenza admissions with FluView rows."""
    flu = (load_fluview_weekly_proxy(fluview)
           if {"wili", "epiweek"}.issubset(fluview.columns) else fluview.copy())
    required = {"weekendingdate", "jurisdiction", "totalconfflunewadm"}
    missing = required.difference(hospital.columns)
    if missing:
        raise ValueError(f"hospital data missing columns: {sorted(missing)}")
    frame = hospital[hospital["jurisdiction"].astype(str).str.upper().eq("AR")].copy()
    frame["week_start"] = (pd.to_datetime(frame["weekendingdate"], errors="coerce")
                            - pd.Timedelta(days=5))
    frame["hospital_influenza_admissions"] = pd.to_numeric(
        frame["totalconfflunewadm"], errors="coerce")
    frame = frame.dropna(subset=["week_start", "hospital_influenza_admissions"])
    frame = frame.drop_duplicates("week_start", keep="last")
    return (flu.merge(frame[["week_start", "hospital_influenza_admissions"]],
                      on="week_start", how="inner")
            .sort_values("week_start").reset_index(drop=True))


def evaluate_hospital_incremental_utility(view: pd.DataFrame) -> dict[str, Any]:
    """Compare FluView state forecasts with and without hospital admissions."""
    folds = []
    for train_year, validation_year, test_year in (
            (2021, 2022, 2023), (2022, 2023, 2024),
            (2023, 2024, 2025), (2024, 2025, 2026)):
        train = view[view.year <= train_year]
        validation = view[view.year == validation_year]
        test = view[view.year == test_year]
        if train.empty or validation.empty or test.empty:
            continue
        lower, upper = np.quantile(train["target"].to_numpy(float), [1 / 3, 2 / 3])
        val_state = _state(validation["target"], lower, upper)
        candidates = {}
        for features in (BASE_FEATURES, HOSPITAL_AUGMENTED_FEATURES):
            scores = {}
            for alpha in (0.1, 1.0, 10.0, 100.0):
                scores[alpha] = _metrics(
                    val_state, _predict(train, validation, features, alpha, lower, upper)
                )["balanced_accuracy"]
            candidates[tuple(features)] = max(scores, key=scores.get)
        actual = _state(test["target"], lower, upper)
        results = {}
        for features in (BASE_FEATURES, HOSPITAL_AUGMENTED_FEATURES):
            results["augmented" if features == HOSPITAL_AUGMENTED_FEATURES else "base"] = _metrics(
                actual, _predict(pd.concat([train, validation]), test, features,
                                 candidates[tuple(features)], lower, upper))
        folds.append({
            "train_year": train_year, "validation_year": validation_year,
            "test_year": test_year, "test_rows": int(len(test)),
            "base": results["base"], "augmented": results["augmented"],
            "balanced_accuracy_delta": results["augmented"]["balanced_accuracy"]
            - results["base"]["balanced_accuracy"],
        })
    if not folds:
        return {"fold_count": 0, "publishable_candidate": False, "folds": []}
    deltas = [fold["balanced_accuracy_delta"] for fold in folds]
    return {
        "protocol": "rolling_origin_fluview_state_hospital_admission_incremental_utility",
        "fold_count": len(folds), "test_rows": int(sum(x["test_rows"] for x in folds)),
        "mean_base_accuracy": float(np.mean([x["base"]["accuracy"] for x in folds])),
        "mean_augmented_accuracy": float(np.mean([x["augmented"]["accuracy"] for x in folds])),
        "mean_base_balanced_accuracy": float(np.mean([x["base"]["balanced_accuracy"] for x in folds])),
        "mean_augmented_balanced_accuracy": float(np.mean([x["augmented"]["balanced_accuracy"] for x in folds])),
        "mean_balanced_accuracy_delta": float(np.mean(deltas)),
        "all_folds_improve": bool(all(delta > 0 for delta in deltas)),
        "publishable_candidate": False,
        "scope": "utility experiment only; hospital and FluView targets are not pharmacy dispensing truth",
        "folds": folds,
    }


def build_fluview_weather_view(
    fluview: pd.DataFrame, weather: pd.DataFrame,
) -> pd.DataFrame:
    """Align current-week observed county weather to next-week FluView."""
    flu = (load_fluview_weekly_proxy(fluview)
           if {"wili", "epiweek"}.issubset(fluview.columns) else fluview.copy())
    if "region" in flu.columns:
        flu = flu[flu["region"].astype(str).str.lower().eq("ar")].copy()
    weather = weather.copy()
    weather = weather.rename(columns={
        "temperature_mean": "weather_temperature_mean",
        "temperature_min": "weather_temperature_min",
        "temperature_max": "weather_temperature_max",
        "precipitation": "weather_precipitation",
        "snowfall": "weather_snowfall",
    })
    required = {"cadence", "period_start", "county_fips", *WEATHER_FEATURES}
    missing = required.difference(weather.columns)
    if missing:
        raise ValueError(f"weather data missing columns: {sorted(missing)}")
    current = weather[weather["cadence"].eq("weekly")].copy()
    current["period_start"] = pd.to_datetime(current["period_start"], errors="coerce").dt.normalize()
    current = current.dropna(subset=["period_start"])
    # Equal-county aggregation avoids letting a station with a larger number
    # of duplicate rows dominate the statewide context.
    current = (current.groupby("period_start", as_index=False)[WEATHER_FEATURES]
               .mean().rename(columns={"period_start": "week_start"}))
    result = flu.merge(current, on="week_start", how="inner", validate="one_to_one")
    return result.sort_values("week_start").reset_index(drop=True)


def evaluate_weather_incremental_utility(view: pd.DataFrame) -> dict[str, Any]:
    """Test whether current-week weather improves next-week FluView states."""
    folds = []
    for train_year, validation_year, test_year in (
            (2018, 2019, 2020), (2019, 2020, 2021), (2020, 2021, 2022),
            (2021, 2022, 2023), (2022, 2023, 2024), (2023, 2024, 2025),
            (2024, 2025, 2026)):
        train = view[view.year <= train_year]
        validation = view[view.year == validation_year]
        test = view[view.year == test_year]
        if train.empty or validation.empty or test.empty:
            continue
        lower, upper = np.quantile(train["target"].to_numpy(float), [1 / 3, 2 / 3])
        val_state = _state(validation["target"], lower, upper)
        candidates = {}
        for features in (BASE_FEATURES, WEATHER_AUGMENTED_FEATURES):
            scores = {}
            for alpha in (0.1, 1.0, 10.0, 100.0):
                scores[alpha] = _metrics(
                    val_state, _predict(train, validation, features, alpha, lower, upper)
                )["balanced_accuracy"]
            candidates[tuple(features)] = max(scores, key=scores.get)
        actual = _state(test["target"], lower, upper)
        results = {}
        for features in (BASE_FEATURES, WEATHER_AUGMENTED_FEATURES):
            key = "augmented" if features == WEATHER_AUGMENTED_FEATURES else "base"
            results[key] = _metrics(
                actual, _predict(pd.concat([train, validation]), test, features,
                                 candidates[tuple(features)], lower, upper))
        folds.append({
            "train_year": train_year, "validation_year": validation_year,
            "test_year": test_year, "test_rows": int(len(test)),
            "base": results["base"], "augmented": results["augmented"],
            "balanced_accuracy_delta": results["augmented"]["balanced_accuracy"]
            - results["base"]["balanced_accuracy"],
        })
    if not folds:
        return {"fold_count": 0, "publishable_candidate": False, "folds": []}
    deltas = [fold["balanced_accuracy_delta"] for fold in folds]
    return {
        "protocol": "rolling_origin_fluview_state_weather_incremental_utility",
        "fold_count": len(folds), "test_rows": int(sum(x["test_rows"] for x in folds)),
        "weather_features": WEATHER_FEATURES,
        "mean_base_accuracy": float(np.mean([x["base"]["accuracy"] for x in folds])),
        "mean_augmented_accuracy": float(np.mean([x["augmented"]["accuracy"] for x in folds])),
        "mean_base_balanced_accuracy": float(np.mean([
            x["base"]["balanced_accuracy"] for x in folds])),
        "mean_augmented_balanced_accuracy": float(np.mean([
            x["augmented"]["balanced_accuracy"] for x in folds])),
        "mean_balanced_accuracy_delta": float(np.mean(deltas)),
        "all_folds_improve": bool(all(delta > 0 for delta in deltas)),
        "publishable_candidate": False,
        "scope": "weather utility experiment; FluView is not pharmacy dispensing truth",
        "folds": folds,
    }


def build_hospital_weather_view(
    hospital: pd.DataFrame, weather: pd.DataFrame,
) -> pd.DataFrame:
    """Align current-week weather to next-week hospital respiratory targets."""
    admissions = (load_hospital_respiratory_weekly(hospital)
                  if {"weekendingdate", "jurisdiction"}.issubset(hospital.columns)
                  else hospital.copy())
    weather = weather.copy().rename(columns={
        "temperature_mean": "weather_temperature_mean",
        "temperature_min": "weather_temperature_min",
        "temperature_max": "weather_temperature_max",
        "precipitation": "weather_precipitation",
        "snowfall": "weather_snowfall",
    })
    required = {"cadence", "period_start", "county_fips", *WEATHER_FEATURES}
    missing = required.difference(weather.columns)
    if missing:
        raise ValueError(f"weather data missing columns: {sorted(missing)}")
    current = weather[weather["cadence"].eq("weekly")].copy()
    current["period_start"] = pd.to_datetime(current["period_start"], errors="coerce").dt.normalize()
    current = (current.dropna(subset=["period_start"])
               .groupby("period_start", as_index=False)[WEATHER_FEATURES].mean())
    admissions = admissions.copy()
    admissions["period_start"] = (pd.to_datetime(admissions["week_end"], errors="coerce")
                                   - pd.Timedelta(days=5)).dt.normalize()
    result = admissions.merge(current, on="period_start", how="inner", validate="many_to_one")
    return result.sort_values(["pathogen", "week_end"]).reset_index(drop=True)


def evaluate_hospital_weather_incremental_utility(view: pd.DataFrame) -> dict[str, Any]:
    """Test whether current-week weather improves next-week influenza states."""
    folds = []
    data = view[view["pathogen"].eq("influenza")].sort_values("week_end")
    for train_year, validation_year, test_year in (
            (2021, 2022, 2023), (2022, 2023, 2024),
            (2023, 2024, 2025), (2024, 2025, 2026)):
        train = data[data.year <= train_year]
        validation = data[data.year.eq(validation_year)]
        test = data[data.year.eq(test_year)]
        if train.empty or validation.empty or test.empty:
            continue
        lower, upper = np.quantile(train["target"].to_numpy(float), [1 / 3, 2 / 3])
        val_state = _state(validation["target"], lower, upper)
        candidates = {}
        for features in (BASE_FEATURES, WEATHER_AUGMENTED_FEATURES):
            scores = {}
            for alpha in (0.1, 1.0, 10.0, 100.0):
                scores[alpha] = _metrics(
                    val_state, _predict(train, validation, features, alpha, lower, upper)
                )["balanced_accuracy"]
            candidates[tuple(features)] = max(scores, key=scores.get)
        actual = _state(test["target"], lower, upper)
        results = {}
        for features in (BASE_FEATURES, WEATHER_AUGMENTED_FEATURES):
            key = "augmented" if features == WEATHER_AUGMENTED_FEATURES else "base"
            results[key] = _metrics(
                actual, _predict(pd.concat([train, validation]), test, features,
                                 candidates[tuple(features)], lower, upper))
        folds.append({
            "train_year": train_year, "validation_year": validation_year,
            "test_year": test_year, "test_rows": int(len(test)),
            "base": results["base"], "augmented": results["augmented"],
            "balanced_accuracy_delta": results["augmented"]["balanced_accuracy"]
            - results["base"]["balanced_accuracy"],
        })
    if not folds:
        return {"fold_count": 0, "publishable_candidate": False, "folds": []}
    deltas = [fold["balanced_accuracy_delta"] for fold in folds]
    return {
        "protocol": "rolling_origin_hospital_influenza_state_weather_incremental_utility",
        "target": "arkansas_weekly_hospital_influenza_admission_pressure_state",
        "fold_count": len(folds), "test_rows": int(sum(x["test_rows"] for x in folds)),
        "weather_features": WEATHER_FEATURES,
        "mean_base_accuracy": float(np.mean([x["base"]["accuracy"] for x in folds])),
        "mean_augmented_accuracy": float(np.mean([x["augmented"]["accuracy"] for x in folds])),
        "mean_base_balanced_accuracy": float(np.mean([
            x["base"]["balanced_accuracy"] for x in folds])),
        "mean_augmented_balanced_accuracy": float(np.mean([
            x["augmented"]["balanced_accuracy"] for x in folds])),
        "mean_balanced_accuracy_delta": float(np.mean(deltas)),
        "all_folds_improve": bool(all(delta > 0 for delta in deltas)),
        "publishable_candidate": False,
        "scope": "weather utility experiment; hospital target is not pharmacy dispensing truth",
        "folds": folds,
    }


def build_hospital_disaster_view(
    hospital: pd.DataFrame, disasters: pd.DataFrame,
) -> pd.DataFrame:
    """Align current-week FEMA county declarations to next-week admissions."""
    admissions = (load_hospital_respiratory_weekly(hospital)
                  if {"weekendingdate", "jurisdiction"}.issubset(hospital.columns)
                  else hospital.copy())
    required = {"variable_id", "geography_id", "observation_time", "value"}
    missing = required.difference(disasters.columns)
    if missing:
        raise ValueError(f"disaster data missing columns: {sorted(missing)}")
    events = disasters[disasters["variable_id"].eq("disaster_active")].copy()
    events["observation_time"] = pd.to_datetime(events["observation_time"], errors="coerce")
    events["value"] = pd.to_numeric(events["value"], errors="coerce")
    events = events.dropna(subset=["observation_time", "value"])
    events["period_start"] = (events["observation_time"]
                               - pd.to_timedelta(events["observation_time"].dt.weekday, unit="D"))
    counts = (events.groupby("period_start", as_index=False)
              .agg(disaster_active_count=("value", "sum"),
                   disaster_counties=("geography_id", "nunique")))
    severity = disasters[disasters["variable_id"].eq("disaster_severity")].copy()
    severity["observation_time"] = pd.to_datetime(severity["observation_time"], errors="coerce")
    severity["value"] = pd.to_numeric(severity["value"], errors="coerce")
    severity = severity.dropna(subset=["observation_time", "value"])
    severity["period_start"] = (severity["observation_time"]
                                 - pd.to_timedelta(severity["observation_time"].dt.weekday, unit="D"))
    severity = severity.groupby("period_start", as_index=False)["value"].max().rename(
        columns={"value": "disaster_severity_max"})
    event_features = counts.merge(severity, on="period_start", how="outer").fillna(0)
    admissions = admissions.copy()
    admissions["period_start"] = (pd.to_datetime(admissions["week_end"], errors="coerce")
                                   - pd.Timedelta(days=5)).dt.normalize()
    result = admissions.merge(event_features, on="period_start", how="left", validate="many_to_one")
    result[DISASTER_FEATURES] = result[DISASTER_FEATURES].fillna(0.0)
    return result.sort_values(["pathogen", "week_end"]).reset_index(drop=True)


def evaluate_hospital_disaster_incremental_utility(view: pd.DataFrame) -> dict[str, Any]:
    """Test whether current-week FEMA pressure improves influenza states."""
    folds = []
    data = view[view["pathogen"].eq("influenza")].sort_values("week_end")
    for train_year, validation_year, test_year in (
            (2021, 2022, 2023), (2022, 2023, 2024),
            (2023, 2024, 2025), (2024, 2025, 2026)):
        train = data[data.year <= train_year]
        validation = data[data.year.eq(validation_year)]
        test = data[data.year.eq(test_year)]
        if train.empty or validation.empty or test.empty:
            continue
        lower, upper = np.quantile(train["target"].to_numpy(float), [1 / 3, 2 / 3])
        val_state = _state(validation["target"], lower, upper)
        candidates = {}
        for features in (BASE_FEATURES, DISASTER_AUGMENTED_FEATURES):
            scores = {}
            for alpha in (0.1, 1.0, 10.0, 100.0):
                scores[alpha] = _metrics(
                    val_state, _predict(train, validation, features, alpha, lower, upper)
                )["balanced_accuracy"]
            candidates[tuple(features)] = max(scores, key=scores.get)
        actual = _state(test["target"], lower, upper)
        results = {}
        for features in (BASE_FEATURES, DISASTER_AUGMENTED_FEATURES):
            key = "augmented" if features == DISASTER_AUGMENTED_FEATURES else "base"
            results[key] = _metrics(
                actual, _predict(pd.concat([train, validation]), test, features,
                                 candidates[tuple(features)], lower, upper))
        folds.append({
            "train_year": train_year, "validation_year": validation_year,
            "test_year": test_year, "test_rows": int(len(test)),
            "base": results["base"], "augmented": results["augmented"],
            "balanced_accuracy_delta": results["augmented"]["balanced_accuracy"]
            - results["base"]["balanced_accuracy"],
        })
    if not folds:
        return {"fold_count": 0, "publishable_candidate": False, "folds": []}
    deltas = [fold["balanced_accuracy_delta"] for fold in folds]
    return {
        "protocol": "rolling_origin_hospital_influenza_state_fema_disaster_incremental_utility",
        "target": "arkansas_weekly_hospital_influenza_admission_pressure_state",
        "fold_count": len(folds), "test_rows": int(sum(x["test_rows"] for x in folds)),
        "disaster_features": DISASTER_FEATURES,
        "mean_base_accuracy": float(np.mean([x["base"]["accuracy"] for x in folds])),
        "mean_augmented_accuracy": float(np.mean([x["augmented"]["accuracy"] for x in folds])),
        "mean_base_balanced_accuracy": float(np.mean([
            x["base"]["balanced_accuracy"] for x in folds])),
        "mean_augmented_balanced_accuracy": float(np.mean([
            x["augmented"]["balanced_accuracy"] for x in folds])),
        "mean_balanced_accuracy_delta": float(np.mean(deltas)),
        "all_folds_improve": bool(all(delta > 0 for delta in deltas)),
        "publishable_candidate": False,
        "scope": "FEMA event utility experiment; hospital target is not pharmacy dispensing truth",
        "folds": folds,
    }
