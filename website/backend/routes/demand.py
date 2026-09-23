"""Account-scoped sales, inventory, forecasting, and signal endpoints."""
from __future__ import annotations

import hashlib
import gzip
import io
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from ml.demand_forecast import DemandDataError, DemandForecaster, build_replenishment_plan, validate_demand_history
from .auth import User, get_current_user
from ..config import settings

router = APIRouter()
MODEL_VERSION = "per_drug_direct_14d_top5_lagged_v3"


def _user_dir(user: User) -> Path:
    # A stable opaque directory prevents usernames from becoming filesystem paths.
    key = hashlib.sha256(user.username.encode()).hexdigest()[:20]
    path = Path(settings.DATA_PATH) / "users" / key
    path.mkdir(parents=True, exist_ok=True)
    return path


def _files(user: User) -> dict[str, Path]:
    root = _user_dir(user)
    return {name: root / filename for name, filename in {
        "sales": "sales_history.csv", "inventory_history": "inventory_history.csv",
        "inventory": "inventory_snapshot.csv", "model": "demand_model.joblib",
        "forecasts": "forecasts.csv", "metrics": "metrics.json",
    }.items()}


async def _read_csv(file: UploadFile, label: str) -> pd.DataFrame:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, detail=f"{label} must be a CSV file")
    payload = await file.read()
    if len(payload) > 50 * 1024 * 1024:
        raise HTTPException(413, detail=f"{label} is larger than the 50 MB demo limit")
    try:
        return pd.read_csv(io.BytesIO(payload))
    except Exception as exc:
        raise HTTPException(400, detail=f"Could not read {label}: {exc}") from exc


def _inventory_snapshot(history: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    required = {"drug_name", "on_hand_units"}
    if not required.issubset(history.columns):
        missing = ", ".join(sorted(required - set(history.columns)))
        raise DemandDataError(f"Inventory CSV is missing required columns: {missing}")
    inventory = history.copy()
    if "date" in inventory:
        inventory["date"] = pd.to_datetime(inventory["date"], errors="coerce")
    elif "as_of_date" in inventory:
        inventory["date"] = pd.to_datetime(inventory["as_of_date"], errors="coerce")
    else:
        inventory["date"] = pd.Timestamp.today().normalize()
    inventory["drug_name"] = inventory["drug_name"].astype(str).str.strip()
    inventory["on_hand_units"] = pd.to_numeric(inventory["on_hand_units"], errors="coerce")
    if "on_order_units" not in inventory:
        inventory["on_order_units"] = 0
    inventory["on_order_units"] = pd.to_numeric(inventory["on_order_units"], errors="coerce")
    if inventory[["date", "drug_name", "on_hand_units", "on_order_units"]].isna().any().any():
        raise DemandDataError("Inventory dates, medication names, on-hand and on-order units must be valid")
    if (inventory[["on_hand_units", "on_order_units"]] < 0).any().any():
        raise DemandDataError("Inventory quantities cannot be negative")
    latest = inventory.sort_values("date").groupby("drug_name", as_index=False).tail(1)
    sale_drugs = set(sales["drug_name"].astype(str))
    inventory_drugs = set(latest["drug_name"])
    missing = sorted(sale_drugs - inventory_drugs)
    if missing:
        raise DemandDataError("Inventory is missing medications from sales history: " + ", ".join(missing[:8]))
    extras = inventory_drugs - sale_drugs
    if extras:
        raise DemandDataError("Inventory contains medications absent from sales history: " + ", ".join(sorted(extras)[:8]))
    lead_days, cover_days = 7, 14
    sales_by_drug = {name: part.sort_values("date") for name, part in sales.groupby("drug_name")}
    result = []
    for record in latest.to_dict(orient="records"):
        drug = record["drug_name"]
        trailing = sales_by_drug[drug].tail(28)["units_sold"].astype(float)
        average, deviation = float(trailing.mean()), float(trailing.std(ddof=0))
        safety = int(np.ceil(1.65 * deviation * np.sqrt(lead_days)))
        result.append({
            "as_of_date": record["date"].date().isoformat(), "drug_name": drug,
            "on_hand_units": int(record["on_hand_units"]), "on_order_units": int(record["on_order_units"]),
            "lead_time_days": lead_days, "safety_stock_units": safety,
            "reorder_point_units": int(np.ceil(average * lead_days + safety)),
            "target_stock_units": int(np.ceil(average * cover_days + safety)),
            "trailing_28d_avg_units": round(average, 2),
        })
    return pd.DataFrame(result)


def _train_and_save(sales: pd.DataFrame, inventory_history: pd.DataFrame, files: dict[str, Path]) -> dict:
    history, _ = validate_demand_history(sales)
    inventory = _inventory_snapshot(inventory_history, history)
    forecaster = DemandForecaster(use_model_signals=True)
    metrics = forecaster.fit(history)
    forecasts = forecaster.forecast(14)
    plan = build_replenishment_plan(inventory, forecasts)
    files["sales"].parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(files["sales"], index=False)
    inventory_history.to_csv(files["inventory_history"], index=False)
    inventory.to_csv(files["inventory"], index=False)
    forecasts.to_csv(files["forecasts"], index=False)
    joblib.dump(forecaster, files["model"])
    payload = {
        **metrics, "model_version": MODEL_VERSION, "trained_at": datetime.now().isoformat(),
        "history_rows": len(history), "inventory_rows": len(inventory), "forecast_rows": len(forecasts),
        "horizon_days": 14, "upload_mode": "user_data",
        "signal_catalog": {"signal_count": forecaster.signals.candidate_count, "source": str(forecaster.signals.source)},
        "planning_summary": {
            "medications": len(plan),
            "reorder_now": int((plan["inventory_status"] == "reorder_now").sum()),
            "total_recommended_order_units": int(plan["recommended_order_units"].sum()),
        },
    }
    files["metrics"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


@router.get("/status")
async def demand_status(user: User = Depends(get_current_user)):
    files = _files(user)
    if not files["metrics"].exists() or not files["model"].exists():
        return {"ready": False, "uploads_enabled": True, "model_version": MODEL_VERSION}
    metrics = json.loads(files["metrics"].read_text(encoding="utf-8"))
    if metrics.get("model_version") != MODEL_VERSION:
        return {"ready": False, "uploads_enabled": True, "model_version": MODEL_VERSION, "retrain_required": True}
    return {"ready": True, "uploads_enabled": True, "metrics": metrics}


@router.post("/train")
async def train_user_model(
    sales_file: UploadFile = File(...), inventory_file: UploadFile = File(...),
    user: User = Depends(get_current_user),
):
    sales = await _read_csv(sales_file, "Sales history")
    inventory = await _read_csv(inventory_file, "Inventory history")
    files = _files(user)
    try:
        return await run_in_threadpool(_train_and_save, sales, inventory, files)
    except (DemandDataError, ValueError) as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, detail=f"Training failed: {exc}") from exc


def _require_trained(user: User) -> tuple[dict[str, Path], dict]:
    files = _files(user)
    if not files["metrics"].exists() or not files["model"].exists():
        raise HTTPException(409, detail="Upload sales and inventory data to train your forecast first")
    metrics = json.loads(files["metrics"].read_text(encoding="utf-8"))
    if metrics.get("model_version") != MODEL_VERSION:
        raise HTTPException(409, detail="This saved model uses an older forecasting method; retrain with your sales and inventory files")
    return files, metrics


def _related_articles(signal_ids: list[str], drugs: list[str], history_end: str) -> dict[str, dict]:
    corpus = Path(__file__).resolve().parents[3] / "data" / "targeted_additions" / "news_article_corpus" / "data" / "articles.jsonl.gz"
    if not corpus.exists():
        return {}
    stop = {"arkansas", "national", "risk", "direction", "monthly", "supply", "demand", "stage", "chain", "policy"}
    topics = {key: [word for word in key.lower().split("_") if len(word) > 3 and word not in stop] for key in signal_ids}
    drugs_lower = [name.split()[0].lower().split("/")[0] for name in drugs]
    best: dict[str, tuple[int, dict]] = {}
    with gzip.open(corpus, "rt", encoding="utf-8") as handle:
        for line in handle:
            try:
                article = json.loads(line)
            except json.JSONDecodeError:
                continue
            title = str(article.get("title") or "")
            url = str(article.get("url") or "")
            if not title or not url.startswith(("http://", "https://")):
                continue
            published = pd.to_datetime(article.get("published_at"), unit="ms", errors="coerce")
            if pd.isna(published) or published.date().isoformat() > history_end:
                continue
            title_lower = title.lower()
            body_lower = str(article.get("body") or "").lower()[:3000]
            for signal_id, words in topics.items():
                if not words:
                    continue
                topic_hits = sum(3 if word in title_lower else 1 if word in body_lower else 0 for word in words)
                if not topic_hits:
                    continue
                drug_hits = sum(4 if drug in title_lower else 2 if drug in body_lower else 0 for drug in drugs_lower if len(drug) > 4)
                score = topic_hits + drug_hits
                if score > best.get(signal_id, (0, {}))[0]:
                    best[signal_id] = (score, {"title": title, "url": url, "source": str(article.get("source") or ""), "published_at": published.date().isoformat(), "match_note": "Related by topic keywords; this article is not the model's attribution source."})
    return {signal_id: item for signal_id, (_, item) in best.items()}


@router.get("/forecasts")
async def forecasts(user: User = Depends(get_current_user)):
    files, metrics = _require_trained(user)
    data = pd.read_csv(files["forecasts"])
    return {"forecasts": data.to_dict(orient="records"), "count": len(data), "metrics": metrics}


@router.get("/planning")
async def planning(user: User = Depends(get_current_user)):
    files, metrics = _require_trained(user)
    plan = build_replenishment_plan(pd.read_csv(files["inventory"]), pd.read_csv(files["forecasts"]))
    plan = plan.astype(object).where(pd.notna(plan), None)
    return {
        "items": plan.to_dict(orient="records"), "count": len(plan),
        "summary": {**metrics["planning_summary"], "forecast_horizon_days": metrics["horizon_days"]},
        "policy": {"lead_time_days": 7, "target_cover_days": 14,
                   "note": "The model predicts 14-day demand directly. Day 1 and 7 are allocations by recent weekday sales; quantities use your latest uploaded on-hand balance."},
    }


@router.get("/signals/relevant")
async def relevant_signals(user: User = Depends(get_current_user)):
    files, _ = _require_trained(user)
    forecaster: DemandForecaster = joblib.load(files["model"])
    forecast_rows = pd.read_csv(files["forecasts"])
    plan = build_replenishment_plan(pd.read_csv(files["inventory"]), forecast_rows)
    inventory_by_drug = pd.read_csv(files["inventory"]).set_index("drug_name")
    oversold = [row.drug_name for row in plan.itertuples()
                if row.forecast_14d_units > inventory_by_drug.loc[row.drug_name, "on_hand_units"] + inventory_by_drug.loc[row.drug_name, "on_order_units"]]
    news_path = Path(__file__).resolve().parents[3] / "model" / "artifacts" / "news" / "news_only_catalog_features.csv.gz"
    foundational = set()
    if news_path.exists():
        foundational = set(pd.read_csv(news_path, nrows=1).columns) - {"date"}
    ranked: dict[str, dict] = {}
    for drug in oversold:
        relevant_for_drug = []
        for column, score in forecaster.signal_scores.get(drug, {}).items():
            metadata = forecaster.signals.metadata.get(column, {})
            signal_id = metadata.get("signal_id", "")
            if metadata.get("signal_origin") != "model_news_output" or signal_id not in foundational:
                continue
            relevant_for_drug.append((signal_id, float(score), metadata))
        for signal_id, score, metadata in sorted(relevant_for_drug, key=lambda row: (-row[1], row[0]))[:3]:
            item = ranked.setdefault(signal_id, {"signal_id": signal_id, "score": 0.0, "drugs": [], "source": metadata.get("source_name", "Foundational news model")})
            item["score"] += float(score)
            item["drugs"].append(drug)
    for item in ranked.values():
        item["score"] = round(item["score"] / len(item["drugs"]), 4)
    results = sorted(ranked.values(), key=lambda row: (-row["score"], row["signal_id"]))
    articles = _related_articles([item["signal_id"] for item in results], oversold, forecaster.metrics["history_end"])
    source_rows = forecaster.signals.rows
    broader = forecast_rows.groupby("horizon_day", as_index=False)["predicted_units"].sum().sort_values("horizon_day")
    broader["cumulative_units"] = broader["predicted_units"].cumsum()
    demand_context = {
        "scope": "All medications in this account",
        "method": "Sum of saved per-drug forecasts; not the foundational signal's own forecast",
        "points": [{"day": int(row.horizon_day), "predicted_units": round(float(row.predicted_units), 2),
                    "cumulative_units": round(float(row.cumulative_units), 2)} for row in broader.itertuples()],
    }
    for item in results:
        item["kind"] = "demand" if "demand" in item["signal_id"] else "news"
        selected_rows = source_rows[
            (source_rows["signal_id"] == item["signal_id"])
            & (source_rows["signal_origin"] == "model_news_output")
        ] if not source_rows.empty else pd.DataFrame()
        if not selected_rows.empty:
            trend = selected_rows.groupby("period_end", as_index=False)["value"].mean().tail(24)
            item["trend"] = [{"date": str(row.period_end)[:10], "value": float(row.value)} for row in trend.itertuples()]
        else:
            item["trend"] = []
        item["article"] = articles.get(item["signal_id"]) if item["kind"] == "news" else None
        item["demand_context"] = demand_context if item["kind"] == "demand" else None
    # News cards need an actual headline; skip unmatched scores and backfill
    # from lower-ranked foundational signals where a related article exists.
    displayable = [item for item in results if item["kind"] == "demand" or item["article"]]
    return {"at_risk_drugs": oversold, "signals": displayable[:8],
            "note": "Each oversold drug contributes its three strongest foundational signals by absolute training-period correlation. Demand charts show the separate aggregate pharmacy forecast; related news articles are topic matches, not model attribution."}
