import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from .signal_store import connect, initialize
from model.prod_pipeline import ProductionModel

SOURCE_KIND = "combined_demand_signals"

def refresh_combined_model() -> dict:
    initialize()
    started = datetime.now(timezone.utc)
    started_text = started.isoformat()
    
    with connect() as db:
        run_id = db.execute("INSERT INTO refresh_runs(source_name, started_at, status) VALUES (?, ?, 'running')", (SOURCE_KIND, started_text)).lastrowid
        
    try:
        root_dir = Path(__file__).resolve().parent.parent.parent
        
        # Fetch the latest 20 news signals from the DB
        with connect() as db:
            news_rows = db.execute("""
                SELECT d.signal_id, r.value 
                FROM signal_records r 
                JOIN signal_definitions d ON d.id = r.signal_uid 
                WHERE r.source_kind = 'article_text_ridge_shadow_v1' 
                  AND r.observation_date = (
                      SELECT MAX(observation_date) FROM signal_records WHERE source_kind = 'article_text_ridge_shadow_v1'
                  )
            """).fetchall()
            
        if not news_rows:
            raise ValueError("No live news signals found in DB. Run refresh_news_model first.")
            
        live_news = {row["signal_id"]: row["value"] for row in news_rows}
        
        # Instantiate and train model
        model = ProductionModel(root_dir)
        model.train()
        
        # Predict using live news
        predictions = model.predict(live_news=live_news)
        
        # Write the 1300+ CMS and 18 ATC signals to the DB
        finished = datetime.now(timezone.utc)
        finished_text = finished.isoformat()
        
        with connect() as db:
            definitions = {row["signal_id"]: row["id"] for row in db.execute("SELECT id, signal_id FROM signal_definitions")}
            
        rows_to_insert = []
        
        # ATC
        for atc, val in predictions["arkansas_atc_signals_18"].items():
            sig_id = f"arkansas_atc_demand_state::{atc}"
            if sig_id in definitions:
                uid = definitions[sig_id]
                row_hash = hashlib.sha256(f"{SOURCE_KIND}:{uid}:{finished_text}".encode()).hexdigest()
                # Use current year + 1 as horizon to outrank baseline
                horizon = str(finished.year + 1)
                rows_to_insert.append((row_hash, uid, finished.date().isoformat(), finished.date().isoformat(), float(val["predicted_demand_state"]), finished_text, finished_text, "combined_production_model", horizon, SOURCE_KIND, ""))
                
        # CMS
        for drug_key, val in predictions["cms_part_d_signals_1300"].items():
            sig_id = f"cms_part_d_demand_state::{drug_key}"
            if sig_id in definitions:
                uid = definitions[sig_id]
                row_hash = hashlib.sha256(f"{SOURCE_KIND}:{uid}:{finished_text}".encode()).hexdigest()
                horizon = str(finished.year + 1)
                rows_to_insert.append((row_hash, uid, finished.date().isoformat(), finished.date().isoformat(), float(val["predicted_demand_state"]), finished_text, finished_text, "combined_production_model", horizon, SOURCE_KIND, ""))
                
        with connect() as db:
            before = db.total_changes
            db.executemany("""
                INSERT OR IGNORE INTO signal_records
                (row_hash, signal_uid, signal_date, observation_date, value, source_timestamp, ingested_at, data_quality, forecast_horizon, source_kind, source_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows_to_insert)
            written = db.total_changes - before
            
            db.execute("UPDATE refresh_runs SET finished_at=?, status='success', rows_written=? WHERE id=?", (finished_text, written, run_id))
            
        return {"source": SOURCE_KIND, "status": "success", "rows_written": written}
        
    except Exception as exc:
        with connect() as db:
            db.execute("UPDATE refresh_runs SET finished_at=?, status='failed', error=? WHERE id=?", (datetime.now(timezone.utc).isoformat(), str(exc)[:500], run_id))
        raise
