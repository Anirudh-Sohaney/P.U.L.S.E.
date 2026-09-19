"""Shared helpers for the S_D pipeline."""
import json, os, re, hashlib, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
BY_SOURCE = os.path.join(ROOT, "data", "by_source")
COMBINED = os.path.join(ROOT, "data", "combined")
DATA = os.path.join(ROOT, "data")

def ensure_dirs():
    for d in [CACHE, BY_SOURCE, COMBINED]:
        os.makedirs(d, exist_ok=True)

def today():
    return datetime.date.today().isoformat()

def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)

def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def sha1(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def norm_ndc(ndc):
    """Normalize an NDC to 11-digit form if possible, else None."""
    if not ndc:
        return None
    digits = re.sub(r"[^0-9]", "", str(ndc))
    if len(digits) == 11:
        return digits
    if len(digits) == 10:
        # 10-digit NDC: 4-4-2 -> 5-4-2
        return digits[:4] + "0" + digits[4:]
    if len(digits) == 9:
        return digits[:4] + "0" + digits[4:8] + "0" + digits[8:]
    return None

def empty_geo():
    return {"country": None, "admin1": None, "admin2": None, "postal_code": None,
            "name": None, "code": None, "level": "unknown"}

def empty_drug(original_name):
    return {
        "original_name": original_name,
        "canonical_name": None,
        "ingredient": None,
        "combination_ingredients": [],
        "brand_name": None,
        "rxnorm_rxcui": None,
        "atc_code": None,
        "ndc": None,
        "manufacturer": None,
        "strength": None,
        "dosage_form": None,
        "route": None,
        "identity_level": "unmapped",
        "mapping_status": "unmapped",
        "mapping_confidence": "none",
    }

def make_record(record_id, period_start, period_end, granularity, geo, drug,
                obs_type, metric, value, unit, semantics, is_quantity, is_monetary,
                source_id, source_name, source_url, source_record_id, native_frequency,
                published_at=None, event_date=None, quality_notes=None,
                is_suppressed=False):
    return {
        "record_id": record_id,
        "period_start": period_start,
        "period_end": period_end,
        "granularity": granularity,
        "geography": geo,
        "drug": drug,
        "observation": {
            "type": obs_type,
            "metric": metric,
            "value": value,
            "unit": unit,
            "semantics": semantics,
            "is_quantity": is_quantity,
            "is_monetary": is_monetary,
        },
        "source": {
            "source_id": source_id,
            "source_name": source_name,
            "source_url": source_url,
            "source_record_id": source_record_id,
            "native_frequency": native_frequency,
            "retrieved_at": today(),
            "published_at": published_at,
            "event_date": event_date,
        },
        "quality": {
            "is_imputed": False,
            "is_aggregated": False,
            "is_suppressed": is_suppressed,
            "notes": quality_notes,
        },
    }