"""Checked CSV/JSON I/O and metadata helpers.

Fail closed: missing inputs raise a descriptive error instead of producing
an empty panel. gzip support is inherited from pandas (compression="infer").
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def load_csv(
    path: Path | str,
    usecols: Optional[List[str]] = None,
    nrows: Optional[int] = None,
    low_memory: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load a (possibly gzipped) CSV, raising FileNotFoundError when absent."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Required data file missing: {p}. "
            "Run the data pipeline first or check --root."
        )
    return pd.read_csv(
        p,
        compression="infer",
        usecols=usecols,
        nrows=nrows,
        low_memory=low_memory,
        **kwargs,
    )


def write_csv(df: pd.DataFrame, path: Path | str) -> Path:
    """Write a DataFrame, creating its parent directory when necessary."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def write_json(obj: Any, path: Path | str) -> Path:
    """Write JSON with stable key ordering and project-aware type conversion."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=_json_default)
    return p


def read_json(path: Path | str) -> Any:
    """Read JSON and fail explicitly when the requested artifact is absent."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required JSON file missing: {p}")
    with p.open() as fh:
        return json.load(fh)


def write_metadata(path: Path | str, **fields: Any) -> Path:
    """Write provenance metadata with a created_at timestamp."""
    meta: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    return write_json(meta, path)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (os.PathLike, Path)):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} not JSON serializable")
