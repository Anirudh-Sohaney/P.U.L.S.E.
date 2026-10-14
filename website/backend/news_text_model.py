"""Safe runtime inference for the locally trained 20-signal text model.

The production backend loads a parameter-only JSON artifact, never a pickle.
This model was trained on ten matched monthly examples and remains a shadow
estimate; its records are deliberately excluded from usable latest values.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "catalog" / "news_text_20_model.json.gz"
MODEL_ARTIFACT_SHA256 = "f2b975b770180485f5af23ba0fde4e68deb3094e1b8fd4ef44f9e9daf467257c"
MODEL_SCHEMA = "pulse_news_text_ridge_v1"
TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


@lru_cache(maxsize=1)
def load_model(path: str = str(MODEL_PATH)) -> dict:
    """Load and validate the compact, non-executable model artifact."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != MODEL_ARTIFACT_SHA256:
        raise ValueError("20-signal model artifact checksum mismatch")
    with gzip.open(Path(path), "rt", encoding="utf-8") as source:
        artifact = json.load(source)
    if artifact.get("schema") != MODEL_SCHEMA:
        raise ValueError("Unsupported 20-signal text model artifact")
    signal_ids = artifact.get("signal_ids")
    if (not isinstance(signal_ids, list) or len(signal_ids) != 20
            or len(set(signal_ids)) != 20 or not all(isinstance(item, str) for item in signal_ids)):
        raise ValueError("20-signal model artifact must contain 20 unique string IDs")
    vocabulary = artifact.get("vocabulary")
    idf = np.asarray(artifact.get("idf"), dtype=np.float64)
    coefficients = np.asarray(artifact.get("coefficients"), dtype=np.float64)
    intercept = np.asarray(artifact.get("intercept"), dtype=np.float64)
    size = len(vocabulary) if isinstance(vocabulary, dict) else -1
    if (size <= 0 or len(idf) != size or coefficients.shape != (20, size)
            or intercept.shape != (20,) or not np.isfinite(idf).all()
            or not np.isfinite(coefficients).all() or not np.isfinite(intercept).all()
            or sorted(vocabulary.values()) != list(range(size))):
        raise ValueError("20-signal model parameter shape or values are invalid")
    if artifact.get("vectorizer") != {
            "token_pattern": r"(?u)\b\w\w+\b", "ngram_range": [1, 2],
            "sublinear_tf": True, "norm": "l2"}:
        raise ValueError("20-signal model vectorizer configuration is unsupported")
    return {**artifact, "idf_array": idf, "coefficient_array": coefficients,
            "intercept_array": intercept,
            "stopword_set": set(artifact.get("stop_words", []))}


def _features(text: str, artifact: dict) -> np.ndarray:
    tokens = [token for token in TOKEN_PATTERN.findall(text.lower())
              if token not in artifact["stopword_set"]]
    terms = tokens + [f"{left} {right}" for left, right in zip(tokens, tokens[1:])]
    counts: dict[int, int] = {}
    vocabulary = artifact["vocabulary"]
    for term in terms:
        index = vocabulary.get(term)
        if index is not None:
            counts[index] = counts.get(index, 0) + 1
    vector = np.zeros(len(artifact["idf_array"]), dtype=np.float64)
    for index, count in counts.items():
        vector[index] = (1.0 + math.log(count)) * artifact["idf_array"][index]
    norm = float(np.linalg.norm(vector))
    if norm:
        vector /= norm
    return vector


def predict(text: str, artifact: dict | None = None) -> dict[str, float]:
    """Estimate all archived outputs from one closed-month article aggregate."""
    if not str(text or "").strip():
        raise ValueError("20-signal text inference requires non-empty source text")
    model = artifact or load_model()
    values = model["coefficient_array"] @ _features(str(text), model) + model["intercept_array"]
    if not np.isfinite(values).all():
        raise ValueError("20-signal model emitted a non-finite output")
    return {name: float(max(0.0, value))
            for name, value in zip(model["signal_ids"], values, strict=True)}
