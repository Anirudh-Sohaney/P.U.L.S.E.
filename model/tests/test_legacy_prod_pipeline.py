"""The historical bundle must never masquerade as a live production run."""

import subprocess
import sys
from pathlib import Path

import pytest

from prod_pipeline import ProductionModel


def test_legacy_inference_fails_closed_without_current_verified_inputs(tmp_path):
    with pytest.raises(RuntimeError, match="Legacy production emission is disabled"):
        ProductionModel(tmp_path).predict()


def test_legacy_script_cannot_overwrite_checked_in_example(tmp_path):
    root = Path(__file__).resolve().parents[2]
    example = root / "model" / "final_predictions.json"
    original = example.read_bytes()
    result = subprocess.run([sys.executable, str(root / "model" / "prod_pipeline.py")],
                            cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "Legacy production emission is disabled" in result.stderr
    assert example.read_bytes() == original
