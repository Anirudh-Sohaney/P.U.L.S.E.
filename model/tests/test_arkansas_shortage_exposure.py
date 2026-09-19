"""Tests for the Arkansas-exposed FDA shortage target bridge."""

import pandas as pd

from pathlib import Path
import importlib.util


SCRIPT = Path(__file__).parents[2] / "data" / "targeted_additions" / "fda_shortage_archive" / "scripts" / "build_arkansas_medicaid_exposure.py"
SPEC = importlib.util.spec_from_file_location("arkansas_exposure", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_attach_preserves_unmatched_zero_semantics():
    medicaid = pd.DataFrame([
        {"ndc9": "000021433", "year": 2020, "quarter": 1,
         "medicaid_prescriptions": 10.0},
        {"ndc9": "000021434", "year": 2020, "quarter": 1,
         "medicaid_prescriptions": 20.0},
    ])
    archive = pd.DataFrame([
        {"ndc9": "000021433", "supplier": "Supplier", "month": "2020-02",
         "shortage_active": 1, "resolution_observed": 0},
    ])
    out = MODULE.attach_fda_shortage_state(medicaid, archive)
    assert out.loc[0, "fda_shortage_active"] == 1
    assert out.loc[0, "fda_archive_row_observed"] == 1
    assert out.loc[1, "fda_shortage_active"] == 0
    assert out.loc[1, "fda_archive_row_observed"] == 0
