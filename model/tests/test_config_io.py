"""Config, I/O, and normalization unit tests (plain asserts, pytest-safe)."""

from pathlib import Path
import json
import tempfile

from arkansas_pharma_signal import io
from arkansas_pharma_signal.config import Config, resolve_root
from arkansas_pharma_signal.entities import normalize_name


def test_resolve_root():
    root = resolve_root(".")
    assert (root / "data").is_dir()
    assert (root / "model").is_dir()


def test_config_paths():
    cfg = Config(root=".")
    assert cfg.data_path("x.csv").name == "x.csv"
    assert cfg.artifacts_dir.name == "artifacts"
    for d in (cfg.panel_dir, cfg.trained_dir, cfg.forecasts_dir,
              cfg.evaluation_dir, cfg.metadata_dir):
        assert "artifacts" in str(d)


def test_load_csv_missing_raises():
    try:
        io.load_csv("/no/such/file.csv.gz")
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass


def test_json_roundtrip(tmp_path: Path):
    path = io.write_json({"a": 1, "b": [1, 2]}, tmp_path / "m.json")
    assert json.loads(path.read_text()) == {"a": 1, "b": [1, 2]}


def test_normalize_name():
    assert normalize_name("Metformin HCl") == "metformin"
    assert normalize_name("ATIVAN") == "ativan"
    assert normalize_name("  Aripiprazole TABLET ") == "aripiprazole"
    assert normalize_name(None) == ""
    assert normalize_name(float("nan")) == ""


def test_write_metadata_timestamp(tmp_path: Path):
    path = io.write_metadata(tmp_path / "meta.json", rows=10)
    meta = json.loads(path.read_text())
    assert meta["rows"] == 10
    assert meta["created_at"]


if __name__ == "__main__":
    for fn in (test_resolve_root, test_config_paths, test_load_csv_missing_raises,
               test_normalize_name):
        fn()
    with tempfile.TemporaryDirectory() as td:
        test_json_roundtrip(Path(td))
        test_write_metadata_timestamp(Path(td))
    print("ok")