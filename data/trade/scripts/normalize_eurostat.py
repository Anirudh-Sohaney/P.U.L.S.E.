#!/usr/bin/env python3
"""Flatten Eurostat Comext JSON (cache/) to long-form CSV in data/.

Datasets are SITC Rev.4 aggregates (not CN8 — CN8 granularity requires the
bulk Comext download, see sources.md). Columns: freq, indic_et, sitc06,
partner, geo, time, value.
"""
import csv
import json
import sys
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / "cache"
DATA = Path(__file__).resolve().parent.parent / "data"

WANT = {
    "comext_intra_eu.json": "comext_intra_eu.csv",
    "comext_extra_eu.json": "comext_extra_eu.csv",
}


def flatten(src: Path, dst: Path) -> int:
    d = json.loads(src.read_text())
    dims = d["dimension"]
    # index map: dimension name -> {code: position}
    idx = {}
    for name, dim in dims.items():
        idx[name] = {code: i for i, code in enumerate(dim["category"]["index"])}
    codes = {name: list(dim["category"]["index"]) for name, dim in dims.items()}
    order = d["id"]
    size = d["size"]

    def unroll(pos: int, flat: int, prefix: dict):
        if pos == len(order):
            val = d["value"].get(str(flat), None)
            if val is not None:
                yield [prefix[n] for n in order] + [val]
            return
        name = order[pos]
        for code in codes[name]:
            prefix[name] = code
            yield from unroll(pos + 1, flat * size[pos] + idx[name][code], prefix)

    with open(dst, "w", newline="") as out:
        wr = csv.writer(out)
        wr.writerow(order + ["value"])
        n = 0
        for row in unroll(0, 0, {}):
            wr.writerow(row)
            n += 1
    return n


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in WANT.items():
        src = CACHE / src_name
        dst = DATA / dst_name
        if dst.exists():
            print(f"skip  {dst_name} (exists)")
            continue
        n = flatten(src, dst)
        print(f"got   {dst_name} ({dst.stat().st_size} bytes, {n} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())