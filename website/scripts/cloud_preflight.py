"""Check cloud deployment inputs before starting Docker Compose.

This does not establish that DNS, TLS, or the backup mount is independent of
the application host; those require checks on the selected cloud server.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import stat
from pathlib import Path

WEBSITE = Path(__file__).resolve().parents[1]
ROOT = WEBSITE.parent
SOURCE = ROOT / "data/targeted_additions/cms_partd_geography_drug/data/arkansas_partd_geography_drug_by_year.csv.gz"
MANIFEST = SOURCE.parent.parent / "source_manifest.json"


def _env_values(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if separator and name.strip() in {"PULSE_DOMAIN", "PULSE_OFFSITE_BACKUP_DIR"}:
            values[name.strip()] = value.strip().strip('"\'')
    return values


def _check_key(path: Path, *, backup: bool, container_uid: int,
               container_gid: int) -> str:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ValueError(f"{path.name} must be a regular file, not a symlink")
    if info.st_mode & 0o027:
        raise ValueError(f"{path.name} must not be group-writable or accessible to others")
    if not ((info.st_uid == container_uid and info.st_mode & stat.S_IRUSR) or
            (info.st_gid == container_gid and info.st_mode & stat.S_IRGRP)):
        raise ValueError(f"{path.name} must be readable by container UID or GID 10001")
    value = path.read_text(encoding="utf-8").strip()
    if backup:
        if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
            raise ValueError("Backup key must be 32 random bytes encoded as 64 hex characters")
    elif len(value.encode("utf-8")) < 32 or len(set(value)) < 8:
        raise ValueError("Signing key must be at least 32 bytes and nontrivial")
    return value


def check(*, domain: str, offsite: Path, signing_key: Path, backup_key: Path,
          source: Path = SOURCE, manifest: Path = MANIFEST,
          container_uid: int = 10001, container_gid: int = 10001) -> dict:
    if (not re.fullmatch(r"[A-Za-z0-9.-]+", domain) or "." not in domain
            or domain.startswith(".") or domain.endswith(".")
            or domain == "pulse.example.com" or domain.lower() == "localhost"):
        raise ValueError("PULSE_DOMAIN must be a real DNS hostname without a scheme or path")
    if not offsite.is_absolute() or not offsite.is_dir():
        raise ValueError("PULSE_OFFSITE_BACKUP_DIR must be an existing absolute directory")
    data_dir = (WEBSITE / "data").resolve()
    target = offsite.resolve()
    if target == data_dir or data_dir in target.parents:
        raise ValueError("The offsite backup directory must be outside website/data")
    mode = target.stat()
    if not ((mode.st_uid == container_uid and mode.st_mode & stat.S_IWUSR and mode.st_mode & stat.S_IXUSR)
            or (mode.st_gid == container_gid and mode.st_mode & stat.S_IWGRP and mode.st_mode & stat.S_IXGRP)):
        raise ValueError("The offsite backup directory must be writable by UID or GID 10001")
    signing = _check_key(signing_key, backup=False, container_uid=container_uid,
                         container_gid=container_gid)
    backup = _check_key(backup_key, backup=True, container_uid=container_uid,
                        container_gid=container_gid)
    if signing == backup:
        raise ValueError("Signing and backup keys must differ")
    expected = json.loads(manifest.read_text(encoding="utf-8"))["normalized_sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("CMS manifest lacks a valid SHA-256 digest")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        header = stream.read(2)
        if header != b"\x1f\x8b":
            raise ValueError("CMS source is not gzip data; hydrate Git LFS before deployment")
        digest.update(header)
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError("CMS source checksum disagrees with its manifest")
    with gzip.open(source, "rb") as stream:
        if not stream.readline().startswith(b"year,"):
            raise ValueError("CMS source has an unexpected CSV header")
    return {"ready_for_compose": True, "domain": domain,
            "cms_source_sha256": expected, "offsite_directory": str(target),
            "note": "Verify DNS, TLS, and independent backup storage on the cloud host."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=WEBSITE / ".env")
    args = parser.parse_args()
    try:
        values = _env_values(args.env_file)
        result = check(
            domain=os.environ.get("PULSE_DOMAIN") or values.get("PULSE_DOMAIN") or "",
            offsite=Path(os.environ.get("PULSE_OFFSITE_BACKUP_DIR") or
                         values.get("PULSE_OFFSITE_BACKUP_DIR") or ""),
            signing_key=WEBSITE / "secrets/pulse_signing_key",
            backup_key=WEBSITE / "secrets/pulse_backup_key",
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"ready_for_compose": False, "error": str(exc)}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
