"""Encrypt and verify a committed PULSE snapshot for separate storage.

The AES-256-GCM key belongs outside the snapshot and signing-key file. Archive
members are plain filenames only; decrypting never extracts arbitrary paths.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from scripts.backup_databases import (
    CMS_SOURCE_NAME, DATABASES, _sync_directory, verify_backup,
)


MAGIC = b"PULSE-BACKUP-V1\n"
NONCE_BYTES = 12
TAG_BYTES = 16
CHUNK_BYTES = 1024 * 1024
ALLOWED_FILES = {*DATABASES, "manifest.json", "cms_partd_source_manifest.json"}


def load_key(path: Path) -> bytes:
    try:
        key = bytes.fromhex(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError) as exc:
        raise ValueError("Backup key must be a readable 64-character hex file") from exc
    if len(key) != 32:
        raise ValueError("Backup key must contain exactly 32 bytes")
    return key


class _EncryptingWriter(io.RawIOBase):
    def __init__(self, target, encryptor):
        self.target = target
        self.encryptor = encryptor

    def writable(self) -> bool:
        return True

    def write(self, data) -> int:
        self.target.write(self.encryptor.update(data))
        return len(data)


def _allowed_name(name: str) -> bool:
    return name in ALLOWED_FILES or bool(CMS_SOURCE_NAME.fullmatch(name))


def _unpack(bundle: Path, destination: Path, key: bytes) -> dict:
    if len(key) != 32:
        raise ValueError("Backup key must contain exactly 32 bytes")
    total = bundle.stat().st_size
    header_size = len(MAGIC) + NONCE_BYTES
    if total <= header_size + TAG_BYTES:
        raise ValueError("Encrypted backup is incomplete")
    descriptor, archive_name = tempfile.mkstemp(
        prefix=f".{destination.name}.archive.", dir=destination.parent)
    archive_path = Path(archive_name)
    try:
        with bundle.open("rb") as source, os.fdopen(descriptor, "wb") as archive:
            header = source.read(header_size)
            if not header.startswith(MAGIC):
                raise ValueError("Encrypted backup format is unknown")
            nonce = header[len(MAGIC):]
            source.seek(total - TAG_BYTES)
            tag = source.read(TAG_BYTES)
            source.seek(header_size)
            decryptor = Cipher(algorithms.AES(key), modes.GCM(nonce, tag)).decryptor()
            decryptor.authenticate_additional_data(header)
            remaining = total - header_size - TAG_BYTES
            while remaining:
                block = source.read(min(CHUNK_BYTES, remaining))
                if not block:
                    raise ValueError("Encrypted backup ended early")
                archive.write(decryptor.update(block))
                remaining -= len(block)
            try:
                archive.write(decryptor.finalize())
            except InvalidTag as exc:
                raise ValueError("Encrypted backup authentication failed") from exc
        with tarfile.open(archive_path, mode="r:gz") as contents:
            members = contents.getmembers()
            names = [member.name for member in members]
            if (len(names) != len(set(names)) or not all(
                    member.isfile() and _allowed_name(member.name) for member in members)):
                raise ValueError("Encrypted backup contains an unsafe member")
            for member in members:
                output = destination / member.name
                descriptor = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                with contents.extractfile(member) as original, os.fdopen(descriptor, "wb") as copy:
                    shutil.copyfileobj(original, copy, CHUNK_BYTES)
                    copy.flush()
                    os.fsync(copy.fileno())
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if set(names) != {*manifest.get("files", {}), "manifest.json"}:
            raise ValueError("Encrypted backup members do not match manifest")
        return verify_backup(destination)
    finally:
        archive_path.unlink(missing_ok=True)


def export_encrypted(snapshot: Path, output: Path, key: bytes) -> Path:
    if len(key) != 32:
        raise ValueError("Backup key must contain exactly 32 bytes")
    manifest = verify_backup(snapshot)
    if "cms_partd_source_manifest.json" not in manifest["files"]:
        raise ValueError("Offsite backup requires the committed CMS source pair")
    if not all(_allowed_name(name) for name in manifest["files"]):
        raise ValueError("Backup manifest contains an unsafe filename")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if output.exists():
        raise FileExistsError(output)
    descriptor, staged_name = tempfile.mkstemp(prefix=f".{output.name}.",
                                               suffix=".tmp", dir=output.parent)
    staged = Path(staged_name)
    nonce = os.urandom(NONCE_BYTES)
    header = MAGIC + nonce
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(header)
            encryptor = Cipher(algorithms.AES(key), modes.GCM(nonce)).encryptor()
            encryptor.authenticate_additional_data(header)
            writer = _EncryptingWriter(destination, encryptor)
            with tarfile.open(fileobj=writer, mode="w|gz") as archive:
                for name in sorted([*manifest["files"], "manifest.json"]):
                    archive.add(snapshot / name, arcname=name, recursive=False)
            destination.write(encryptor.finalize())
            destination.write(encryptor.tag)
            destination.flush()
            os.fsync(destination.fileno())
        with tempfile.TemporaryDirectory(prefix="pulse-encrypted-verify-") as temporary:
            _unpack(staged, Path(temporary), key)
        if output.exists():
            raise FileExistsError(output)
        os.link(staged, output)
        _sync_directory(output.parent)
        return output
    finally:
        staged.unlink(missing_ok=True)


def verify_encrypted(bundle: Path, key: bytes) -> dict:
    """Authenticate and validate an existing export before treating it as done."""
    with tempfile.TemporaryDirectory(prefix="pulse-encrypted-verify-") as temporary:
        return _unpack(bundle, Path(temporary), key)


def restore_encrypted(bundle: Path, destination: Path, key: bytes) -> dict:
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staged = Path(tempfile.mkdtemp(prefix="pulse-decrypted-", dir=destination.parent))
    staged.chmod(0o700)
    try:
        manifest = _unpack(bundle, staged, key)
        if destination.exists():
            raise FileExistsError(destination)
        _sync_directory(staged)
        os.replace(staged, destination)
        _sync_directory(destination.parent)
        return manifest
    finally:
        if staged.exists() and staged.resolve().parent == destination.parent.resolve():
            shutil.rmtree(staged)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("export", "restore"))
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--key-file", required=True, type=Path)
    args = parser.parse_args()
    key = load_key(args.key_file)
    if args.operation == "export":
        print(export_encrypted(args.source, args.destination, key))
    else:
        manifest = restore_encrypted(args.source, args.destination, key)
        print(json.dumps({"destination": str(args.destination),
                          "files": sorted(manifest["files"])}))


if __name__ == "__main__":
    main()
