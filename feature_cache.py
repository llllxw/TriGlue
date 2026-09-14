"""Fail-closed, content/config-addressed cache provenance."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from schema import sha256_file


def digest_json(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def cache_hit(path: Path, source: dict) -> bool:
    """Existing artifacts without matching provenance are never silently reused."""
    meta = path.with_name(path.name + ".meta.json")
    if not path.exists() and not meta.exists():
        return False
    if not path.is_file() or not meta.is_file():
        raise ValueError(f"incomplete/untracked cache artifact: {path}; use a new feature root")
    record = json.loads(meta.read_text(encoding="utf-8"))
    if record.get("source_digest") != digest_json(source):
        raise ValueError(f"cache source/config conflict: {path}; use a new feature root")
    if record.get("output_sha256") != sha256_file(path):
        raise ValueError(f"cache checksum mismatch: {path}")
    return True


def record_artifact(path: Path, source: dict, **details) -> None:
    atomic_json(path.with_name(path.name + ".meta.json"), {
        "source": source, "source_digest": digest_json(source),
        "output_sha256": sha256_file(path), **details,
    })
