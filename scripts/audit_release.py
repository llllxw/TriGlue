#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from schema import sha256_file


REQUIRED_FILES = (
    "README.md",
    "QUICKSTART.zh-CN.md",
    "environment.yml",
    "requirements.txt",
    "pyproject.toml",
    "predict.py",
    "train.py",
    "explain.py",
    "calibration.py",
    "data_process.py",
    "model.py",
    "Dataset.py",
    "inference.py",
    "gragh_model.py",
    "sequence_model.py",
    "multimodal_fusion.py",
    "structure_resolver.py",
    "structure_prediction.py",
    "fold_worker.py",
    "feature_cache.py",
    "schema.py",
    "cli.py",
    "screen.py",
    "uncertainty.py",
    "examples/minimal_input.csv",
    "examples/train.csv",
    "examples/val.csv",
    "examples/triplets.csv",
    "scripts/smoke_test.py",
    "scripts/export_release.py",
    "tests/test_release_layout.py",
    "checkpoints/README.md",
    "data/README.md",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a TriGlue public release tree.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output")
    args = parser.parse_args()
    root = args.root.resolve()
    missing = [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]
    placeholder_terms = ("REPLACE_WITH", "TODO_PUBLIC_URL", "TODO_DOI")
    placeholders = []
    hashes = {}
    for relative in REQUIRED_FILES:
        path = root / relative
        if not path.is_file():
            continue
        hashes[relative] = sha256_file(path)
        if path.suffix.lower() in {".md", ".txt", ".yml", ".yaml", ".json", ".py"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            for term in placeholder_terms:
                if term in text:
                    placeholders.append({"file": relative, "term": term})
    code_status = "PASS" if not missing and not placeholders else "FAIL"
    payload = {
        "code_package_status": code_status,
        "scope": "required source files and placeholders only; runtime is tested separately",
        "root": str(root),
        "missing_required_files": missing,
        "unresolved_placeholders": placeholders,
        "required_file_sha256": hashes,
    }
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if code_status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
