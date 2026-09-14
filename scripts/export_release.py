#!/usr/bin/env python
"""Export source only, using an allowlist rather than packaging a working tree."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    ".gitignore", ".gitattributes", "README.md", "QUICKSTART.zh-CN.md",
    "requirements.txt", "requirements-dev.txt", "environment.yml", "pyproject.toml",
    "LICENSE", "LICENSE.md", "LICENSE.txt",
    "data_process.py", "Dataset.py", "model.py", "gragh_model.py",
    "sequence_model.py", "multimodal_fusion.py", "predict.py", "inference.py",
    "schema.py", "feature_cache.py", "structure_resolver.py", "structure_prediction.py",
    "fold_worker.py", "cli.py", "screen.py", "uncertainty.py", "train.py", "explain.py", "calibration.py",
}
SUBDIRECTORIES = {
    "docs": {".md"}, "examples": {".csv", ".md"},
    "scripts": {".py"}, "tests": {".py", ".csv", ".md"},
    "checkpoints": {".md"}, "data": {".md"},
}


def source_files(root=ROOT):
    root = Path(root).resolve()
    selected = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or path.is_symlink():
            continue
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts[:-1]):
            continue
        if len(relative.parts) == 1:
            include = path.name in ROOT_FILES
        else:
            include = path.suffix in SUBDIRECTORIES.get(relative.parts[0], set())
        if include:
            selected.append(path)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.suffix != ".zip":
        raise ValueError("output must end in .zip")
    if output == ROOT or ROOT in output.parents:
        raise ValueError("place the upload archive outside the source directory")
    if output.exists() or output.with_suffix(".sha256").exists():
        raise FileExistsError("choose a new output name; existing release artifacts are not overwritten")
    files = source_files()
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x", compression=ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, str(Path("TriGlue") / path.relative_to(ROOT)))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(json.dumps({"status": "SOURCE_EXPORTED", "output": str(output),
                      "n_files": len(files), "sha256": digest,
                      "scope": "source only; weights/data and public release completeness are audited separately"}, indent=2))


if __name__ == "__main__":
    main()
