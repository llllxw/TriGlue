"""Isolated optional folding backend; never sends user sequences to a service."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def predict_structure(sequence: str, destination: Path, *, python: str = sys.executable,
                      device: str = "cuda", timeout: int = 1800, chunk_size: int = 64,
                      model_path: str | None = None) -> dict:
    invalid = sorted(set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"))
    if invalid:
        raise ValueError(f"ESMFold requires standard residues; unsupported symbols: {invalid}")
    worker = Path(__file__).with_name("fold_worker.py")
    command = [python, str(worker), "--output", str(destination), "--device", device,
               "--chunk-size", str(chunk_size)]
    if model_path:
        command += ["--model-path", str(Path(model_path).expanduser().resolve())]
    try:
        result = subprocess.run(command, input=json.dumps({"sequence": sequence}), text=True,
                                capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ESMFold exceeded {timeout} s; provide a verified structure or adjust the timeout") from exc
    if result.returncode or not destination.is_file():
        raise RuntimeError("ESMFold failed. Use --fold-python from a working ESMFold environment, "
                           "or provide protein*_structure. Diagnostic: " + result.stderr[-3000:])
    metadata = Path(str(destination) + ".metrics.json")
    return json.loads(metadata.read_text()) if metadata.is_file() else {
        "backend": "esmfold_v1", "device": device, "chunk_size": chunk_size}
