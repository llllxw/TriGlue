#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference import predict_prepared
from data_process import add_structure_arguments, prepare_input, structure_options


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Predict from SMILES and two sequences, or from cached features.")
    result.add_argument("--input", required=True, help="triplet CSV")
    result.add_argument("--feature-root", help="existing prepared feature bundle; skips raw preprocessing")
    result.add_argument("--work-dir", default="results/preparation", help="new provenance-checked feature cache for raw input")
    result.add_argument("--prepare-device", choices=("cpu", "cuda"), default="cpu")
    result.add_argument("--checkpoint", action="append", required=True)
    result.add_argument("--run-config", action="append", required=True)
    result.add_argument("--temperature", action="append", type=float)
    result.add_argument("--output", required=True)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--symmetric", action="store_true")
    result.add_argument("--cache-size", type=int, default=64)
    result.add_argument(
        "--molformer-model",
        default="ibm-research/MoLFormer-XL-both-10pct",
        help="Hugging Face model ID or local snapshot directory; local is recommended for reproducibility",
    )
    add_structure_arguments(result)
    return result


def main() -> int:
    args = parser().parse_args()
    # Fail before expensive folding when release artifacts are missing.
    for filename in args.checkpoint + args.run_config:
        if not Path(filename).expanduser().is_file():
            raise FileNotFoundError(f"required checkpoint/config not found: {filename}")
    if not args.feature_root:
        prepared, _ = prepare_input(args.input, args.work_dir, device=args.prepare_device,
                                    **structure_options(args))
        args.input = str(prepared)
        args.feature_root = args.work_dir
    _, manifest = predict_prepared(
        args.input,
        args.feature_root,
        args.checkpoint,
        args.run_config,
        args.output,
        batch_size=args.batch_size,
        device=args.device,
        temperatures=args.temperature,
        symmetric=args.symmetric,
        cache_size=args.cache_size,
        molformer_model=args.molformer_model,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
