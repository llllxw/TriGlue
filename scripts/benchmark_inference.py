#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import tempfile
import time
import resource
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import predict_prepared


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TriGlue inference on named hardware.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--run-config", action="append", required=True)
    parser.add_argument("--temperature", action="append", type=float)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument(
        "--molformer-model", default="ibm-research/MoLFormer-XL-both-10pct"
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if args.warmup < 1 or args.repeats < 1:
        raise ValueError("--warmup and --repeats must be >=1; cold loading is excluded from warm measurements")
    import numpy as np
    import pandas as pd
    import torch

    device_resolved = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_resolved == "auto":
        device_resolved = "cpu"
    if device_resolved == "cuda":
        torch.cuda.reset_peak_memory_stats()
    end_to_end = []
    forward_only = []
    n_triplets = len(pd.read_csv(args.input))
    first_total = None
    runtime_cache = {}
    with tempfile.TemporaryDirectory(prefix="triglue-benchmark-") as temp:
        for run_index in range(args.warmup + args.repeats):
            output = Path(temp) / f"prediction_{run_index}.csv"
            started = time.perf_counter()
            _, manifest = predict_prepared(
                args.input,
                args.feature_root,
                args.checkpoint,
                args.run_config,
                output,
                batch_size=args.batch_size,
                device=args.device,
                temperatures=args.temperature,
                symmetric=args.symmetric,
                molformer_model=args.molformer_model,
                runtime_cache=runtime_cache,
            )
            total = time.perf_counter() - started
            model_time = sum(item["model_seconds"] for item in manifest["models"])
            if args.symmetric:
                model_time += sum(
                    item.get("reverse_order_model_seconds", 0.0) for item in manifest["models"]
                )
            if run_index == 0:
                first_total = total
            if run_index >= args.warmup:
                end_to_end.append(total)
                forward_only.append(model_time)
    payload = {
        "status": "PASS",
        "timestamp_note": "record the release timestamp externally",
        "platform": platform.platform(),
        "cpu": next((line.split(":", 1)[1].strip() for line in
                 Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unknown")
                 if Path("/proc/cpuinfo").exists() else platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "device_requested": args.device,
        "device_resolved": device_resolved,
        "gpu_name": torch.cuda.get_device_name(0) if device_resolved == "cuda" else None,
        "n_triplets": n_triplets,
        "batch_size": args.batch_size,
        "n_models": len(args.checkpoint),
        "model_manifests": manifest["models"],
        "symmetric": args.symmetric,
        "warmup_runs": args.warmup,
        "timed_repeats": args.repeats,
        "cold_start_end_to_end_seconds": first_total,
        "end_to_end_seconds": end_to_end,
        "end_to_end_mean_seconds": statistics.mean(end_to_end),
        "end_to_end_sd_seconds": statistics.stdev(end_to_end) if len(end_to_end) > 1 else 0.0,
        "forward_seconds": forward_only,
        "forward_mean_seconds": statistics.mean(forward_only),
        "forward_median_seconds": statistics.median(forward_only),
        "forward_p95_seconds": float(np.quantile(forward_only, 0.95)),
        "latency_scope": "each latency is for the complete input table; use one input row for single-triplet latency",
        "runtime_mode": "all requested checkpoints remain resident across warm repetitions",
        "peak_process_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024),
        "forward_triplets_per_second": n_triplets / statistics.mean(forward_only),
        "peak_cuda_memory_mb": (
            torch.cuda.max_memory_allocated() / 1024**2 if device_resolved == "cuda" else None
        ),
        "timing_scope": (
            "cold_start includes model construction/checkpoint loading; repeated end_to_end uses resident models "
            "and includes audits, hashes, feature reads and CSV writes. forward includes data reads/batching and "
            "device transfer, not just neural kernels. No raw structure or feature generation is included."
        ),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
