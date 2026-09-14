#!/usr/bin/env python
"""Measured raw-vs-prepared cost, including local folding and process-tree RAM.

Each cold raw run gets its own empty feature/structure directory. Model assets
are preinstalled; neither downloads nor fresh OS page-cache eviction are claimed.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def cpu_model():
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def worker(args):
    started = time.perf_counter()
    from data_process import prepare_input
    from inference import predict_prepared
    import torch
    import resource

    run = Path(args.output_root)
    run.mkdir(parents=True, exist_ok=True)
    runtime = {}
    preparation = None

    def call(index):
        nonlocal preparation
        begin = time.perf_counter()
        source = args.input
        features = args.feature_root
        if args.mode == "raw":
            source, preparation = prepare_input(
                source, run / "features", device="cuda", fold_python=sys.executable,
                fold_device="cuda", fold_model_path=args.fold_model,
                structure_backend="esmfold")
            features = run / "features"
        _, manifest = predict_prepared(source, features, [args.checkpoint], [args.run_config],
                                       run / f"prediction_{index}.csv", device="cpu", batch_size=1,
                                       symmetric=False, runtime_cache=runtime,
                                       molformer_model=args.molformer_model)
        return time.perf_counter() - begin, manifest

    _, first_manifest = call("cold")
    cold_seconds = time.perf_counter() - started
    cold_preparation = preparation
    # A separate warm-up follows the cold call; timed repeats are cache hits.
    call("warmup")
    warm_seconds, score_seconds = [], []
    for index in range(args.warm_repeats):
        elapsed, manifest = call(index)
        warm_seconds.append(elapsed)
        score_seconds.append(sum(m["model_seconds"] for m in manifest["models"]))
    structural = []
    if args.mode == "raw":
        for path in sorted((run / "features/structures").glob("*.meta.json")):
            structural.append(json.loads(path.read_text())["details"])
    payload = {
        "status": "PASS", "mode": args.mode, "cold_first_call_seconds": cold_seconds,
        "cold_preparation": cold_preparation, "warm_complete_call_seconds": warm_seconds,
        "warm_scoring_stage_seconds": score_seconds, "folding_records": structural,
        "parent_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "parent_peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "parent_peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "torch_num_threads": torch.get_num_threads(), "first_prediction_manifest": first_manifest,
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "protocol": "single_checkpoint_fixed_features_eval_dropout_disabled_no_swap",
    }
    (run / "worker_metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"mode": args.mode, "cold_seconds": cold_seconds, "status": "PASS"}), flush=True)


def supervise(command, folder):
    import psutil
    folder.mkdir(parents=True, exist_ok=False)
    peak_mib, samples = 0.0, 0
    started = time.perf_counter()
    with (folder / "execution.log").open("w") as logfile:
        child = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT)
        while child.poll() is None:
            try:
                root = psutil.Process(child.pid)
                processes = [root] + root.children(recursive=True)
                total = 0
                for process in processes:
                    try:
                        total += process.memory_info().rss
                    except psutil.Error:
                        pass
                peak_mib = max(peak_mib, total / 1024**2)
                samples += 1
            except psutil.Error:
                pass
            time.sleep(0.1)
    report = {"exit_code": child.returncode, "wall_seconds": time.perf_counter() - started,
              "sampled_peak_process_tree_rss_mib": peak_mib, "sampling_interval_seconds": 0.1,
              "n_samples": samples, "command": command}
    (folder / "process_monitor.json").write_text(json.dumps(report, indent=2))
    if child.returncode:
        raise RuntimeError(f"benchmark job failed; inspect {folder / 'execution.log'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=("raw", "prepared"))
    parser.add_argument("--input")
    parser.add_argument("--feature-root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--fold-model", required=True)
    parser.add_argument("--molformer-model", default="ibm-research/MoLFormer-XL-both-10pct")
    parser.add_argument("--cold-repeats", type=int, default=3)
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--protein1-fasta")
    parser.add_argument("--protein2-fasta")
    parser.add_argument("--smiles")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.cold_repeats < 1 or args.warm_repeats < 1:
        raise ValueError("repeat counts must be positive")
    import pandas as pd
    from Bio import SeqIO

    folder = Path(args.output_root).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    first = str(SeqIO.read(args.protein1_fasta, "fasta").seq)
    second = str(SeqIO.read(args.protein2_fasta, "fasta").seq)
    raw_input = folder / "raw_input.csv"
    if raw_input.exists():
        raise ValueError("benchmark folder already contains input; use a fresh folder")
    pd.DataFrame([{"smiles": args.smiles, "protein1_sequence": first,
                   "protein2_sequence": second}]).to_csv(raw_input, index=False)
    config = {**vars(args), "sequence_lengths": [len(first), len(second)],
              "started_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
              "platform": platform.platform(), "cpu_model": cpu_model(), "model_downloads_included": False,
              "raw_input": "exactly three columns; no structures or feature IDs", "gpu":
              subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()}
    (folder / "benchmark_config.json").write_text(json.dumps(config, indent=2))
    base = [sys.executable, str(Path(__file__).resolve()), "--worker", "--checkpoint", args.checkpoint,
            "--run-config", args.run_config, "--fold-model", args.fold_model,
            "--molformer-model", args.molformer_model, "--warm-repeats", str(args.warm_repeats)]
    for index in range(args.cold_repeats):
        run = folder / f"raw_{index + 1}"
        print(f"Starting cold raw run {index + 1}/{args.cold_repeats}", flush=True)
        supervise(base + ["--mode", "raw", "--input", str(raw_input), "--output-root", str(run)], run)
        print(json.loads((run / "worker_metrics.json").read_text())["cold_first_call_seconds"], flush=True)
    # Use exactly the same newly generated feature bundle as raw_1: this avoids
    # confusing old cropped graphs with full-sequence ESMFold graph costs.
    prepared_root = folder / "raw_1/features"
    for index in range(args.cold_repeats):
        run = folder / f"prepared_{index + 1}"
        print(f"Starting cold prepared run {index + 1}/{args.cold_repeats}", flush=True)
        supervise(base + ["--mode", "prepared", "--input", str(prepared_root / "prepared_triplets.csv"),
                           "--feature-root", str(prepared_root), "--output-root", str(run)], run)
    print("Both input modes completed", flush=True)


if __name__ == "__main__":
    main()
