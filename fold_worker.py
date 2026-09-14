"""Run in an ESMFold-specific environment, separate from TriGlue/DGL."""
from __future__ import annotations

import argparse
import json
import sys
import time
import resource
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--model-path")
    args = parser.parse_args()
    started = time.perf_counter()
    import torch

    sequence = json.load(sys.stdin)["sequence"]
    if args.chunk_size < 1:
        raise ValueError("chunk size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected folding environment")
    hf_directory = bool(args.model_path and Path(args.model_path).is_dir())
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    if hf_directory:
        from transformers import EsmForProteinFolding
        model = EsmForProteinFolding.from_pretrained(args.model_path, local_files_only=True,
                                                    low_cpu_mem_usage=True)
        # ESMFold's large language-model backbone uses FP16 on GPU; the folding
        # trunk remains FP32. This is explicitly recorded in the benchmark.
        if args.device == "cuda":
            model.esm = model.esm.half()
    elif args.model_path:
        from esm.esmfold.v1.pretrained import _load_model
        model = _load_model(args.model_path)
    else:
        import esm
        model = esm.pretrained.esmfold_v1()
    model = model.eval().to(args.device)
    if args.device == "cpu":
        model = model.float()
    if hf_directory:
        model.trunk.set_chunk_size(args.chunk_size)
    else:
        model.set_chunk_size(args.chunk_size)
    if args.device == "cuda":
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    fold_started = time.perf_counter()
    with torch.inference_mode():
        pdb = model.infer_pdb(sequence)
    Path(args.output).write_text(pdb, encoding="utf-8")
    if args.device == "cuda":
        torch.cuda.synchronize()
    metrics = {
        "backend": "transformers_esmfold_v1" if hf_directory else "fair_esm_esmfold_v1",
        "sequence_length": len(sequence), "chunk_size": args.chunk_size,
        "device": args.device, "torch": torch.__version__,
        "precision": "esm_fp16_trunk_fp32" if args.device == "cuda" else "fp32",
        "model_load_seconds": load_seconds, "fold_and_pdb_seconds": time.perf_counter() - fold_started,
        "worker_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2 if args.device == "cuda" else 0,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2 if args.device == "cuda" else 0,
    }
    Path(args.output + ".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
