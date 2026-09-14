#!/usr/bin/env python
"""Check model reuse, reordered inputs and reloads against one declared protocol."""
from __future__ import annotations
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference import predict_prepared
from schema import read_csv, write_json


def main():
    parser = argparse.ArgumentParser()
    for name in ("input", "feature-root", "checkpoint", "run-config", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--molformer-model", default="ibm-research/MoLFormer-XL-both-10pct")
    args = parser.parse_args()
    frame = read_csv(args.input)
    if len(frame) == 1:
        import pandas as pd
        duplicate = frame.copy()
        duplicate["triplet_id"] = duplicate.triplet_id + "_consistency_duplicate"
        frame = pd.concat([frame, duplicate], ignore_index=True)
    shared = {}
    scores = []
    with tempfile.TemporaryDirectory(prefix="triglue-consistency-") as temp:
        for index, (table, batch_size, cache) in enumerate([
            (frame, 1, shared), (frame.iloc[::-1], 2, shared), (frame, 2, {}),
        ]):
            source = Path(temp) / f"input{index}.csv"
            table.to_csv(source, index=False)
            output, _ = predict_prepared(source, args.feature_root, [args.checkpoint], [args.run_config],
                                         Path(temp) / f"output{index}.csv", batch_size=batch_size,
                                         device=args.device, symmetric=True, runtime_cache=cache,
                                         molformer_model=args.molformer_model)
            scores.append(output.set_index("triplet_id").inducibility_score_mean.sort_index().to_numpy())
    differences = [float(np.max(np.abs(value - scores[0]))) for value in scores[1:]]
    report = {"status": "PASS" if max(differences) <= 1e-6 else "FAIL", "n_rows": len(frame),
              "maximum_absolute_score_differences": differences, "absolute_tolerance": 1e-6,
              "checks": ["reorder_and_reuse", "fresh_reload"],
              "protocol": "checkpoint_fixed_random_features_v1",
              "scope": "software consistency on supplied cases, not generalization or biological validation"}
    write_json(report, args.output)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
