#!/usr/bin/env python
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from schema import audit_triplets, normalize_raw_triplets, read_csv
from screen import enumerate_screen
from uncertainty import aggregate_predictions


def main() -> None:
    triplets = read_csv(ROOT / "examples" / "triplets.csv")
    audit = audit_triplets(triplets, require_sequences=True)
    assert audit["status"] == "PASS", audit
    minimal = normalize_raw_triplets(read_csv(ROOT / "examples/minimal_input.csv"))
    assert "label" not in minimal
    assert audit_triplets(minimal, require_sequences=True)["status"] == "PASS"
    with tempfile.TemporaryDirectory(prefix="triglue-smoke-") as temp:
        temp = Path(temp)
        enumerated = enumerate_screen(
            ROOT / "examples" / "compounds.csv",
            ROOT / "examples" / "protein_pairs.csv",
            temp / "enumerated.csv",
            max_triplets=10,
        )
        assert len(enumerated) == 4
        aggregated = aggregate_predictions(
            [
                ROOT / "tests" / "fixtures" / "mock_predictions_seed1.csv",
                ROOT / "tests" / "fixtures" / "mock_predictions_seed2.csv",
                ROOT / "tests" / "fixtures" / "mock_predictions_seed3.csv",
            ],
            temp / "ensemble.csv",
        )
        assert len(aggregated) == 2
        assert aggregated["n_models"].eq(3).all()
        assert pd.api.types.is_integer_dtype(aggregated["rank_global"])
    print(
        json.dumps(
            {
                "status": "PASS",
                "validated_triplets": len(triplets),
                "validated_three_field_inputs": len(minimal),
                "enumerated_triplets": len(enumerated),
                "aggregated_models": 3,
                "heavy_models_downloaded": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
