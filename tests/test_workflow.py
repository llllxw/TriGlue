from pathlib import Path

import pandas as pd
import pytest

from schema import audit_triplets, read_csv
from screen import enumerate_screen
from uncertainty import aggregate_predictions


ROOT = Path(__file__).resolve().parents[1]


def test_example_input_passes() -> None:
    report = audit_triplets(read_csv(ROOT / "examples" / "triplets.csv"), require_sequences=True)
    assert report["status"] == "PASS"
    assert report["n_errors"] == 0


def test_duplicate_triplet_fails() -> None:
    frame = read_csv(ROOT / "examples" / "triplets.csv")
    frame.loc[1, "triplet_id"] = frame.loc[0, "triplet_id"]
    report = audit_triplets(frame)
    assert report["status"] == "FAIL"


def test_enumeration_guard(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="above --max-triplets"):
        enumerate_screen(
            ROOT / "examples" / "compounds.csv",
            ROOT / "examples" / "protein_pairs.csv",
            tmp_path / "screen.csv",
            max_triplets=3,
        )


def test_ensemble_alignment_and_summary(tmp_path: Path) -> None:
    output = aggregate_predictions(
        [
            ROOT / "tests" / "fixtures" / "mock_predictions_seed1.csv",
            ROOT / "tests" / "fixtures" / "mock_predictions_seed2.csv",
            ROOT / "tests" / "fixtures" / "mock_predictions_seed3.csv",
        ],
        tmp_path / "ensemble.csv",
    )
    assert output.loc[0, "inducibility_score_mean"] == pytest.approx((0.82 + 0.78 + 0.85) / 3)
    assert pd.api.types.is_integer_dtype(output["rank_global"])
