from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from schema import read_csv


ID_COLUMNS = ("triplet_id", "compound_id", "protein1_id", "protein2_id")
SCORE_COLUMNS = ("inducibility_score", "inducibility_score_mean", "score", "probability", "y_prob")


def ensemble_summary(values):
    """Across-model dispersion; one model cannot estimate ensemble uncertainty."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError('expected finite scores shaped (triplets, models)')
    if ((values < 0) | (values > 1)).any():
        raise ValueError('scores must lie in [0, 1]')
    n = values.shape[1]
    empty = np.full(values.shape[0], np.nan)
    lo = np.quantile(values, 0.025, axis=1) if n > 1 else empty.copy()
    hi = np.quantile(values, 0.975, axis=1) if n > 1 else empty.copy()
    return dict(inducibility_score_mean=values.mean(axis=1),
                ensemble_sd=values.std(axis=1, ddof=1) if n > 1 else empty.copy(),
                ensemble_q025=lo, ensemble_q975=hi, ensemble_interval_width=hi-lo,
                n_models=n, uncertainty_status='ensemble_dispersion' if n>1 else 'single_model_not_available')


def _score_column(frame: pd.DataFrame) -> str:
    found = [column for column in SCORE_COLUMNS if column in frame.columns]
    if len(found) != 1:
        raise ValueError(f"each prediction file must contain exactly one of {SCORE_COLUMNS}; found {found}")
    return found[0]


def aggregate_predictions(paths: list[str | Path], output_path: str | Path) -> pd.DataFrame:
    if not paths:
        raise ValueError("at least one prediction file is required")
    if len({str(Path(p).resolve()) for p in paths}) != len(paths):
        raise ValueError('supply different per-checkpoint prediction files')
    frames = []
    reference_ids: list[str] | None = None
    base: pd.DataFrame | None = None
    for model_index, path in enumerate(paths, start=1):
        frame = read_csv(path)
        missing = [column for column in ID_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"{path} missing identifier columns: {missing}")
        if frame["triplet_id"].duplicated().any():
            raise ValueError(f"{path} contains duplicate triplet_id values")
        score_column = _score_column(frame)
        frame[score_column] = pd.to_numeric(frame[score_column], errors="raise")
        if not frame[score_column].between(0, 1).all():
            raise ValueError(f"{path}: scores must be in [0, 1]")
        frame = frame.sort_values("triplet_id").reset_index(drop=True)
        ids = frame["triplet_id"].tolist()
        if reference_ids is None:
            reference_ids = ids
            keep = list(ID_COLUMNS)
            if "smiles" in frame.columns:
                keep.append("smiles")
            base = frame[keep].copy()
        elif ids != reference_ids:
            raise ValueError(f"{path}: triplet_id set/order does not match the first prediction file")
        elif not frame[list(ID_COLUMNS)].equals(base[list(ID_COLUMNS)]):
            raise ValueError(f'{path}: compound/protein identities differ for aligned triplets')
        if 'n_models' in frame and not pd.to_numeric(frame.n_models).eq(1).all():
            raise ValueError('aggregate per-checkpoint predictions, not already-averaged ensembles')
        frames.append(frame[score_column].to_numpy(dtype=float))
    assert base is not None
    values = np.vstack(frames).T
    for name, value in ensemble_summary(values).items():
        base[name] = value
    base["rank_global"] = base["inducibility_score_mean"].rank(
        method="min", ascending=False
    ).astype(int)
    pair_key = base["protein1_id"].astype(str) + "__" + base["protein2_id"].astype(str)
    base["rank_within_protein_pair"] = base.groupby(pair_key)["inducibility_score_mean"].rank(
        method="min", ascending=False
    ).astype(int)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(destination, index=False, encoding="utf-8")
    return base
