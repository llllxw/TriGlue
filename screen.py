from __future__ import annotations

from pathlib import Path
import tempfile
import os

import pandas as pd

from schema import read_csv


COMPOUND_COLUMNS = ("compound_id", "smiles")
PAIR_COLUMNS = (
    "pair_id",
    "protein1_id",
    "protein1_sequence",
    "protein1_structure",
    "protein2_id",
    "protein2_sequence",
    "protein2_structure",
)


def read_pairs(path):
    pairs = read_csv(path)
    base = Path(path).expanduser().resolve().parent
    for column in ("protein1_structure", "protein2_structure"):
        if column in pairs:
            pairs[column] = pairs[column].map(
                lambda value: str((base / Path(value).expanduser()).resolve()) if value else "")
    return pairs


def enumerate_screen(
    compounds_path: str | Path,
    protein_pairs_path: str | Path,
    output_path: str | Path,
    *,
    max_triplets: int = 100_000,
) -> pd.DataFrame:
    compounds = read_csv(compounds_path)
    pairs = read_pairs(protein_pairs_path)
    for column in ("protein1_structure", "protein2_structure"):
        if column not in pairs:
            pairs[column] = ""
    missing_compounds = [column for column in COMPOUND_COLUMNS if column not in compounds.columns]
    missing_pairs = [column for column in PAIR_COLUMNS if column not in pairs.columns]
    if missing_compounds:
        raise ValueError(f"compound table missing columns: {missing_compounds}")
    if missing_pairs:
        raise ValueError(f"protein-pair table missing columns: {missing_pairs}")
    if compounds.empty or pairs.empty:
        raise ValueError("compound and protein-pair tables must both contain at least one row")
    if compounds["compound_id"].duplicated().any():
        raise ValueError("compound_id must be unique in the compound table")
    if pairs["pair_id"].duplicated().any():
        raise ValueError("pair_id must be unique in the protein-pair table")
    n_triplets = len(compounds) * len(pairs)
    if n_triplets > max_triplets:
        raise ValueError(
            f"screen would create {n_triplets:,} rows, above --max-triplets={max_triplets:,}; "
            "increase the guard only after checking storage and feature-generation cost"
        )
    compounds = compounds.copy()
    pairs = pairs.copy()
    compounds["__join"] = 1
    pairs["__join"] = 1
    output = compounds.merge(pairs, on="__join", how="inner").drop(columns="__join")
    output.insert(
        0,
        "triplet_id",
        output["compound_id"].astype(str) + "__" + output["pair_id"].astype(str),
    )
    preferred = [
        "triplet_id",
        "compound_id",
        "smiles",
        "protein1_id",
        "protein1_sequence",
        "protein1_structure",
        "protein2_id",
        "protein2_sequence",
        "protein2_structure",
        "pair_id",
    ]
    output = output[preferred + [column for column in output.columns if column not in preferred]]
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False, encoding="utf-8")
    return output


def enumerate_screen_stream(compounds_path, protein_pairs_path, output_path, *,
                            max_triplets=100_000, chunk_size=1000):
    """Stream compound chunks crossed with individual pairs; output is atomic.

    The pair panel and compound ID set are retained in memory, not C × P rows.
    This is bounded enumeration, not a benchmark of proteome-scale screening.
    """
    if chunk_size < 1 or max_triplets < 1:
        raise ValueError("chunk_size and max_triplets must be positive")
    pairs = read_pairs(protein_pairs_path)
    for column in ("protein1_structure", "protein2_structure"):
        if column not in pairs:
            pairs[column] = ""
    missing = [c for c in PAIR_COLUMNS if c not in pairs]
    if missing or pairs.empty or pairs.pair_id.duplicated().any():
        raise ValueError(f"invalid protein-pair table; missing={missing}, require nonempty unique pair_id")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    seen = set()
    fd, temp_name = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    os.close(fd)
    try:
        for compounds in pd.read_csv(compounds_path, encoding="utf-8-sig", dtype=str,
                                      keep_default_na=False, chunksize=chunk_size):
            if any(c not in compounds for c in COMPOUND_COLUMNS):
                raise ValueError(f"compound table requires {COMPOUND_COLUMNS}")
            ids = compounds.compound_id.tolist()
            if len(set(ids)) != len(ids) or seen.intersection(ids):
                raise ValueError("compound_id must be unique across chunks")
            seen.update(ids)
            for _, pair in pairs.iterrows():
                batch = compounds.copy()
                for column, value in pair.items():
                    if column in batch:
                        raise ValueError(f"ambiguous shared column: {column}")
                    batch[column] = value
                batch.insert(0, "triplet_id", batch.compound_id + "__" + str(pair.pair_id))
                if count + len(batch) > max_triplets:
                    raise ValueError(f"screen exceeds --max-triplets={max_triplets}; output was not replaced")
                batch.to_csv(temp_name, mode="a", header=count == 0, index=False)
                count += len(batch)
        if count == 0:
            raise ValueError("empty compound table")
        os.replace(temp_name, destination)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return {"status": "PASS", "n_rows": count, "n_compounds": len(seen), "n_pairs": len(pairs),
            "output": str(destination), "chunk_size": chunk_size}
