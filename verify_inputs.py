"""Verify the published dataset and splits, and prepare inputs from release metadata."""
import argparse
from collections import Counter
import json
from pathlib import Path

import pandas as pd

from schema import _canonical_smiles, read_csv, sha256_file


def verify(dataset, splits, provenance):
    splits = Path(splits)
    provenance = Path(provenance)
    record = json.loads(provenance.read_text(encoding="utf-8"))
    if sha256_file(Path(dataset)) != record["dataset_sha256"]:
        raise ValueError("Dataset differs from the published Supplementary Table 1")
    frame = read_csv(dataset)
    if frame.sample_id.duplicated().any() or not frame.label.isin(["0", "1"]).all():
        raise ValueError("Require unique sample IDs and binary labels")
    for relative, digest in record["file_sha256"].items():
        path = Path(dataset) if relative == "data/dataset.csv" else splits / Path(relative).name
        if sha256_file(path) != digest:
            raise ValueError(f"File differs from the published version: {path}")

    ids = set(frame.sample_id)
    groups = None
    test_counts = Counter()
    for name in ["fixed"] + [f"fold{i}" for i in range(1, 6)]:
        table = read_csv(splits / f"{name}.csv")
        if table.sample_id.duplicated().any() or set(table.sample_id) != ids:
            raise ValueError(f"{name}: each dataset sample must occur exactly once")
        if set(table.partition) != {"train", "val", "test"} or table.parent_group_id.eq("").any():
            raise ValueError(f"{name}: expected three partitions and nonempty parent groups")
        if table.groupby("parent_group_id").partition.nunique().gt(1).any():
            raise ValueError(f"{name}: parent group crosses partitions")
        current = table.set_index("sample_id").parent_group_id.sort_index()
        if groups is None:
            groups = current
        elif not current.equals(groups):
            raise ValueError(f"{name}: inconsistent parent groups across splits")
        if name != "fixed":
            test_counts.update(table.loc[table.partition.eq("test"), "sample_id"])
    if test_counts != Counter(frame.sample_id):
        raise ValueError("Each sample must appear in exactly one outer test fold")
    return {"status": "PASS", "n_samples": len(frame), "n_folds": 5,
            "scope": "dataset and split checksums, parent groups and test-fold coverage"}


def prepare_triplets(dataset, splits, metadata):
    """Join final sample IDs to the published feature IDs in dataset order."""
    metadata = Path(metadata)
    if not metadata.is_file():
        raise FileNotFoundError(f"Release metadata missing: {metadata}. Extract TriGlueDesktop-data-v0.1.0.tar.gz or supply --metadata.")
    splits = Path(splits)
    manifest = json.loads((splits / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(metadata) != manifest["metadata_sha256"]:
        raise ValueError(f"Metadata differs from the published feature release: {metadata}")
    source = read_csv(dataset)
    meta = read_csv(metadata).set_index("sample_id", verify_integrity=True)
    if source.original_sample_id.duplicated().any() or set(source.original_sample_id) != set(meta.index):
        raise ValueError("The dataset and feature catalog must match one-to-one on original sample ID")
    meta = meta.loc[source.original_sample_id].copy()
    if meta.label.tolist() != source.label.tolist() or meta.SMILES.tolist() != source.SMILES.tolist():
        raise ValueError("Labels or compounds differ between the dataset and the feature catalog")

    fixed = read_csv(splits / "fixed.csv").set_index("sample_id")
    if meta.parent_group.tolist() != fixed.loc[source.sample_id, "parent_group_id"].tolist():
        raise ValueError("Published split and feature catalog have different parent groups")
    groups = meta.parent_group.to_dict()
    link_count = 0
    exact_groups = {}
    for row in meta.itertuples():
        parents = row.all_parent_sample_ids or row.parent_sample_id
        for parent in filter(None, parents.split(";")):
            if groups.get(parent) != row.parent_group:
                raise ValueError(f"Parent link crosses groups or cannot be resolved: {row.Index}, {parent}")
            link_count += 1
        smiles, error = _canonical_smiles(row.canonical_smiles)
        if error:
            raise ValueError(f"{row.Index}: {error}")
        key = (smiles, *sorted((row.protein1_seq, row.protein2_seq)))
        if exact_groups.setdefault(key, row.parent_group) != row.parent_group:
            raise ValueError("Identical sequence triplet occurs in different parent groups")
    if link_count != manifest["n_parent_links"]:
        raise ValueError("Parent link count differs from the split manifest")

    columns = {"ID": "compound_id", "canonical_smiles": "smiles", "protein1": "protein1_id",
               "protein2": "protein2_id", "protein1_seq": "protein1_sequence",
               "protein2_seq": "protein2_sequence", "label": "label", "parent_group": "parent_group_id"}
    prepared = meta[list(columns)].rename(columns=columns).reset_index(drop=True)
    prepared.insert(0, "triplet_id", source.sample_id.to_numpy())
    return prepared


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "data/dataset.csv")
    parser.add_argument("--splits", type=Path, default=root / "splits")
    parser.add_argument("--metadata", type=Path, help="also check the extracted release feature catalog and parent links")
    args = parser.parse_args()
    report = verify(args.dataset, args.splits, root / "provenance.json")
    if args.metadata is not None:
        report["verified_feature_rows"] = len(prepare_triplets(args.dataset, args.splits, args.metadata))
    print(json.dumps(report, indent=2))
