from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_SCORE_COLUMNS = (
    "triplet_id",
    "compound_id",
    "smiles",
    "protein1_id",
    "protein2_id",
)
SEQUENCE_COLUMNS = ("protein1_sequence", "protein2_sequence")
STRUCTURE_COLUMNS = ("protein1_structure", "protein2_structure")
PROTEIN_ALPHABET = set("ACDEFGHIKLMNPQRSTVWYBXZOUJ")
SAFE_ID = re.compile(r"^[A-Za-z0-9_.:+-]+$")
RAW_COLUMNS = ("smiles", "protein1_sequence", "protein2_sequence")


def normalize_raw_triplets(frame: pd.DataFrame, *, base_dir: str | Path = ".") -> pd.DataFrame:
    """Normalize three-field input without changing existing explicit identifiers.

    Content IDs are stable across files; row IDs retain duplicate hypotheses.
    RDKit is mandatory here: validation must never silently accept bad chemistry.
    """
    from rdkit import Chem

    frame = frame.copy().reset_index(drop=True).fillna("")
    if "SMILES" in frame and "smiles" not in frame:
        frame = frame.rename(columns={"SMILES": "smiles"})
    missing = [name for name in RAW_COLUMNS if name not in frame]
    if missing:
        raise ValueError(f"raw input requires {RAW_COLUMNS}; missing {missing}")
    if frame.empty:
        raise ValueError("input table contains no rows")
    for column in ("triplet_id", "compound_id", "protein1_id", "protein2_id", *STRUCTURE_COLUMNS):
        if column not in frame:
            frame[column] = ""
    for index, row in frame.iterrows():
        smiles = str(row["smiles"]).strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"row {index + 2}: invalid or empty SMILES")
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        # Raw mode is normalized once. Prepared legacy inputs bypass this path.
        frame.at[index, "input_smiles"] = smiles
        frame.at[index, "smiles"] = canonical
        if not str(row["compound_id"]).strip():
            frame.at[index, "compound_id"] = "cmp_" + hashlib.sha256(canonical.encode()).hexdigest()[:24]
        if not str(row["triplet_id"]).strip():
            frame.at[index, "triplet_id"] = f"triplet_{index + 1:08d}"
        for side in ("protein1", "protein2"):
            sequence = re.sub(r"\s+", "", str(row[f"{side}_sequence"])).upper()
            if not sequence or len(sequence) > 1200 or set(sequence) - PROTEIN_ALPHABET:
                raise ValueError(f"row {index + 2}: {side} requires a valid sequence of 1–1200 residues; no truncation is performed")
            frame.at[index, f"{side}_sequence"] = sequence
            if not str(row[f"{side}_id"]).strip():
                frame.at[index, f"{side}_id"] = "prot_" + hashlib.sha256(sequence.encode()).hexdigest()[:24]
            value = str(row[f"{side}_structure"]).strip()
            if value:
                path = Path(value).expanduser()
                frame.at[index, f"{side}_structure"] = str((Path(base_dir) / path).resolve())
    report = audit_triplets(frame, require_sequences=True)
    if report["status"] != "PASS":
        raise ValueError(json.dumps(report["issues"][:20], ensure_ascii=False))
    return frame


@dataclass(frozen=True)
class AuditIssue:
    level: str
    row: int | None
    column: str | None
    message: str


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} must be UTF-8 or UTF-8 with BOM") from exc
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _canonical_smiles(value: str) -> tuple[str | None, str | None]:
    try:
        from rdkit import Chem
    except ImportError:
        if not value.strip():
            return None, "empty SMILES"
        return value.strip(), None
    mol = Chem.MolFromSmiles(value.strip())
    if mol is None:
        return None, "RDKit could not parse SMILES"
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), None


def _feature_paths(root: Path, compound_id: str, protein_id: str) -> dict[str, Path]:
    return {
        "compound_graph": root / "compound_graph" / f"{compound_id}.bin",
        "compound_embedding": root / "compound_3d_embedding" / f"{compound_id}.npy",
        "protein_one_hot": root / "protein_embedding" / "one_hot" / f"{protein_id}.npy",
        "protein_graph": root / "protein_graph" / f"{protein_id}.bin",
        "protein_esm2": root / "protein_3d_embedding" / f"{protein_id}.npy",
    }


def audit_triplets(
    frame: pd.DataFrame,
    *,
    require_sequences: bool = False,
    require_structures: bool = False,
    feature_root: str | Path | None = None,
) -> dict:
    issues: list[AuditIssue] = []
    if frame.empty:
        issues.append(AuditIssue("error", None, None, "input table contains no rows"))
    missing = [column for column in REQUIRED_SCORE_COLUMNS if column not in frame.columns]
    if missing:
        issues.append(AuditIssue("error", None, None, f"missing required columns: {missing}"))
        return _report(frame, issues, {})
    if require_sequences:
        missing_seq = [column for column in SEQUENCE_COLUMNS if column not in frame.columns]
        if missing_seq:
            issues.append(AuditIssue("error", None, None, f"missing sequence columns: {missing_seq}"))
    if require_structures:
        missing_struct = [column for column in STRUCTURE_COLUMNS if column not in frame.columns]
        if missing_struct:
            issues.append(AuditIssue("error", None, None, f"missing structure columns: {missing_struct}"))

    duplicate_ids = frame["triplet_id"].duplicated(keep=False)
    for idx in frame.index[duplicate_ids]:
        issues.append(AuditIssue("error", int(idx) + 2, "triplet_id", "duplicate triplet_id"))

    canonical: dict[str, str] = {}
    compound_sources = {}
    protein_sources = {}
    for idx, row in frame.iterrows():
        line = int(idx) + 2
        for column in REQUIRED_SCORE_COLUMNS:
            value = str(row[column]).strip()
            if not value:
                issues.append(AuditIssue("error", line, column, "value is empty"))
        for column in ("triplet_id", "compound_id", "protein1_id", "protein2_id"):
            value = str(row[column]).strip()
            if value and not SAFE_ID.fullmatch(value):
                issues.append(
                    AuditIssue(
                        "error",
                        line,
                        column,
                        "use only letters, numbers, dot, underscore, colon, plus, and hyphen",
                    )
                )
        normalized, error = _canonical_smiles(str(row["smiles"]))
        if error:
            issues.append(AuditIssue("error", line, "smiles", error))
        elif normalized is not None:
            canonical[str(row["triplet_id"])] = normalized
            cid = str(row["compound_id"])
            if cid in compound_sources and compound_sources[cid] != normalized:
                issues.append(AuditIssue("error", line, "compound_id", "same ID has conflicting molecular structures"))
            compound_sources[cid] = normalized

        for column in SEQUENCE_COLUMNS:
            if column not in frame.columns:
                continue
            sequence = re.sub(r"\s+", "", str(row[column]).upper())
            if require_sequences and not sequence:
                issues.append(AuditIssue("error", line, column, "sequence is empty"))
            invalid = sorted(set(sequence) - PROTEIN_ALPHABET)
            if invalid:
                issues.append(
                    AuditIssue("error", line, column, f"invalid amino-acid symbols: {invalid}")
                )
            if len(sequence) > 1200:
                issues.append(
                    AuditIssue(
                        "error",
                        line,
                        column,
                        f"length {len(sequence)} exceeds the supported 1,200-residue limit; automatic truncation is disabled",
                    )
                )
            pid = str(row[column.replace("_sequence", "_id")])
            if sequence and pid in protein_sources and protein_sources[pid] != sequence:
                issues.append(AuditIssue("error", line, column, "same protein ID has conflicting sequences"))
            if sequence:
                protein_sources[pid] = sequence

        for column in STRUCTURE_COLUMNS:
            if column not in frame.columns:
                continue
            value = str(row[column]).strip()
            if require_structures and not value:
                issues.append(AuditIssue("error", line, column, "structure path is empty"))
            elif value and not Path(value).expanduser().is_file():
                issues.append(AuditIssue("error", line, column, f"file not found: {value}"))

    feature_counts: dict[str, int] = {}
    if feature_root is not None and not missing:
        root = Path(feature_root).expanduser().resolve()
        checked: set[Path] = set()
        for idx, row in frame.iterrows():
            line = int(idx) + 2
            paths = {}
            paths.update(_feature_paths(root, str(row["compound_id"]), str(row["protein1_id"])))
            p2_paths = _feature_paths(
                root, str(row["compound_id"]), str(row["protein2_id"])
            )
            for key in ("protein_one_hot", "protein_graph", "protein_esm2"):
                paths[f"protein2_{key}"] = p2_paths[key]
            for key, path in paths.items():
                if path in checked:
                    continue
                checked.add(path)
                feature_counts[key] = feature_counts.get(key, 0) + 1
                if not path.is_file():
                    issues.append(AuditIssue("error", line, key, f"feature file not found: {path}"))
                    continue
                metadata_path = path.with_name(path.name + ".meta.json")
                if metadata_path.is_file():
                    try:
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                        if metadata.get("output_sha256") != sha256_file(path):
                            raise ValueError("feature checksum mismatch")
                        origin = metadata.get("source", {}).get("input")
                        if "compound" in key:
                            expected_input = str(row["smiles"])
                            actual_input = origin.get("smiles") if isinstance(origin, dict) else origin
                        else:
                            side = "protein2" if key.startswith("protein2_") else "protein1"
                            expected_input = str(row.get(f"{side}_sequence", ""))
                            actual_input = origin.get("sequence") if isinstance(origin, dict) else origin
                        if expected_input and actual_input != expected_input:
                            raise ValueError("cached feature input does not match this row")
                    except Exception as exc:
                        issues.append(AuditIssue("error", line, key, f"invalid provenance: {exc}"))
                elif (root / "feature_build_summary.json").is_file():
                    issues.append(AuditIssue("error", line, key, "new feature bundle is missing per-artifact provenance"))
                if path.suffix != ".npy":
                    continue
                try:
                    array = np.load(path, mmap_mode="r", allow_pickle=False)
                except Exception as exc:
                    issues.append(AuditIssue("error", line, key, f"cannot read {path}: {exc}"))
                    continue
                if not np.isfinite(array).all():
                    issues.append(AuditIssue("error", line, key, f"nonfinite values in {path}"))
                expected = {
                    "compound_embedding": 768,
                    "protein_one_hot": 25,
                    "protein2_protein_one_hot": 25,
                    "protein_esm2": 1280,
                    "protein2_protein_esm2": 1280,
                }.get(key)
                if expected is not None and (array.ndim < 2 or array.shape[-1] != expected):
                    issues.append(
                        AuditIssue(
                            "error",
                            line,
                            key,
                            f"expected final dimension {expected}, got shape {tuple(array.shape)} in {path}",
                        )
                    )
    return _report(frame, issues, canonical, feature_counts)


def _report(
    frame: pd.DataFrame,
    issues: Iterable[AuditIssue],
    canonical: dict[str, str],
    feature_counts: dict[str, int] | None = None,
) -> dict:
    issue_list = list(issues)
    return {
        "status": "PASS" if not any(issue.level == "error" for issue in issue_list) else "FAIL",
        "n_rows": int(len(frame)),
        "n_errors": sum(issue.level == "error" for issue in issue_list),
        "n_warnings": sum(issue.level == "warning" for issue in issue_list),
        "columns": list(frame.columns),
        "canonical_smiles": canonical,
        "feature_files_checked": feature_counts or {},
        "issues": [asdict(issue) for issue in issue_list],
    }


def write_json(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
