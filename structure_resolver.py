"""Resolve exact-sequence monomer structures, never substitute homologs."""
from __future__ import annotations

import json
import re
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

from feature_cache import cache_hit, digest_json, record_artifact
from schema import sha256_file
from structure_prediction import predict_structure

AA3 = dict(zip(
    "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(),
    "ACDEFGHIKLMNPQRSTVWY"))


def select_exact_chain(source: Path, sequence: str, destination: Path, *, chain_id: str = "",
                       predicted: bool = False, confidence_scale: float = 1.0) -> dict:
    from Bio.PDB import PDBIO, PDBParser, Select

    structure = PDBParser(QUIET=True).get_structure("protein", str(source))
    models = list(structure)
    if not models:
        raise ValueError(f"no model in structure: {source}")
    candidates = []
    for chain in models[0]:
        if chain_id and chain.id.strip() != chain_id.strip():
            continue
        residues = [r for r in chain if r.id[0] == " " and r.resname in AA3]
        observed = "".join(AA3[r.resname] for r in residues)
        if observed == sequence:
            candidates.append((chain, residues))
    if not candidates:
        raise ValueError("no exact full-length coordinate sequence match; check chain, mutations and missing residues. "
                         "Use a complete predicted monomer instead of silently cropping the input.")
    if len(candidates) > 1:
        raise ValueError("multiple exact matching chains; specify protein1_chain/protein2_chain")
    chain, residues = candidates[0]
    for residue in residues:
        if any(atom not in residue for atom in ("N", "CA", "C", "O")):
            raise ValueError(f"missing backbone atoms in residue {residue.id}")
        if not all(np.isfinite(atom.coord).all() for atom in residue):
            raise ValueError("nonfinite structure coordinates")
        if any(atom.is_disordered() for atom in residue):
            raise ValueError("alternate atom locations require explicit preprocessing")
    accepted = {id(r) for r in residues}

    class ChainSelect(Select):
        def accept_model(self, model):
            return model is models[0]

        def accept_chain(self, item):
            return item is chain

        def accept_residue(self, residue):
            return id(residue) in accepted

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(destination), ChainSelect())
    confidence = float(np.mean([r["CA"].bfactor for r in residues])) * confidence_scale if predicted else None
    return {"selected_chain": chain.id, "sequence_match": "exact", "n_residues": len(residues),
            "mean_plddt": confidence,
            "quality_status": ("low_confidence" if confidence < 50 else "predicted_not_experimentally_verified")
            if confidence is not None else "user_structure_confidence_not_assessed"}


def _fetch_alphafold(accession: str, destination: Path) -> str:
    if not re.fullmatch(r"[A-Za-z0-9]+(?:-[0-9]+)?", accession):
        raise ValueError("invalid UniProt accession")
    endpoint = "https://alphafold.ebi.ac.uk/api/prediction/" + accession
    request = urllib.request.Request(endpoint, headers={"User-Agent": "TriGlue/0.2"})
    with urllib.request.urlopen(request, timeout=30) as response:
        entries = json.load(response)
    if len(entries) != 1 or not entries[0].get("pdbUrl"):
        raise ValueError("AlphaFold lookup did not return one unambiguous PDB model")
    url = entries[0]["pdbUrl"]
    if not url.startswith("https://alphafold.ebi.ac.uk/files/"):
        raise ValueError("unexpected AlphaFold download host")
    with urllib.request.urlopen(url, timeout=60) as response:
        destination.write_bytes(response.read(30_000_000))
    return url


def resolve_structure(sequence: str, cache_dir: Path, *, supplied: str = "", chain_id: str = "",
                      accession: str = "", allow_download: bool = False, backend: str = "esmfold",
                      fold_python: str = sys.executable, fold_device: str = "cuda",
                      timeout: int = 1800, model_path: str | None = None) -> tuple[Path, dict]:
    if backend not in ("esmfold", "none"):
        raise ValueError(f"unsupported folding backend: {backend}")
    supplied_path = Path(supplied).expanduser().resolve() if supplied else None
    model_asset = Path(model_path) if model_path else None
    if model_asset and model_asset.is_dir():
        model_asset = model_asset / "pytorch_model.bin"
    source = {"sequence": sequence, "chain": chain_id, "schema": "exact-monomer-v2-confidence-scale",
              "supplied_sha256": sha256_file(supplied_path) if supplied_path else None,
              "accession": accession if allow_download and not supplied else "",
              "backend": backend if not supplied else "supplied",
              "fold_weights_sha256": sha256_file(model_asset) if model_asset and not supplied else None,
              "fold_adapter_sha256": sha256_file(Path(__file__).with_name("fold_worker.py")) if not supplied else None}
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / (digest_json(source) + ".pdb")
    if cache_hit(output, source):
        metadata = json.loads(output.with_name(output.name + ".meta.json").read_text())
        return output, {**metadata["details"], "cache_hit": True}
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="resolve-", dir=cache_dir) as temporary:
        raw = Path(temporary) / "raw.pdb"
        selected = Path(temporary) / "selected.pdb"
        download_error = None
        details = None
        if supplied_path:
            details = select_exact_chain(supplied_path, sequence, selected, chain_id=chain_id)
            details.update(source="user_structure", source_reference=str(supplied_path))
        elif accession and allow_download:
            try:
                url = _fetch_alphafold(accession, raw)
                details = select_exact_chain(raw, sequence, selected, chain_id=chain_id, predicted=True)
                details.update(source="alphafold_database", source_reference=url)
            except Exception as exc:
                download_error = str(exc)
        if details is None:
            if backend == "none":
                raise ValueError("no usable structure; provide protein*_structure or enable --structure-backend esmfold. "
                                 + (download_error or ""))
            fold_metrics = predict_structure(sequence, raw, python=fold_python, device=fold_device,
                              timeout=timeout, model_path=model_path)
            # Transformers 4.46.3 writes categorical pLDDT in [0, 1] into
            # PDB B-factor fields; fair-esm/AlphaFold use the [0, 100] scale.
            # Normalize only the reported confidence, never atom coordinates.
            confidence_scale = 100.0 if (fold_metrics or {}).get("backend") == "transformers_esmfold_v1" else 1.0
            details = select_exact_chain(raw, sequence, selected, predicted=True,
                                         confidence_scale=confidence_scale)
            details.update(source="esmfold_v1", source_reference="local_prediction")
            details["fold_metrics"] = fold_metrics
        details.update(elapsed_seconds=time.perf_counter() - started, cache_hit=False,
                       download_error=download_error)
        selected.replace(output)
        record_artifact(output, source, details=details)
    return output, details
