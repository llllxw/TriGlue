"""Full preprocessing implementation: raw input, molecular/protein features and cache management."""
from __future__ import annotations

import argparse
import re
from functools import lru_cache
import importlib.metadata
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from feature_cache import atomic_json, cache_hit, record_artifact
from schema import audit_triplets, normalize_raw_triplets, read_csv, sha256_file
from structure_resolver import resolve_structure


PROTEIN_ALPHABET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7, "I": 8, "H": 9,
    "K": 10, "M": 11, "L": 12, "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17,
    "R": 18, "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25,
}
RESIDUES = [
    "GLY", "ALA", "VAL", "LEU", "ILE", "PRO", "PHE", "TYR", "TRP", "SER", "THR",
    "CYS", "MET", "ASN", "GLN", "ASP", "GLU", "LYS", "ARG", "HIS", "M", "X",
]
METALS = {"FE", "SR", "GA", "IN", "ZN", "CU", "MN", "K", "NI", "NA", "CD", "MG", "CO", "HG", "CS", "CA"}
MAX_PROTEIN_LENGTH = 1200


def one_hot_protein(sequence: str) -> np.ndarray:
    sequence = re.sub(r"\s+", "", sequence.upper())
    if not sequence or len(sequence) > MAX_PROTEIN_LENGTH:
        raise ValueError("protein sequence must contain 1–1200 residues; truncation is disabled")
    output = np.zeros((MAX_PROTEIN_LENGTH, 25), dtype=np.float32)
    for index, residue in enumerate(sequence):
        output[index, PROTEIN_ALPHABET.get(residue, 24) - 1] = 1.0
    return output


def _one_hot_unknown(value, allowed):
    value = value if value in allowed else allowed[-1]
    return [int(value == item) for item in allowed]


def compound_graph(smiles: str):
    try:
        import dgl
        import torch
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("compound graphs require DGL, PyTorch, and RDKit") from exc
    symbols = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "B", "Si", "Fe", "Zn", "Cu", "Mn", "Mo", "other"]
    degrees = list(range(7))
    hybrids = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        "other",
    ]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")

    def atom_features(atom):
        values = (
            _one_hot_unknown(atom.GetSymbol(), symbols)
            + _one_hot_unknown(atom.GetDegree(), degrees)
            + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
            + _one_hot_unknown(atom.GetHybridization(), hybrids)
            + [int(atom.GetIsAromatic())]
            + _one_hot_unknown(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        )
        try:
            values += _one_hot_unknown(atom.GetProp("_CIPCode"), ["R", "S"]) + [
                int(atom.HasProp("_ChiralityPossible"))
            ]
        except Exception:
            values += [0, 0, int(atom.HasProp("_ChiralityPossible"))]
        return (values + [0] * 44)[:44]

    graph = dgl.graph(([], []), num_nodes=mol.GetNumAtoms())
    graph.ndata["feats"] = torch.tensor(
        [atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32
    )
    source, destination = [], []
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        source.extend([begin, end])
        destination.extend([end, begin])
    if source:
        graph.add_edges(source, destination)
    return graph


def protein_graph(structure_path: str, cutoff: float = 10.0):
    try:
        import dgl
        import MDAnalysis as mda
        import torch
        from MDAnalysis.analysis import distances
    except ImportError as exc:
        raise RuntimeError("protein graphs require DGL, PyTorch, and MDAnalysis") from exc
    universe = mda.Universe(structure_path)
    residues = list(universe.residues)
    if not residues:
        raise ValueError(f"structure contains no residues: {structure_path}")

    def normalized_name(name: str) -> str:
        name = name.strip().upper()
        return "M" if name in METALS else name if name in RESIDUES else "X"

    def node_features(residue):
        values = _one_hot_unknown(normalized_name(residue.resname), RESIDUES)
        try:
            coordinates = residue.atoms.positions
            pairwise = distances.self_distance_array(coordinates)
            maximum = float(pairwise.max() * 0.1) if pairwise.size else 0.0
            minimum = float(pairwise.min() * 0.1) if pairwise.size else 0.0
            selected = [residue.atoms.select_atoms(f"name {name}").positions for name in ("CA", "N", "C", "O")]
            ca, n, c, o = selected
            backbone = [
                float(np.linalg.norm(ca - o) * 0.1),
                float(np.linalg.norm(o - n) * 0.1),
                float(np.linalg.norm(n - c) * 0.1),
            ]
        except Exception:
            maximum = minimum = 0.0
            backbone = [0.0, 0.0, 0.0]
        angles = []
        for name in ("phi", "psi", "omega", "chi1"):
            try:
                selection = getattr(residue, f"{name}_selection")()
                value = float(selection.dihedral.value()) if selection else 0.0
            except Exception:
                value = 0.0
            angles.append(value * 0.01)
        values += [maximum, minimum] + backbone + angles
        return (values + [0.0] * 41)[:41]

    source, destination = [], []
    for first in range(len(residues)):
        for second in range(first + 1, len(residues)):
            matrix = distances.distance_array(
                residues[first].atoms.positions, residues[second].atoms.positions
            )
            if matrix.size and float(matrix.min()) <= cutoff:
                source.extend([first, second])
                destination.extend([second, first])
    graph = dgl.graph((source, destination), num_nodes=len(residues))
    graph.ndata["feats"] = torch.tensor(
        [node_features(residue) for residue in residues], dtype=torch.float32
    )
    return graph


@lru_cache(maxsize=1)
def _esm2_encoder(device):
    import esm
    import torch
    model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
    return model.eval().to(torch.device(device)), alphabet


def esm2_embeddings(proteins: list[tuple[str, str]], output_root: Path, device: str):
    import torch
    model, alphabet = _esm2_encoder(device)
    resolved = torch.device(device)
    converter = alphabet.get_batch_converter()
    for protein_id, sequence in proteins:
        destination = output_root / "protein_3d_embedding" / f"{protein_id}.npy"
        if destination.is_file() and np.load(destination, mmap_mode="r").shape[-1] == 1280:
            continue
        if len(sequence) > MAX_PROTEIN_LENGTH:
            raise ValueError("ESM-2 input exceeds supported length")
        _, _, tokens = converter([(protein_id, sequence)])
        with torch.inference_mode():
            result = model(tokens.to(resolved), repr_layers=[model.num_layers], return_contacts=False)
        array = result["representations"][model.num_layers][0, 1:-1].cpu().float().numpy()
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, array.astype(np.float32))


@lru_cache(maxsize=1)
def _unimol_encoder(use_cuda):
    from unimol_tools import UniMolRepr
    return UniMolRepr(data_type="molecule", batch_size=32, remove_hs=False,
                      model_name="unimolv2", model_size="84m", use_cuda=use_cuda)


def release_encoders():
    _esm2_encoder.cache_clear()
    _unimol_encoder.cache_clear()
    import gc
    gc.collect()
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def unimol2_embeddings(compounds: list[tuple[str, str]], output_root: Path, use_cuda: bool):
    from unimol_tools.data.conformer import inner_smi2coords

    # Use the same seed/mode as UniMolV2Feature. Its historical helper can
    # silently fall back to 2D: reject that case before producing a score.
    for compound_id, smiles in compounds:
        mol = inner_smi2coords(smiles, seed=42, mode="fast", remove_hs=False, return_mol=True)
        if mol.GetNumConformers() == 0 or not mol.GetConformer().Is3D():
            raise ValueError(f"3D conformer generation failed for {compound_id}; no 2D fallback is accepted")
        if mol.GetNumAtoms() > 256:
            raise ValueError(f"{compound_id} exceeds the 256-atom Uni-Mol feature limit (including H)")
        if not np.isfinite(mol.GetConformer().GetPositions()).all():
            raise ValueError(f"nonfinite conformer for {compound_id}")

    encoder = _unimol_encoder(use_cuda)
    pending = [
        (compound_id, smiles)
        for compound_id, smiles in compounds
        if not (output_root / "compound_3d_embedding" / f"{compound_id}.npy").is_file()
    ]
    if not pending:
        return
    result = encoder.get_repr([smiles for _, smiles in pending], return_atomic_reprs=False)
    arrays = result["cls_repr"]
    for (compound_id, _), array in zip(pending, arrays):
        array = np.asarray(array, dtype=np.float32).reshape(1, -1)
        if array.shape != (1, 768):
            raise ValueError(f"Uni-Mol2 output for {compound_id} has shape {array.shape}, expected (1, 768)")
        destination = output_root / "compound_3d_embedding" / f"{compound_id}.npy"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, array)


def add_structure_arguments(parser):
    parser.add_argument("--structure-backend", choices=("esmfold", "none"), default="esmfold")
    parser.add_argument("--structure-cache", help="shared content-addressed structure directory")
    parser.add_argument("--fold-python", default=sys.executable, help="Python in a working ESMFold environment")
    parser.add_argument("--fold-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--fold-timeout", type=int, default=1800)
    parser.add_argument("--fold-model-path", help="trusted local ESMFold .pt weights or Hugging Face model directory")
    parser.add_argument("--allow-structure-download", action="store_true",
                        help="allow AlphaFold DB lookup using optional protein*_uniprot; sequences are not uploaded")


def structure_options(args):
    return {name: getattr(args, name) for name in (
        "structure_backend", "structure_cache", "fold_python", "fold_device", "fold_timeout",
        "fold_model_path", "allow_structure_download")}


def prepare_input(input_csv, output_root, *, device="cpu", structure_backend="esmfold",
                  structure_cache=None, fold_python=sys.executable, fold_device="cuda",
                  fold_timeout=1800, fold_model_path=None, allow_structure_download=False,
                  skip_unimol2=False, skip_esm2=False, skip_protein_graphs=False):
    started = time.perf_counter()
    source_path = Path(input_csv).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        frame = normalize_raw_triplets(read_csv(source_path), base_dir=source_path.parent)
    except Exception as exc:
        atomic_json(output / "feature_build_summary.json", {
            "status": "FAIL", "failed_stage": "input_validation", "error": str(exc)})
        raise
    partial = skip_unimol2 or skip_esm2 or skip_protein_graphs
    summary = {"status": "RUNNING", "input_sha256": sha256_file(source_path),
               "n_triplets": len(frame), "partial_features": partial, "timings": {}}
    atomic_json(output / "feature_build_summary.json", summary)
    frame["preparation_status"] = "pending"
    frame["preparation_error"] = ""
    frame.to_csv(output / "normalized_input.csv", index=False)
    artifacts = []
    stage = "structures"
    try:
        # Resolve each distinct request once, even across the two protein roles.
        resolved = {}
        protein_requests = {}
        stage_start = time.perf_counter()
        for index, row in frame.iterrows():
            for side in ("protein1", "protein2"):
                pid = row[f"{side}_id"]
                sequence = row[f"{side}_sequence"]
                request = (sequence, row[f"{side}_structure"], str(row.get(f"{side}_chain", "")),
                           str(row.get(f"{side}_uniprot", "")))
                if pid in protein_requests and protein_requests[pid] != request:
                    raise ValueError(f"conflicting structure requests for protein ID {pid}")
                protein_requests[pid] = request
                if skip_protein_graphs:
                    continue
                if request not in resolved:
                    print(f"[structures] resolving {pid} ({len(sequence)} residues)", flush=True)
                    resolved[request] = resolve_structure(
                        sequence, Path(structure_cache) if structure_cache else output / "structures",
                        supplied=request[1], chain_id=request[2], accession=request[3],
                        allow_download=allow_structure_download, backend=structure_backend,
                        fold_python=fold_python, fold_device=fold_device, timeout=fold_timeout,
                        model_path=fold_model_path)
                structure, details = resolved[request]
                frame.at[index, f"{side}_structure"] = str(structure)
                for key in ("source", "quality_status", "mean_plddt", "sequence_match", "selected_chain"):
                    frame.at[index, f"{side}_{key}"] = details[key]
        summary["timings"]["structures_seconds"] = time.perf_counter() - stage_start
        frame.to_csv(output / "resolved_input.csv", index=False)

        versions = {}
        for package in ("numpy", "rdkit", "dgl", "torch", "MDAnalysis", "fair-esm", "unimol-tools"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "not_installed"
        generator_hash = sha256_file(Path(__file__))
        summary["versions"] = versions

        def ensure(relative, value, builder, *, shape=None, graph_dim=None):
            path = output / relative
            source = {"input": value, "feature": relative.split("/")[0], "schema": "public-v2",
                      "generator_sha256": generator_hash, "versions": versions}
            hit = cache_hit(path, source)
            if not hit:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix="feature-", dir=path.parent) as temporary:
                    temporary_path = Path(temporary) / path.name
                    builder(temporary_path)
                    validate_artifact(temporary_path, shape=shape, graph_dim=graph_dim)
                    temporary_path.replace(path)
                record_artifact(path, source, shape=shape, graph_node_dimension=graph_dim)
            else:
                validate_artifact(path, shape=shape, graph_dim=graph_dim)
            artifacts.append({"relative_path": relative, "sha256": sha256_file(path), "cache_hit": hit})

        def save_graph(path, graph):
            import dgl
            dgl.save_graphs(str(path), [graph])

        # Heavy encoders run only for misses. Temporary roots prevent them from
        # reusing pre-existing untracked artifacts or leaving partial outputs.
        def embed(path, kind, identifier, value):
            with tempfile.TemporaryDirectory(prefix="encoder-", dir=output) as temporary:
                root = Path(temporary)
                if kind == "compound_3d_embedding":
                    unimol2_embeddings([(identifier, value)], root, device == "cuda")
                else:
                    esm2_embeddings([(identifier, value)], root, device)
                (root / kind / f"{identifier}.npy").replace(path)

        stage = "compound_features"
        stage_start = time.perf_counter()
        for cid, smiles in frame[["compound_id", "smiles"]].drop_duplicates().itertuples(index=False, name=None):
            print(f"[features] compound {cid}", flush=True)
            ensure(f"compound_graph/{cid}.bin", smiles,
                   lambda p: save_graph(p, compound_graph(smiles)), graph_dim=44)
            if not skip_unimol2:
                ensure(f"compound_3d_embedding/{cid}.npy", {"smiles": smiles, "model": "unimolv2-84m", "seed": 42},
                       lambda p: embed(p, "compound_3d_embedding", cid, smiles), shape=(1, 768))
        summary["timings"]["compound_features_seconds"] = time.perf_counter() - stage_start
        release_encoders()
        stage = "protein_features"
        stage_start = time.perf_counter()
        proteins = {}
        for side in ("protein1", "protein2"):
            for pid, seq, structure in frame[[f"{side}_id", f"{side}_sequence", f"{side}_structure"]].itertuples(index=False, name=None):
                proteins[pid] = (seq, structure)
        for pid, (seq, structure) in proteins.items():
            print(f"[features] protein {pid}", flush=True)
            ensure(f"protein_embedding/one_hot/{pid}.npy", seq,
                   lambda p: np.save(p, one_hot_protein(seq)), shape=(1200, 25))
            if not skip_protein_graphs:
                ensure(f"protein_graph/{pid}.bin", {"sequence": seq, "pdb_sha256": sha256_file(Path(structure))},
                       lambda p: save_graph(p, protein_graph(structure)), graph_dim=41)
            if not skip_esm2:
                ensure(f"protein_3d_embedding/{pid}.npy", {"sequence": seq, "model": "esm2_t33_650M_UR50D", "layer": 33},
                       lambda p: embed(p, "protein_3d_embedding", pid, seq), shape=(len(seq), 1280))
        summary["timings"]["protein_features_seconds"] = time.perf_counter() - stage_start
        if not partial:
            audit = audit_triplets(frame, feature_root=output)
            if audit["status"] != "PASS":
                raise ValueError(json.dumps(audit["issues"]))
        frame["preparation_status"] = "partial" if partial else "ready"
        frame.to_csv(output / "prepared_triplets.csv", index=False)
        summary.update(status="PARTIAL" if partial else "PASS", n_compounds=frame.compound_id.nunique(),
                       n_proteins=len(proteins))
    except Exception as exc:
        frame["preparation_status"] = "failed"
        frame["preparation_error"] = f"{stage}: {exc}"
        summary.update(status="FAIL", failed_stage=stage, error=str(exc))
        raise
    finally:
        release_encoders()
        frame.to_csv(output / "preparation_status.csv", index=False)
        pd.DataFrame(artifacts, columns=["relative_path", "sha256", "cache_hit"]).to_csv(output / "feature_manifest.csv", index=False)
        summary["elapsed_seconds"] = time.perf_counter() - started
        atomic_json(output / "feature_build_summary.json", summary)
    return output / "prepared_triplets.csv", summary


def validate_artifact(path, *, shape=None, graph_dim=None):
    if shape is not None:
        array = np.load(path, allow_pickle=False)
        if tuple(array.shape) != tuple(shape) or not np.isfinite(array).all():
            raise ValueError(f"invalid array {path}: expected finite {shape}, got {array.shape}")
    if graph_dim is not None:
        import dgl
        import torch
        graphs = dgl.load_graphs(str(path))[0]
        if len(graphs) != 1 or graphs[0].num_nodes() == 0:
            raise ValueError(f"expected one nonempty graph: {path}")
        x = graphs[0].ndata.get("feats")
        if x is None or x.ndim != 2 or x.shape[1] != graph_dim or not torch.isfinite(x).all():
            raise ValueError(f"invalid graph node features: {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare TriGlue features from SMILES and two sequences.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--validate-only", action="store_true", help="normalize input; do not fold or load encoders")
    for name in ("unimol2", "esm2", "protein-graphs"):
        parser.add_argument("--skip-" + name, action="store_true")
    add_structure_arguments(parser)
    args = parser.parse_args(argv)
    if args.validate_only:
        frame = normalize_raw_triplets(read_csv(args.input), base_dir=Path(args.input).resolve().parent)
        root = Path(args.output_root)
        root.mkdir(parents=True, exist_ok=True)
        frame.to_csv(root / "normalized_input.csv", index=False)
        print(json.dumps({"status": "INPUT_VALIDATED_ONLY", "n_rows": len(frame)}, indent=2))
        return 0
    _, summary = prepare_input(args.input, args.output_root, device=args.device,
                               skip_unimol2=args.skip_unimol2, skip_esm2=args.skip_esm2,
                               skip_protein_graphs=args.skip_protein_graphs, **structure_options(args))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
