from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from schema import normalize_raw_triplets, audit_triplets
from feature_cache import cache_hit, record_artifact
from structure_resolver import resolve_structure, select_exact_chain
from data_process import prepare_input

ROOT = Path(__file__).resolve().parents[1]


def raw():
    return pd.DataFrame([{"smiles": "CCO", "protein1_sequence": "AC", "protein2_sequence": "AC"}])


def write_pdb(path, chains=("A",), residues=("ALA", "CYS"), confidence=80):
    lines = []
    serial = 0
    for chain in chains:
        for resid, residue in enumerate(residues, 1):
            for atom in ("N", "CA", "C", "O"):
                serial += 1
                lines.append(f"ATOM  {serial:5d} {atom:^4s} {residue:3s} {chain}{resid:4d}    "
                             f"{float(serial):8.3f}{(serial % 3) * 0.7:8.3f}{(serial % 5) * 0.2:8.3f}{1.:6.2f}{confidence:6.2f}          {atom[0]:>2s}  \n")
        lines.append("TER\n")
    path.write_text("".join(lines) + "END\n")
    return path


def test_three_fields_generate_stable_ids():
    first = normalize_raw_triplets(raw())
    second = normalize_raw_triplets(raw())
    assert first.equals(second)
    assert first.protein1_id[0] == first.protein2_id[0]
    assert audit_triplets(first, require_sequences=True)["status"] == "PASS"
    assert "label" not in first


@pytest.mark.parametrize("column,value", [("smiles", "not-smiles"), ("smiles", ""),
                                          ("protein1_sequence", "A" * 1201),
                                          ("protein2_sequence", "AC*"), ("protein1_sequence", "")])
def test_invalid_raw_rejected(column, value):
    frame = raw()
    frame[column] = value
    with pytest.raises(ValueError):
        normalize_raw_triplets(frame)


def test_same_id_conflicting_sequence_rejected():
    frame = raw()
    frame["protein1_id"] = frame["protein2_id"] = "p"
    frame["protein2_sequence"] = "AD"
    with pytest.raises(ValueError, match="conflicting"):
        normalize_raw_triplets(frame)


def test_equivalent_smiles_are_normalized_once():
    frame = pd.concat([raw(), raw()], ignore_index=True)
    frame.loc[1, "smiles"] = "OCC"
    output = normalize_raw_triplets(frame)
    assert output.compound_id.nunique() == 1
    assert output.smiles.nunique() == 1
    assert output.input_smiles.nunique() == 2


def test_cache_rejects_conflict_and_tampering(tmp_path):
    path = tmp_path / "feature.npy"
    assert not cache_hit(path, {"sequence": "AC"})
    np.save(path, np.ones((2, 3)))
    with pytest.raises(ValueError, match="untracked"):
        cache_hit(path, {"sequence": "AC"})
    record_artifact(path, {"sequence": "AC"})
    assert cache_hit(path, {"sequence": "AC"})
    with pytest.raises(ValueError, match="conflict"):
        cache_hit(path, {"sequence": "AD"})
    np.save(path, np.zeros((2, 3)))
    with pytest.raises(ValueError, match="checksum"):
        cache_hit(path, {"sequence": "AC"})


def test_structure_exact_match_and_ambiguity(tmp_path):
    pdb = write_pdb(tmp_path / "input.pdb", chains=("A", "B"))
    with pytest.raises(ValueError, match="multiple"):
        select_exact_chain(pdb, "AC", tmp_path / "out.pdb")
    detail = select_exact_chain(pdb, "AC", tmp_path / "out.pdb", chain_id="B", predicted=True)
    assert detail["selected_chain"] == "B"
    assert detail["mean_plddt"] == 80
    with pytest.raises(ValueError, match="exact"):
        select_exact_chain(pdb, "AD", tmp_path / "bad.pdb")


def test_supplied_structure_cached_without_network(tmp_path, monkeypatch):
    pdb = write_pdb(tmp_path / "input.pdb")
    import structure_resolver as module
    monkeypatch.setattr(module, "_fetch_alphafold", lambda *a: pytest.fail("network not allowed"))
    path, detail = resolve_structure("AC", tmp_path / "cache", supplied=str(pdb), backend="none")
    assert detail["source"] == "user_structure"
    assert detail["mean_plddt"] is None  # experimental B factor is not pLDDT
    again, detail = resolve_structure("AC", tmp_path / "cache", supplied=str(pdb), backend="none")
    assert again == path and detail["cache_hit"]


def test_fold_backend_orchestration_is_mocked(tmp_path, monkeypatch):
    import structure_resolver as module
    calls = []
    def fake_fold(sequence, destination, **kwargs):
        calls.append(sequence)
        write_pdb(destination, confidence=40)
    monkeypatch.setattr(module, "predict_structure", fake_fold)
    path, detail = resolve_structure("AC", tmp_path / "cache")
    assert path.is_file() and detail["quality_status"] == "low_confidence"
    resolve_structure("AC", tmp_path / "cache")
    assert calls == ["AC"]


def test_no_structure_failure_is_explicit(tmp_path):
    input_csv = tmp_path / "input.csv"
    raw().to_csv(input_csv, index=False)
    with pytest.raises(ValueError, match="no usable structure"):
        prepare_input(input_csv, tmp_path / "features", structure_backend="none")
    report = json.loads((tmp_path / "features/feature_build_summary.json").read_text())
    assert report["status"] == "FAIL" and report["failed_stage"] == "structures"
    assert not (tmp_path / "features/prepared_triplets.csv").exists()


def test_transformers_fold_confidence_scale(tmp_path, monkeypatch):
    import structure_resolver as module
    def fake_fold(sequence, destination, **kwargs):
        write_pdb(destination, confidence=0.8)
        return {"backend": "transformers_esmfold_v1"}
    monkeypatch.setattr(module, "predict_structure", fake_fold)
    _, detail = resolve_structure("AC", tmp_path / "cache")
    assert detail["mean_plddt"] == pytest.approx(80)
    assert detail["quality_status"] == "predicted_not_experimentally_verified"
    _, again = resolve_structure("AC", tmp_path / "cache")
    assert again["cache_hit"] and again["mean_plddt"] == pytest.approx(80)


def test_validate_only_root_entry(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "data_process.py"), "--input",
                             str(ROOT / "examples/minimal_input.csv"), "--output-root", str(tmp_path),
                             "--validate-only"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "INPUT_VALIDATED_ONLY" in result.stdout
    assert not (tmp_path / "structures").exists()


def test_partial_lightweight_features(tmp_path):
    pytest.importorskip("dgl")
    input_csv = tmp_path / "input.csv"
    raw().to_csv(input_csv, index=False)
    prepared, summary = prepare_input(input_csv, tmp_path / "features", skip_unimol2=True,
                                      skip_esm2=True, skip_protein_graphs=True)
    assert summary["status"] == "PARTIAL"
    frame = pd.read_csv(prepared)
    assert frame.preparation_status.eq("partial").all()
    _, again = prepare_input(input_csv, tmp_path / "features", skip_unimol2=True,
                             skip_esm2=True, skip_protein_graphs=True)
    assert again["status"] == "PARTIAL"


def test_streamed_enumeration_matches_original(tmp_path):
    from screen import enumerate_screen, enumerate_screen_stream
    a = enumerate_screen(ROOT / "examples/compounds.csv", ROOT / "examples/protein_pairs.csv", tmp_path / "a.csv")
    report = enumerate_screen_stream(ROOT / "examples/compounds.csv", ROOT / "examples/protein_pairs.csv",
                                      tmp_path / "b.csv", chunk_size=1)
    b = pd.read_csv(tmp_path / "b.csv", keep_default_na=False)
    assert report["n_rows"] == len(a)
    pd.testing.assert_frame_equal(a.sort_values("triplet_id").reset_index(drop=True),
                                  b[a.columns].sort_values("triplet_id").reset_index(drop=True))
    before = (tmp_path / "b.csv").read_bytes()
    with pytest.raises(ValueError, match="exceeds"):
        enumerate_screen_stream(ROOT / "examples/compounds.csv", ROOT / "examples/protein_pairs.csv",
                                tmp_path / "b.csv", chunk_size=1, max_triplets=1)
    assert (tmp_path / "b.csv").read_bytes() == before


def test_full_features_with_supplied_synthetic_structure(tmp_path):
    """Opt-in real encoders; synthetic coordinates test plumbing, not biology."""
    import os
    if os.environ.get("TRIGLUE_HEAVY_TESTS") != "1":
        pytest.skip("set TRIGLUE_HEAVY_TESTS=1 to exercise locally cached heavy encoders")
    pdb = write_pdb(tmp_path / "input.pdb")
    frame = raw()
    frame["protein1_structure"] = frame["protein2_structure"] = str(pdb)
    source = tmp_path / "raw.csv"
    frame.to_csv(source, index=False)
    prepared, summary = prepare_input(source, tmp_path / "features", structure_backend="none")
    assert summary["status"] == "PASS"
    assert audit_triplets(pd.read_csv(prepared), feature_root=tmp_path / "features")["status"] == "PASS"
    checkpoint = os.environ.get("TRIGLUE_TEST_CHECKPOINT")
    config = os.environ.get("TRIGLUE_TEST_RUN_CONFIG")
    if checkpoint and config:
        prediction_path = tmp_path / "prediction.csv"
        result = subprocess.run([
            sys.executable, str(ROOT / "predict.py"), "--input", str(source), "--work-dir", str(tmp_path / "features"),
            "--structure-backend", "none", "--checkpoint", checkpoint, "--run-config", config,
            "--output", str(prediction_path), "--device", "cpu", "--symmetric",
        ], text=True, capture_output=True, timeout=180)
        assert result.returncode == 0, result.stderr
        prediction = pd.read_csv(prediction_path)
        assert prediction.prediction_status.eq("scored").all()
        assert np.isfinite(prediction.inducibility_score_mean).all()


def test_root_models_are_the_runtime_implementation():
    pytest.importorskip("dgl")
    import model
    from inference import _import_runtime
    assert model is _import_runtime()[4]


def test_freeze_random_features_requires_recognized_encoder():
    from types import SimpleNamespace
    from inference import freeze_molformer_random_features
    feature_map = SimpleNamespace(deterministic=False, orthogonal_random_weights=lambda: None)
    model = SimpleNamespace(molf=SimpleNamespace(modules=lambda: [feature_map]),
                            ablate=SimpleNamespace(use_cmp1d=True))
    assert freeze_molformer_random_features(model) == 1
    assert feature_map.deterministic is True
    model.molf.modules = lambda: []
    with pytest.raises(RuntimeError, match="cannot locate"):
        freeze_molformer_random_features(model)


def test_invalid_input_marks_existing_build_failed(tmp_path):
    root = tmp_path / "features"
    root.mkdir()
    (root / "feature_build_summary.json").write_text('{"status": "PASS"}')
    frame = raw()
    frame["protein1_sequence"] = "A" * 1201
    frame.to_csv(tmp_path / "invalid.csv", index=False)
    with pytest.raises(ValueError):
        prepare_input(tmp_path / "invalid.csv", root)
    assert json.loads((root / "feature_build_summary.json").read_text())["status"] == "FAIL"


def test_attention_dropout_is_disabled_in_eval():
    import torch
    from multimodal_fusion import MultiHeadAttention
    attention = MultiHeadAttention(16, 4, att_dropout=0.5).eval()
    inputs = torch.ones(2, 3, 16)
    assert torch.equal(attention(inputs), attention(inputs))
