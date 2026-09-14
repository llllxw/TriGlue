from __future__ import annotations

import gc
import json
import math
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from schema import audit_triplets, read_csv, sha256_file
from uncertainty import ensemble_summary


def _import_runtime():
    try:
        import dgl
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError(
            "Full inference requires the pinned PyTorch/DGL environment. "
            "Create it with: conda env create -f environment.yml"
        ) from exc
    import gragh_model as graph_model
    import model as model_module
    import multimodal_fusion as fusion_module
    return torch, dgl, DataLoader, Dataset, model_module, graph_model, fusion_module


def _resolve_device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
    return torch.device(requested)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "ablation_config" not in payload:
        raise ValueError(f"run config lacks ablation_config: {path}")
    return payload


def _load_state(torch, path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    output = np.empty_like(values, dtype=float)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def freeze_molformer_random_features(model) -> int:
    """Use checkpoint buffers, not fresh random projections on each eval call.

    This is an explicit deployment protocol change from historical stochastic
    evaluation. It does not alter weights and must be recorded in outputs.
    """
    changed = 0
    for module in model.molf.modules():
        if hasattr(module, "orthogonal_random_weights") and hasattr(module, "deterministic"):
            module.deterministic = True
            changed += 1
    if model.ablate.use_cmp1d and changed == 0:
        raise RuntimeError("cannot locate MoLFormer random-feature maps; inspect the encoder revision")
    return changed


def _move_batch(batch, device):
    smiles, compound_graph, compound_embedding, p1_one_hot, p1_graph, p1_esm2, p2_one_hot, p2_graph, p2_esm2 = batch
    return (
        smiles,
        compound_graph.to(device),
        compound_embedding.to(device),
        p1_one_hot.to(device),
        p1_graph.to(device),
        p1_esm2.to(device),
        p2_one_hot.to(device),
        p2_graph.to(device),
        p2_esm2.to(device),
    )


def _score_one_checkpoint(
    frame: pd.DataFrame,
    feature_root: Path,
    checkpoint: Path,
    run_config: Path,
    *,
    batch_size: int,
    device_name: str,
    temperature: float,
    cache_size: int,
    molformer_model: str,
    runtime_cache: dict | None = None,
) -> tuple[np.ndarray, dict]:
    torch, dgl, DataLoader, Dataset, model_module, graph_model, fusion_module = _import_runtime()
    device = _resolve_device(torch, device_name)
    model_module.device = device
    graph_model.device = device
    fusion_module.device = device
    model_module.TriComplexClassifier.MOLFORMER = molformer_model
    key = (str(checkpoint), sha256_file(checkpoint), str(run_config), sha256_file(run_config),
           str(device), molformer_model)
    if runtime_cache is None:
        runtime_cache = {}
    if key in runtime_cache:
        model, missing = runtime_cache[key]
    else:
        config = _load_config(run_config)
        allowed = {item.name for item in fields(model_module.AblationConfig)}
        unknown = set(config["ablation_config"]) - allowed
        if unknown:
            raise ValueError(f"unsupported checkpoint configuration fields: {sorted(unknown)}")
        ablation = model_module.AblationConfig(
            **{key: value for key, value in config["ablation_config"].items() if key in allowed}
        )
        dropout = float(config.get("dropout", 0.1))
        model = model_module.TriComplexClassifier(
            dropout=dropout,
            prot1d_dim=25,
            use_checkpoint="never",
            enable_smiles_cache=True,
            ablate=ablation,
        )
        state = _load_state(torch, checkpoint)
        incompatibility = model.load_state_dict(state, strict=False)
        allowed_missing = {
            "compound_object_norm.weight",
            "compound_object_norm.bias",
            "protein_object_norm.weight",
            "protein_object_norm.bias",
        }
        missing = set(incompatibility.missing_keys)
        unexpected = set(incompatibility.unexpected_keys)
        if unexpected or not missing.issubset(allowed_missing):
            raise RuntimeError(
                "checkpoint architecture mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        model = model.to(device).eval()
        freeze_molformer_random_features(model)
        runtime_cache[key] = (model, missing)
    from Dataset import PreparedFeatureDataset as DatasetClass
    dataset = DatasetClass(frame, feature_root, cache_size=cache_size)
    logits = np.empty(len(frame), dtype=float)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    pair_groups = frame.groupby(["protein1_id", "protein2_id"], sort=False).indices
    with torch.inference_mode():
        for indices in pair_groups.values():
            ordered_indices = np.asarray(indices, dtype=int)
            subset = torch.utils.data.Subset(dataset, ordered_indices.tolist())
            loader = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=DatasetClass.collate,
                pin_memory=device.type == "cuda",
            )
            offset = 0
            for batch in loader:
                output = model(*_move_batch(batch, device))
                values = output.detach().cpu().numpy().reshape(-1)
                if not np.isfinite(values).all():
                    raise ValueError("model returned nonfinite logits; no prediction CSV will be written")
                selected = ordered_indices[offset : offset + len(values)]
                logits[selected] = values
                offset += len(values)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    scores = _sigmoid(logits / temperature)
    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "run_config": str(run_config),
        "temperature": temperature,
        "device": str(device),
        "molformer_model": molformer_model,
        "n_triplets": len(frame),
        "n_protein_pair_groups": len(pair_groups),
        "batch_size": batch_size,
        "model_seconds": elapsed,
        "triplets_per_second": len(frame) / elapsed if elapsed else math.inf,
        "identity_layernorm_migration": sorted(missing),
        "molformer_eval_protocol": "checkpoint_fixed_random_features_v1",
        "attention_dropout_protocol": "disabled_in_eval",
    }
    del dataset
    return scores, metadata


def predict_prepared(
    input_csv: str | Path,
    feature_root: str | Path,
    checkpoints: Sequence[str | Path],
    run_configs: Sequence[str | Path],
    output_csv: str | Path,
    *,
    batch_size: int = 16,
    device: str = "auto",
    temperatures: Sequence[float] | None = None,
    symmetric: bool = False,
    cache_size: int = 64,
    molformer_model: str = "ibm-research/MoLFormer-XL-both-10pct",
    runtime_cache: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    if batch_size < 1 or cache_size < 1:
        raise ValueError("batch_size and cache_size must be positive")
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    checkpoint_hashes = [sha256_file(Path(p).expanduser().resolve()) for p in checkpoints]
    if len(set(checkpoint_hashes)) != len(checkpoint_hashes):
        raise ValueError('duplicate checkpoint weights cannot estimate ensemble uncertainty')
    if len(run_configs) == 1 and len(checkpoints) > 1:
        run_configs = list(run_configs) * len(checkpoints)
    if len(run_configs) != len(checkpoints):
        raise ValueError("supply one --run-config per checkpoint, or one shared run config")
    if temperatures is None or len(temperatures) == 0:
        temperatures = [1.0] * len(checkpoints)
    elif len(temperatures) == 1 and len(checkpoints) > 1:
        temperatures = list(temperatures) * len(checkpoints)
    if len(temperatures) != len(checkpoints):
        raise ValueError("supply one --temperature per checkpoint, or one shared temperature")
    if any(value <= 0 or not np.isfinite(value) for value in temperatures):
        raise ValueError("temperatures must be finite and greater than zero")
    frame = read_csv(input_csv)
    if "preparation_status" in frame and not frame["preparation_status"].eq("ready").all():
        raise ValueError("input contains failed/partial preparation; full inference requires all features")
    build_summary = Path(feature_root) / "feature_build_summary.json"
    if build_summary.is_file():
        build = json.loads(build_summary.read_text())
        if build.get("status") != "PASS":
            raise ValueError("feature root has an incomplete/failed build; inspect feature_build_summary.json")
    audit = audit_triplets(frame, feature_root=feature_root)
    if audit["status"] != "PASS":
        preview = audit["issues"][:10]
        raise ValueError(f"input/feature audit failed: {json.dumps(preview, ensure_ascii=False)}")
    feature_root = Path(feature_root).expanduser().resolve()
    all_scores = []
    model_metadata = []
    reversed_frame = None
    if symmetric:
        reversed_frame = frame.copy()
        reversed_frame[["protein1_id", "protein2_id"]] = frame[
            ["protein2_id", "protein1_id"]
        ].to_numpy()
    for checkpoint, config, temperature in zip(checkpoints, run_configs, temperatures):
        current_cache = runtime_cache if runtime_cache is not None else {}
        score, metadata = _score_one_checkpoint(
            frame,
            feature_root,
            Path(checkpoint).expanduser().resolve(),
            Path(config).expanduser().resolve(),
            batch_size=batch_size,
            device_name=device,
            temperature=float(temperature),
            cache_size=cache_size,
            molformer_model=molformer_model,
            runtime_cache=current_cache,
        )
        if reversed_frame is not None:
            reversed_score, reverse_metadata = _score_one_checkpoint(
                reversed_frame,
                feature_root,
                Path(checkpoint).expanduser().resolve(),
                Path(config).expanduser().resolve(),
                batch_size=batch_size,
                device_name=device,
                temperature=float(temperature),
                cache_size=cache_size,
                molformer_model=molformer_model,
                runtime_cache=current_cache,
            )
            score = 0.5 * (score + reversed_score)
            metadata["reverse_order_model_seconds"] = reverse_metadata["model_seconds"]
        all_scores.append(score)
        model_metadata.append(metadata)
        if runtime_cache is None:
            current_cache.clear()
            gc.collect()
    values = np.vstack(all_scores).T
    output = frame.copy()
    output["prediction_status"] = "scored"
    for name, value in ensemble_summary(values).items():
        output[name] = value
    output["symmetric_protein_order"] = bool(symmetric)
    output["rank_global"] = output["inducibility_score_mean"].rank(
        method="min", ascending=False
    ).astype(int)
    pair_key = output.apply(
        lambda row: "__".join(sorted([str(row["protein1_id"]), str(row["protein2_id"])])),
        axis=1,
    )
    output["rank_within_protein_pair"] = output.groupby(pair_key)["inducibility_score_mean"].rank(
        method="min", ascending=False
    ).astype(int)
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False, encoding="utf-8")
    manifest = {
        "status": "PASS",
        "input": str(Path(input_csv).resolve()),
        "feature_root": str(feature_root),
        "output": str(destination.resolve()),
        "n_triplets": len(frame),
        "n_models": len(checkpoints),
        "symmetric": symmetric,
        "models": model_metadata,
    }
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output, manifest
