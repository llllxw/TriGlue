#!/usr/bin/env python
"""Train TriGlue from prepared, labelled train and validation tables."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from schema import audit_triplets, read_csv, sha256_file


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', required=True, help='prepared training CSV with label')
    p.add_argument('--val', required=True, help='prepared validation CSV with label')
    p.add_argument('--feature-root', required=True)
    p.add_argument('--val-feature-root', help='validation features; defaults to --feature-root')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--molformer-model', required=True, help='local MoLFormer snapshot')
    p.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto')
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=20)
    return p


def read_training_table(path, feature_root):
    frame = read_csv(path)
    if 'label' not in frame:
        raise ValueError(f'{path}: missing binary label column')
    labels = pd.to_numeric(frame.label, errors='coerce')
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(f'{path}: label must contain only 0 or 1')
    if labels.nunique() != 2:
        raise ValueError(f'{path}: include both positive and negative examples')
    if 'preparation_status' in frame and not frame.preparation_status.eq('ready').all():
        raise ValueError(f'{path}: incomplete feature preparation')
    frame['label'] = labels.astype(int)
    report = audit_triplets(frame, feature_root=feature_root)
    if report['status'] != 'PASS':
        raise ValueError(f'{path}: {report["issues"][:5]}')
    return frame


def validate_split(train, val):
    if set(train.triplet_id) & set(val.triplet_id):
        raise ValueError('training and validation triplet_id overlap')
    def keys(frame):
        return {(row.smiles, *sorted((row.protein1_id, row.protein2_id)))
                for row in frame.itertuples()}
    if keys(train) & keys(val):
        raise ValueError('the same compound/protein triplet occurs in train and validation')


def main(argv=None):
    args = parser().parse_args(argv)
    if args.epochs < 1 or args.batch_size < 2 or args.patience < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('epochs/patience must be positive, batch-size >= 2, lr finite and positive')
    if not Path(args.molformer_model).is_dir():
        raise FileNotFoundError('provide a local MoLFormer directory using --molformer-model')
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('output-dir is not empty; select a new run directory')
    train = read_training_table(args.train, args.feature_root)
    val_root = args.val_feature_root or args.feature_root
    val = read_training_table(args.val, val_root)
    validate_split(train, val)

    import torch
    import dgl
    from torch.utils.data import DataLoader
    from Dataset import LabelledFeatureDataset
    from inference import _import_runtime, _move_batch, _resolve_device, freeze_molformer_random_features
    _, _, _, _, model_module, graph_module, fusion_module = _import_runtime()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dgl.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _resolve_device(torch, args.device)
    for module in (model_module, graph_module, fusion_module):
        module.device = device
    model_module.TriComplexClassifier.MOLFORMER = str(Path(args.molformer_model).resolve())
    ablation = model_module.AblationConfig(use_cmp1d=True)
    model = model_module.TriComplexClassifier(prot1d_dim=25, dropout=0.2,
        ablate=ablation, enable_smiles_cache=False, use_checkpoint='never').to(device)
    # Use the same deterministic frozen encoder protocol as prediction.
    freeze_molformer_random_features(model)
    model.smiles_mask_rate = 0.15
    root = Path(args.feature_root).resolve()
    training = LabelledFeatureDataset(train, root, training=True)
    validation = LabelledFeatureDataset(val, Path(val_root).resolve(), training=False)
    # BatchNorm requires >=2 examples; avoid dropping a one-row remainder.
    batches = TrainingBatches(len(training), args.batch_size, args.seed)
    loader = DataLoader(training, batch_sampler=batches, collate_fn=training.collate, num_workers=0)
    val_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False,
                            collate_fn=validation.collate, num_workers=0)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    output.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), dropout=0.2, alpha=0.1, optimizer='adam', weight_decay=0.0,
                  smiles_mask_rate=0.15, edge_drop=0.10, embedding_noise_sigma=0.02,
                  ablation_config=asdict(ablation), selection_metric='validation_accuracy',
                  molformer_eval_protocol='checkpoint_fixed_random_features_v1',
                  train_sha256=sha256_file(Path(args.train)), val_sha256=sha256_file(Path(args.val)))
    (output/'run_config.json').write_text(json.dumps(config, indent=2)+'\n')
    best = -1.0
    stale = 0
    history = []
    for epoch in range(1, args.epochs+1):
        model.train()
        model.molf.eval()
        total, n = 0.0, 0
        for features, labels in loader:
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, contrastive = model(*_move_batch(features, device), return_contrastive=True,
                                        contrastive_mask=labels.eq(1))
            loss = criterion(logits.reshape(-1), labels) + 0.1 * contrastive
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()*len(labels)
            n += len(labels)
        model.eval()
        scores = []
        with torch.inference_mode():
            for features, _ in val_loader:
                logits = model(*_move_batch(features, device)).reshape(-1)
                scores.extend(torch.sigmoid(logits).cpu().tolist())
        if not np.isfinite(scores).all():
            raise RuntimeError('nonfinite validation score')
        accuracy = float(np.mean((np.array(scores)>=0.5)==val.label.to_numpy()))
        history.append(dict(epoch=epoch, train_loss=total/n, validation_accuracy=accuracy))
        pd.DataFrame(history).to_csv(output/'training_history.csv', index=False)
        if accuracy > best:
            best, stale = accuracy, 0
            torch.save(model.state_dict(), output/'best.pth')
            predicted = val.copy()
            predicted['inducibility_score'] = scores
            predicted.to_csv(output/'validation_predictions.csv', index=False)
            (output/'training_summary.json').write_text(json.dumps(dict(best_epoch=epoch,
                best_validation_accuracy=best, selection_metric='validation_accuracy'),indent=2)+'\n')
        else:
            stale += 1
        print(f'Epoch {epoch}: loss={total/n:.4f}, validation_ACC={accuracy:.4f}', flush=True)
        if stale >= args.patience:
            break
    print(f'TRAINING_COMPLETE: {output / "best.pth"}', flush=True)
    return 0


class TrainingBatches:
    """Shuffle each epoch and distribute a singleton across adjacent batches."""
    def __init__(self, size, batch_size, seed):
        self.size, self.batch_size = size, batch_size
        self.rng = np.random.default_rng(seed)
    def __iter__(self):
        indices = self.rng.permutation(self.size).tolist()
        batches = [indices[i:i+self.batch_size] for i in range(0,self.size,self.batch_size)]
        if len(batches)>1 and len(batches[-1])==1:
            if len(batches[-2])>2:
                batches[-1].insert(0,batches[-2].pop())
            else:
                batches[-2].extend(batches.pop())
        return iter(batches)
    def __len__(self):
        count = (self.size+self.batch_size-1)//self.batch_size
        return count-1 if self.batch_size==2 and self.size>2 and self.size%2 else count


if __name__ == '__main__':
    raise SystemExit(main())
