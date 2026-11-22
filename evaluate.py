"""Evaluate labelled predictions, including candidate-ranking metrics."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)


def evaluate(labels, scores):
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if len(labels) == 0 or len(labels) != len(scores):
        raise ValueError('Require nonempty, aligned labels and scores')
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError('Require binary labels and finite scores')
    if ((scores < 0) | (scores > 1)).any() or len(np.unique(labels)) != 2:
        raise ValueError('Require probabilities in [0,1] and both label classes')
    labels = labels.astype(int)
    predicted = (scores >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    result = dict(n=len(labels), positives=int(labels.sum()), AUROC=float(roc_auc_score(labels, scores)),
                  AP=float(average_precision_score(labels, scores)),
                  accuracy=float(accuracy_score(labels, predicted)),
                  precision=float(precision_score(labels, predicted, zero_division=0)),
                  recall=float(recall_score(labels, predicted, zero_division=0)),
                  specificity=float(tn / (tn + fp)), F1=float(f1_score(labels, predicted, zero_division=0)),
                  balanced_accuracy=float(balanced_accuracy_score(labels, predicted)))
    clipped=np.clip(scores,1e-12,1-1e-12)
    result['NLL']=float(np.mean(-labels*np.log(clipped)-(1-labels)*np.log1p(-clipped)))
    result['Brier']=float(np.mean((scores-labels)**2))
    confidence = np.maximum(scores, 1.0 - scores)
    correct = predicted == labels
    bins = np.minimum((confidence * 15).astype(int), 14)
    result['ECE'] = float(sum(
        np.mean(bins == b) * abs(confidence[bins == b].mean() - correct[bins == b].mean())
        for b in np.unique(bins)))
    # Stable sorting provides deterministic tie handling; sample ordering must be fixed.
    ordered = labels[np.argsort(-scores, kind='stable')]
    for fraction in [.01, .02, .05, .10, .15, .20, .30, .50]:
        count = max(1, math.ceil(len(labels) * fraction))
        precision = float(ordered[:count].mean())
        suffix = str(int(round(100 * fraction)))
        result['precision_top_' + suffix + 'pct'] = precision
        result['EF_top_' + suffix + 'pct'] = precision / float(labels.mean())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', required=True, help='one CSV per run, or one pooled OOF CSV')
    parser.add_argument('--output', required=True)
    parser.add_argument('--score-column')
    args = parser.parse_args()
    records = []
    for filename in args.input:
        frame = pd.read_csv(filename)
        label = 'label' if 'label' in frame else 'y_true'
        candidates = [c for c in ['inducibility_score', 'inducibility_score_mean', 'y_prob', 'score'] if c in frame]
        column = args.score_column
        if not column:
            if len(candidates) != 1:
                raise ValueError('Choose --score-column explicitly: ' + str(candidates))
            column = candidates[0]
        records.append(dict(input=str(Path(filename).resolve()), **evaluate(frame[label], frame[column])))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False, encoding='utf-8')
    print(json.dumps(dict(output=str(output), runs=len(records))))


if __name__ == '__main__':
    main()
