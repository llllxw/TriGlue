#!/usr/bin/env python
"""Calibrate raw model predictions using a separate labelled validation set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import log_loss, brier_score_loss

from schema import read_csv, sha256_file
from uncertainty import ID_COLUMNS, _score_column


def apply_temperature(probabilities, temperature):
    return expit(logit(np.clip(probabilities,1e-7,1-1e-7))/temperature)


def fit_temperature(labels, probabilities):
    def objective(log_temperature):
        p = apply_temperature(probabilities,np.exp(log_temperature))
        return log_loss(labels,np.clip(p,1e-7,1-1e-7),labels=[0,1])
    result = minimize_scalar(objective,bounds=(-5,5),method='bounded')
    if not result.success:
        raise RuntimeError('temperature fitting failed')
    return float(np.exp(result.x))


def load_predictions(paths, require_labels=False):
    if len({str(Path(p).resolve()) for p in paths})!=len(paths):
        raise ValueError('supply one distinct file per model')
    reference = None
    values = []
    for path in paths:
        frame = read_csv(path)
        if any(c not in frame for c in ID_COLUMNS) or frame.empty or frame.triplet_id.duplicated().any():
            raise ValueError(f'{path}: require nonempty predictions with unique triplet_id and entity IDs')
        if 'n_models' in frame and not pd.to_numeric(frame.n_models).eq(1).all():
            raise ValueError('supply per-model raw predictions, not an ensemble mean')
        manifest = Path(str(path)+'.manifest.json')
        if manifest.exists():
            metadata = json.loads(manifest.read_text())
            if any(m.get('temperature',1)!=1 for m in metadata.get('models',[])):
                raise ValueError('supply raw predictions without temperature scaling')
        frame = frame.sort_values('triplet_id').reset_index(drop=True)
        label_col = 'label' if 'label' in frame else 'y_true' if 'y_true' in frame else None
        if label_col:
            labels = pd.to_numeric(frame[label_col],errors='coerce')
            if not labels.isin([0,1]).all():
                raise ValueError(f'{path}: labels must be binary')
            frame['label'] = labels.astype(int)
        elif require_labels:
            raise ValueError('validation predictions require ground-truth labels')
        if reference is None:
            reference = frame.copy()
        else:
            columns = list(ID_COLUMNS)+(['smiles'] if 'smiles' in reference else [])
            if any(c not in frame for c in columns) or not frame[columns].equals(reference[columns]):
                raise ValueError('prediction files must contain the same aligned triplets')
            if ('label' in frame)!=('label' in reference) or ('label' in frame and not frame.label.equals(reference.label)):
                raise ValueError('prediction labels do not agree across models')
        score = pd.to_numeric(frame[_score_column(frame)],errors='coerce').to_numpy(float)
        if not np.isfinite(score).all() or ((score<0)|(score>1)).any():
            raise ValueError('probabilities must be finite and within [0,1]')
        values.append(score)
    if require_labels and reference.label.nunique()!=2:
        raise ValueError('validation must contain both classes')
    return reference,np.column_stack(values)


def reliability(labels, probabilities):
    p = np.clip(probabilities,1e-7,1-1e-7)
    confidence = np.maximum(p,1-p)
    correct = (p>=0.5)==labels
    edges = np.linspace(.5,1,11)
    rows=[]
    for i,(low,high) in enumerate(zip(edges[:-1],edges[1:])):
        mask=(confidence>=low)&((confidence<=high) if i==9 else (confidence<high))
        rows.append(dict(bin=i+1,bin_lower=low,bin_upper=high,n=int(mask.sum()),
            mean_confidence=float(confidence[mask].mean()) if mask.any() else np.nan,
            observed_accuracy=float(correct[mask].mean()) if mask.any() else np.nan))
    bins=pd.DataFrame(rows)
    ece=float((bins.n/len(p)*(bins.mean_confidence-bins.observed_accuracy).abs().fillna(0)).sum())
    return bins,dict(ECE=ece,NLL=float(log_loss(labels,p,labels=[0,1])),Brier=float(brier_score_loss(labels,p)))


def plot_outputs(output, bins, metrics, predictions):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    for name,table in bins.items():
        fig,ax=plt.subplots(figsize=(4,4))
        good=table.n>0
        ax.plot([.5,1],[.5,1],'--',color='gray')
        ax.plot(table.loc[good,'mean_confidence'],table.loc[good,'observed_accuracy'],'o-')
        ax.set(xlim=(.5,1),ylim=(0,1),xlabel='Mean predicted-class confidence',ylabel='Observed accuracy',title=name.replace('_',' ').title())
        ax.text(.03,.97,'\n'.join(f'{k}={v:.4f}' for k,v in metrics[name].items()),transform=ax.transAxes,va='top')
        fig.tight_layout();fig.savefig(output/f'reliability_{name}.svg');plt.close(fig)
    for column,filename,title in [('ensemble_sd','uncertainty_comparison.svg','Prediction standard deviation'),
                                  ('calibrated_confidence','confidence_comparison.svg','Calibrated confidence')]:
        fig,ax=plt.subplots(figsize=(4,4))
        for position,(name,correct) in enumerate([('Correct',True),('Incorrect',False)],1):
            values=predictions.loc[predictions.correct.eq(correct),column].dropna().to_numpy()
            if len(values):ax.boxplot([values],positions=[position],widths=.5)
            else:ax.text(position,.5,'No samples',ha='center',transform=ax.get_xaxis_transform())
        ax.set_xticks([1,2]);ax.set_xticklabels(['Correct','Incorrect'])
        ax.set(xlim=(.5,2.5),ylabel=title)
        fig.tight_layout();fig.savefig(output/filename);plt.close(fig)


def analyze(validation_paths, prediction_paths, output_dir):
    if len(validation_paths)!=len(prediction_paths) or len(validation_paths)<2:
        raise ValueError('provide matched validation/prediction files for at least two models, in the same model order')
    output=Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('use a new output directory')
    val,v=load_predictions(validation_paths,require_labels=True)
    target,p=load_predictions(prediction_paths)
    if set(val.triplet_id)&set(target.triplet_id):
        raise ValueError('validation and prediction samples must be separate')
    def identities(frame):
        compound='smiles' if 'smiles' in frame else 'compound_id'
        return {(row[compound],*sorted((row['protein1_id'],row['protein2_id']))) for row in frame.to_dict('records')}
    if identities(val)&identities(target):
        raise ValueError('the same triplet occurs in validation and prediction data')
    temperatures=[fit_temperature(val.label.to_numpy(),v[:,i]) for i in range(v.shape[1])]
    ensemble_temperature=fit_temperature(val.label.to_numpy(),v.mean(axis=1))
    scaled=np.column_stack([apply_temperature(p[:,i],t) for i,t in enumerate(temperatures)])
    keep=list(ID_COLUMNS)+[c for c in ('smiles','label') if c in target]
    result=target[keep].copy()
    result['ensemble_mean_raw']=p.mean(axis=1)
    result['ensemble_mean_calibrated']=apply_temperature(p.mean(axis=1),ensemble_temperature)
    result['ensemble_sd_raw']=p.std(axis=1,ddof=1)
    result['ensemble_sd']=scaled.std(axis=1,ddof=1)
    result['calibrated_confidence']=np.maximum(result.ensemble_mean_calibrated,1-result.ensemble_mean_calibrated)
    result['predicted_label']=(result.ensemble_mean_calibrated>=.5).astype(int)
    result['n_models']=p.shape[1]
    bins={};metrics={}
    if 'label' in result:
        result['correct']=result.predicted_label.eq(result.label)
        for name,col in [('raw','ensemble_mean_raw'),('calibrated','ensemble_mean_calibrated')]:
            bins[name],metrics[name]=reliability(result.label.to_numpy(),result[col].to_numpy())
    output.mkdir(parents=True,exist_ok=True)
    result.to_csv(output/'calibrated_predictions.csv',index=False)
    config=dict(model_temperatures=temperatures,ensemble_temperature=ensemble_temperature,
        validation=[dict(path=str(p),sha256=sha256_file(Path(p))) for p in validation_paths],
        predictions=[dict(path=str(p),sha256=sha256_file(Path(p))) for p in prediction_paths])
    (output/'temperatures.json').write_text(json.dumps(config,indent=2)+'\n')
    if metrics:
        pd.DataFrame([dict(probabilities=k,**m) for k,m in metrics.items()]).to_csv(output/'calibration_metrics.csv',index=False)
        for name,table in bins.items():table.to_csv(output/f'reliability_{name}.csv',index=False)
        plot_outputs(output,bins,metrics,result)
    return dict(status='PASS',n_models=p.shape[1],n_samples=len(result),labelled_evaluation=bool(metrics),output_dir=str(output))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation',nargs='+',required=True)
    parser.add_argument('--predictions',nargs='+',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args(argv)
    print(json.dumps(analyze(args.validation,args.predictions,args.output_dir),indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
