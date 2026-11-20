"""Train and evaluate the fixed five folds and five seeds from config.json."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from evaluate import evaluate
from verify_inputs import verify
from uncertainty import ensemble_summary
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-root', required=True)
    parser.add_argument('--molformer-model', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--dry-run', action='store_true', help='write commands without training or inference')
    args = parser.parse_args()
    verify(root / 'data/dataset.csv', root / 'splits', root / 'provenance.json')
    settings = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    prepared = pd.read_csv(root / 'data/prepared_triplets.csv', dtype=str, keep_default_na=False)
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    plan = []
    for seed in settings['seeds']:
        for fold in range(1, 6):
            run = output / ('seed' + str(seed)) / ('fold' + str(fold))
            inputs = output / 'inputs' / ('fold' + str(fold))
            commands = [
                [sys.executable, str(root / 'train.py'), '--train', str(inputs / 'train.csv'),
                 '--val', str(inputs / 'val.csv'), '--feature-root', args.feature_root,
                 '--molformer-model', args.molformer_model, '--output-dir', str(run),
                 '--seed', str(seed), '--device', args.device,'--cached-features-only'],
                [sys.executable, str(root / 'predict.py'), '--input', str(inputs / 'test.csv'),
                 '--feature-root', args.feature_root, '--checkpoint', str(run / 'best.pth'),
                 '--run-config', str(run / 'run_config.json'), '--molformer-model', args.molformer_model,
                 '--device', args.device, '--output', str(run / 'test_predictions.csv')],
            ]
            plan.append({'seed': seed, 'fold': fold, 'commands': commands})
    (output / 'run_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    for fold in range(1, 6):
        partitions = pd.read_csv(root / 'splits' / ('fold' + str(fold) + '.csv'), dtype=str)
        inputs = output / 'inputs' / ('fold' + str(fold)); inputs.mkdir(parents=True, exist_ok=True)
        for partition in ['train', 'val', 'test']:
            ids = set(partitions.loc[partitions.partition.eq(partition), 'sample_id'])
            prepared.loc[prepared.triplet_id.isin(ids)].to_csv(inputs / (partition + '.csv'), index=False)
    if args.dry_run:
        print(json.dumps({'status': 'PLAN_WRITTEN', 'runs': len(plan), 'commands': sum(len(r['commands']) for r in plan)}))
        return
    if not Path(args.feature_root).is_dir() or not Path(args.molformer_model).is_dir():
        raise FileNotFoundError('Provide the complete feature bundle and local MoLFormer snapshot')
    for item in plan:
        for command in item['commands']:
            subprocess.run(command, check=True, cwd=str(root))
    metrics = [];seed_oof=[]
    for seed in settings['seeds']:
        frames = [pd.read_csv(output / ('seed' + str(seed)) / ('fold' + str(fold)) / 'test_predictions.csv')
                  for fold in range(1, 6)]
        oof = pd.concat(frames, ignore_index=True)
        if oof.triplet_id.duplicated().any() or set(oof.triplet_id) != set(prepared.triplet_id):
            raise ValueError('OOF predictions must cover each final-data sample once')
        oof = oof.sort_values('triplet_id').reset_index(drop=True)
        seed_oof.append(oof)
        oof.to_csv(output / ('oof_seed' + str(seed) + '.csv'), index=False)
        metrics.append({'seed': seed, **evaluate(oof.label, oof.inducibility_score_mean)})
    pd.DataFrame(metrics).to_csv(output / 'seed_metrics.csv', index=False)
    ensemble=seed_oof[0].drop(columns=['score_model_1']).copy()
    if any(not f.triplet_id.equals(ensemble.triplet_id) or not f.label.equals(ensemble.label) for f in seed_oof):
        raise ValueError('Seed OOF identities or labels differ')
    values=np.stack([f.inducibility_score_mean.to_numpy() for f in seed_oof],axis=1)
    for name,value in ensemble_summary(values).items():ensemble[name]=value
    ensemble['raw_score_mean']=np.mean([f.raw_score_mean.to_numpy() for f in seed_oof],axis=0)
    for seed,frame in zip(settings['seeds'],seed_oof):ensemble['score_seed_'+str(seed)]=frame.inducibility_score_mean
    ensemble.to_csv(output/'oof_ensemble.csv',index=False)
    pd.DataFrame([{'probabilities':column,**evaluate(ensemble.label,ensemble[column])}
                  for column in ['raw_score_mean','inducibility_score_mean']]).to_csv(output/'ensemble_metrics.csv',index=False)
    print(json.dumps({'status': 'COMPLETE', 'output': str(output)}))


if __name__ == '__main__':
    main()
