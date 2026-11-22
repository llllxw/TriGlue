"""Predict labelled or unlabelled triplets from fixed cached features."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference import predict_prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--feature-root', required=True)
    parser.add_argument('--checkpoint', action='append', required=True)
    parser.add_argument('--run-config', action='append', required=True)
    parser.add_argument('--molformer-model', required=True)
    parser.add_argument('--unimol-dir', required=True)
    parser.add_argument('--temperature', type=float, action='append')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    defaults = json.loads(Path(__file__).with_name('config.json').read_text(encoding='utf-8'))
    parser.add_argument('--batch-size', type=int, default=defaults['batch_size'])
    args = parser.parse_args()
    _, manifest = predict_prepared(args.input, args.feature_root,
        args.checkpoint, args.run_config, args.output,
        molformer_model=args.molformer_model, device=args.device,
        batch_size=args.batch_size, temperatures=args.temperature, unimol_dir=args.unimol_dir)
    print(json.dumps({'status':manifest['status'], 'rows':manifest['n_triplets'],
                      'models':manifest['n_models'], 'output':manifest['output']}))


if __name__ == '__main__':
    main()
