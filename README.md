# TriGlue

Manuscript-based reference implementation with one model (`model.py`), one parameter configuration (`config.json`), the final 7,484-record dataset, and fixed parent-grouped partitions.

## Setup

```bash
conda env create -f environment.yml
conda activate triglue
python verify_inputs.py
```

The supplied environment runs on CPU. Extract the data and encoder [resource archives](https://github.com/llllxw/TriGlue/releases/tag/v0.1.0). Replace `FEATURES` with the extracted `features` directory and `MOLFORMER` with the local MoLFormer snapshot directory containing `config.json` and the encoder weights.

## Run

Create the fixed input partitions and commands for five folds and five seeds:

```bash
python run_cv.py --feature-root FEATURES --molformer-model MOLFORMER --device cpu --output-dir results/plan --dry-run
```

Train, predict and evaluate one fold:

```bash
python train.py --train results/plan/inputs/fold1/train.csv --val results/plan/inputs/fold1/val.csv --feature-root FEATURES --molformer-model MOLFORMER --cached-features-only --device cpu --output-dir results/model
python predict.py --input results/plan/inputs/fold1/test.csv --feature-root FEATURES --molformer-model MOLFORMER --checkpoint results/model/best.pth --run-config results/model/run_config.json --device cpu --output results/predictions.csv
python evaluate.py --input results/predictions.csv --output results/metrics.csv
```

To execute all 25 runs, omit `--dry-run` and use a new output directory. This produces per-seed and five-seed ensemble out-of-fold predictions and evaluation metrics. Training uses at most 50 epochs, batch size 16, Adam at 1e-4, and validation-accuracy early stopping with patience 20. Temperature scaling is fitted only on validation data.

## Scope

The partitions were reconstructed from released parent groups; they have not been verified as the historical experiment partitions. Cached-feature training applies SMILES masking and graph-edge dropout; raw-coordinate Gaussian augmentation requires coordinate resources absent from the released cache. This model requires newly trained checkpoints and is incompatible with the desktop Release classifier weights. Published paper metrics have not been numerically reproduced with this implementation.
