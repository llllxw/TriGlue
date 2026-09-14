# TriGlue: training and prediction

Train TriGlue and score candidate triplets using a compound SMILES and two protein sequences. Binding-interface annotations and ternary structures are not required. Protein graphs are built from monomer PDB structures: supply existing structures or use the preprocessing ESMFold option to generate missing ones.

[中文说明](QUICKSTART.zh-CN.md)

## Install

```bash
conda env create -f environment.yml
conda activate triglue
python scripts/smoke_test.py
```

Use Linux and the supplied environment. Prepared-feature prediction supports CPU; memory requirements depend on sequence length. Prepare MoLFormer, ESM-2 and Uni-Mol2:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('ibm-research/MoLFormer-XL-both-10pct', revision='7b12d946c181a37f6012b9dc3b002275de070314', local_dir='models/molformer')"
python -c "import esm; esm.pretrained.esm2_t33_650M_UR50D()"
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('dptech/Uni-Mol2', filename='modelzoo/84M/checkpoint.pt', local_dir='models/unimol2')"
export UNIMOL_WEIGHT_DIR="$PWD/models/unimol2"
```

Keep `UNIMOL_WEIGHT_DIR` set when preprocessing. For the automatic structure-generation commands below, also prepare ESMFold; this step is unnecessary when complete matching PDB structures are supplied:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/esmfold_v1', local_dir='models/esmfold')"
```

## Data

Prediction CSV files require `smiles,protein1_sequence,protein2_sequence`. Training also requires `triplet_id,label`; identifiers must not overlap between training and validation, and labels must be 0 or 1. Use UTF-8 encoding.

`examples/train.csv` and `examples/val.csv` are synthetic examples for checking the workflow. Their labels have no experimental meaning; replace them with your labelled data for scientific use.

Provide sequences, not protein names. Use standard amino-acid letters and no more than 1200 residues per sequence. Compounds must contain no more than 256 atoms including hydrogens. Optional `protein1_structure` and `protein2_structure` columns accept PDB paths matching the complete sequences; use `--structure-backend none` for this route.

## Prepare features

```bash
python data_process.py --input examples/train.csv --output-root features/train \
  --device cuda --fold-device cuda --fold-model-path models/esmfold
python data_process.py --input examples/val.csv --output-root features/val \
  --device cuda --fold-device cuda --fold-model-path models/esmfold
```

Successful preparation writes `prepared_triplets.csv`. Add `--validate-only` to check input format only. If both protein PDB paths are supplied, replace `--fold-device cuda --fold-model-path models/esmfold` with `--structure-backend none`.

## Train

```bash
python train.py --train features/train/prepared_triplets.csv \
  --val features/val/prepared_triplets.csv \
  --feature-root features/train --val-feature-root features/val \
  --molformer-model models/molformer --device cuda --output-dir results/run1
```

Defaults are Adam, learning rate 1e-4, batch size 16 and at most 50 epochs, with early stopping after 20 epochs without improvement in validation accuracy. The loss is BCE plus 0.1 times InfoNCE. Training uses 15% SMILES token masking (80/10/10), 10% graph edge dropout and Gaussian noise with σ=0.02 on Uni-Mol2/ESM-2 embeddings. Validation and prediction disable augmentation. Add `--epochs 1` for a single-epoch execution check; this shortens the default training duration.

Use a new output directory for each run. Training saves `best.pth`, `run_config.json`, training records and validation predictions.

## Predict

```bash
python predict.py --input features/val/prepared_triplets.csv --feature-root features/val \
  --checkpoint results/run1/best.pth --run-config results/run1/run_config.json \
  --molformer-model models/molformer --device cpu --output results/predictions.csv
```

For new candidates, prepare their features as above and substitute the input and feature paths. `inducibility_score_mean` is the model score; `rank_within_protein_pair` ranks candidates within the same protein pair. The score is not a calibrated experimental success probability.

If execution fails, inspect the reported row/path and `feature_build_summary.json`. Common causes are missing model files, invalid inputs, sequence–structure mismatches and insufficient GPU memory.

## Uncertainty and interpretation

Train multiple models with different `--seed` values as above, for example in `results/run1` and `results/run2`. Each run saves `validation_predictions.csv`. Prepare features for a separate test set that does not overlap the validation set, then predict with each model:

```bash
for run in run1 run2; do
  python predict.py --input features/test/prepared_triplets.csv --feature-root features/test \
    --checkpoint results/$run/best.pth --run-config results/$run/run_config.json \
    --molformer-model models/molformer --device cpu --output results/$run/test_predictions.csv
done
```

Run calibration and uncertainty analysis. List validation and prediction files in matching model order; append more files for additional models:

```bash
python calibration.py \
  --validation results/run1/validation_predictions.csv results/run2/validation_predictions.csv \
  --predictions results/run1/test_predictions.csv results/run2/test_predictions.csv \
  --output-dir results/calibration
```

Outputs in `results/calibration` include calibrated predictions in CSV, temperature parameters, ECE/NLL/Brier metrics and reliability diagrams before and after calibration, and uncertainty and confidence distributions for correct and incorrect classifications. `ensemble_sd` is the sample standard deviation of individually calibrated model probabilities; `calibrated_confidence` is the calibrated classification confidence.

At least two models are required. Temperatures are fitted only on labelled validation predictions. Test predictions containing `label` produce all evaluation outputs; unlabelled predictions produce only calibrated predictions and temperatures. Use a new output directory.

Run atom/residue occlusion for one candidate:

```bash
python explain.py --input features/val/prepared_triplets.csv --feature-root features/val \
  --triplet-id val_1 --checkpoint results/run1/best.pth --run-config results/run1/run_config.json \
  --molformer-model models/molformer --device cpu --output-dir results/explanation_val1
```

Outputs include atom and residue importance CSV files, a highlight image of up to six influential atoms, and a residue-importance image. `delta_logit` is the baseline logit minus the occluded logit; `abs_delta_logit` is its absolute value. Atom occlusion zeros one molecular graph-node feature vector. Residue occlusion zeros one one-hot position, keeping the other channels fixed.
