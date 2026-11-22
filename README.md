# TriGlue Enables Interpretable Prioritization of Molecular Glue-Induced Ternary Complex Formation

Train TriGlue on the 7,484 compound–protein–protein triplets in Supplementary Table 1 and rank held-out triplets by calibrated score. For interactive prediction from a SMILES string and two proteins, download [TriGlue Predictor for Windows or Linux](https://github.com/llllxw/TriGlue/releases/tag/v0.1.0).

## Install

```bash
conda env create -f environment.yml
conda activate triglue
python -m pip install --no-deps unimol-tools==0.1.3.post1
python verify_inputs.py
```

Extract the three `TriGlueDesktop-{data,encoders,models}-v0.1.0.tar.gz` archives into the repository directory. They provide the prepared features, Uni-Mol2 encoder and MoLFormer snapshot used below. The desktop classifiers in the models archive are used by the desktop application. [Download resources](https://github.com/llllxw/TriGlue/releases/tag/v0.1.0)

## Data

| File in `data/` | Contents |
| --- | --- |
| `dataset.csv` | Supplementary Table 1, including triplets, labels and sources |
| `prepared_triplets.csv` | Training input with protein sequences, feature identifiers, labels and parent group for each triplet |
| `sequences.csv` | Protein identifier, sequence and checksum |
| `groups.csv` | Parent group for each sample |
| `parent_links.csv` | Links from generated negatives to their contributing positive parents |

The fixed split (Section 2.1) and five cross-validation folds (Section 2.5) are in `splits/`. Each parent group belongs to one partition per split. Use the following Bash commands to write the corresponding input tables.

```bash
FEATURES=resources/features
UNIMOL=resources/encoders/unimol
MOLFORMER=resources/huggingface/hub/models--ibm-research--MoLFormer-XL-both-10pct/snapshots/7b12d946c181a37f6012b9dc3b002275de070314
python run_cv.py --feature-root "$FEATURES" --unimol-dir "$UNIMOL" \
  --molformer-model "$MOLFORMER" --output-dir results/plan --dry-run
```

## Train

Train one fold with the settings in `config.json`. For the single fixed split, use `results/plan/inputs/fixed` in place of `results/plan/inputs/fold1` throughout.

```bash
FOLD=results/plan/inputs/fold1
python train.py --train "$FOLD/train.csv" --val "$FOLD/val.csv" \
  --feature-root "$FEATURES" --unimol-dir "$UNIMOL" --molformer-model "$MOLFORMER" \
  --device cpu --output-dir results/model
```

MoLFormer-XL, ESM-2 and Uni-Mol2 are frozen. Training updates the TextCNN, both GINs, projections, fusion layers and classifier. Adam uses a learning rate of 1e-4 and batch size 16. Training runs for up to 50 epochs and stops after 20 consecutive epochs without an improvement in validation accuracy. Training applies SMILES masking, graph-edge dropout and Gaussian noise to compound coordinates before Uni-Mol2 encoding.

## Predict

```bash
python predict.py --input "$FOLD/test.csv" --feature-root "$FEATURES" \
  --unimol-dir "$UNIMOL" --molformer-model "$MOLFORMER" --checkpoint results/model/best.pth \
  --run-config results/model/run_config.json --output results/test.csv
```

`inducibility_score_mean` is the positive-class probability after temperature scaling on validation data. To score an ensemble, repeat the `--checkpoint` and `--run-config` arguments in matching order. The output includes the mean probability, sample standard deviation and empirical 2.5th and 97.5th percentiles across models.

## Evaluate

```bash
python evaluate.py --input results/test.csv --score-column inducibility_score_mean \
  --output results/test_metrics.csv
```

For all five folds and five seeds, run `run_cv.py` with the same resource paths, a new output directory and no `--dry-run`. It writes per-fold metrics, pooled out-of-fold predictions for each seed and `five_seed_summary.csv`.

The reconstructed fixed split contains 4,491/1,506/1,487 training/validation/test samples. The five folds average 4,488.0/1,499.2/1,496.8. The manuscript reports 4,489/1,498/1,497 for both. Table S2 summarizes archived runs with an earlier dataset and a different checkpoint-selection rule. See the [split manifest](splits/manifest.json) and [experiment provenance](provenance.json).
