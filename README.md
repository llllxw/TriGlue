# TriGlue final-data retraining

Train one three-entity model on the final 7,484-row Supplementary Table 1. The fixed partitions group each parent triplet with its derived negatives. The five seeds and training hyperparameters are defined in `config.json`.

## Setup

Create the CPU environment with `conda env create -f environment.yml` and activate it with `conda activate triglue`. Install `python -m pip install --no-deps unimol-tools==0.1.3.post1` to preserve the pinned RDKit build. Run `python verify_inputs.py`.

Extract the resource archives into one directory. Substitute these paths in the commands below. The archives are available from the [TriGlue release](https://github.com/llllxw/TriGlue/releases/tag/v0.1.0).

| Argument placeholder | Extracted path | Archive |
| --- | --- | --- |
| `FEATURES` | `resources/features` | `TriGlueDesktop-data-v0.1.0.tar.gz` |
| `UNIMOL` | `resources/encoders/unimol` | `TriGlueDesktop-encoders-v0.1.0.tar.gz` |
| `MOLFORMER` | `resources/huggingface/hub/models--ibm-research--MoLFormer-XL-both-10pct/snapshots/7b12d946c181a37f6012b9dc3b002275de070314` | `TriGlueDesktop-models-v0.1.0.tar.gz` |

The research workflow uses the MoLFormer snapshot from the last archive and trains the TriGlue classifier locally.

## Train and evaluate

Write the 25 commands and fixed training, validation and test tables:

```bash
python run_cv.py --feature-root FEATURES --molformer-model MOLFORMER --unimol-dir UNIMOL --output-dir results/plan --dry-run
```

The `inputs/fixed` directory holds the single split reconstructed using the rule in Section 2.1. It contains 4,491 training, 1,506 validation and 1,487 test samples. The five `inputs/fold*` directories follow the selection rule in Section 2.5 and average 4,488.0, 1,499.2 and 1,496.8 samples, respectively. The manuscript reports 4,489, 1,498 and 1,497 in both sections.

Run one cross-validation fold:

```bash
python train.py --train results/plan/inputs/fold1/train.csv --val results/plan/inputs/fold1/val.csv --feature-root FEATURES --molformer-model MOLFORMER --unimol-dir UNIMOL --output-dir results/model --device cpu
python predict.py --input results/plan/inputs/fold1/test.csv --feature-root FEATURES --molformer-model MOLFORMER --unimol-dir UNIMOL --checkpoint results/model/best.pth --run-config results/model/run_config.json --output results/test.csv
python evaluate.py --input results/test.csv --score-column inducibility_score_mean --output results/test_metrics.csv
```

For a complete experiment run `run_cv.py` with a fresh output directory and omit `--dry-run`. The results include fold-level scores, pooled out-of-fold predictions for each seed, the mean and sample SD across those five metrics in `five_seed_summary.csv`, and a separate ensemble analysis. Raw probabilities are the primary ranking scores for comparison with historical Table S2. The validation-set temperature is used for calibrated scores. Each trained classifier checkpoint is written locally.

## Implementation

MoLFormer-XL, ESM-2 and Uni-Mol2 are frozen. The protein TextCNN, both GINs, projections, fusion and prediction layers are trained. Training applies SMILES masking and graph-edge dropout. RDKit builds an ETKDGv3 conformer for each molecular SMILES, initialized with seed 42 and MMFF-optimized when parameterized. On each training occurrence, Gaussian noise with standard deviation 0.02 Å perturbs the heavy-atom coordinates immediately before frozen Uni-Mol2 encoding. Validation and prediction encode the same clean conformers. Protein structures contribute through the released residue graphs, and ESM-2 takes protein **sequences**. A failure to generate a 3D conformer stops the run and identifies the input.

The 21-channel protein TextCNN input excludes unknown residues. The fixed ESM-2 and protein-graph features retain the source sequence and structure indexing. The supplied partitions were reconstructed for the final dataset. The archived training runs that produced Table S2 used a different dataset and checkpoint-selection rule. Their metrics do not measure this retrained model.
