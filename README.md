# TriGlue

## Introduction
TriGlue is a tri-partite multimodal deep learning framework for predicting protein–protein interaction outcomes mediated by a small molecule (molecular glue). The model takes a ternary input (**compound**, **protein 1**, and **protein 2**) and integrates complementary **1D/2D/3D** representations for each entity.

## Environment Requirement
The code has been tested running under **Python 3.8.20**. The required package are as follows (adjust versions to match your environment):

- **pytorch**
- **numpy**
- **pandas**
- **scikit-learn**
- **dgl** 
- **transformers** 
- **rdkit** 

## Source codes
- **Dataset.py**: dataset definition and `collate_fn` for batching ternary samples (compound, protein1, protein2) with 1D/2D/3D inputs.
- **model.py**: TriGlue main model including modality encoders, intra-modal fusion, tri-modal fusion, and hybrid prediction head; supports ablation switches and optional contrastive loss.
- **train.py**: fixed-split cross-validation training script; logs per-epoch metrics, saves best model per fold, and exports best validation predictions.
- **sequence_model.py**: protein 1D encoder implementation.
- **gragh_model.py**: graph encoders for compound/protein 2D representations.
- **multimodal_fusion.py**: fusion modules including IntraFusion and TriModalFusion used for intra-modal refinement and tri-modal interaction.
