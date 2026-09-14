# Input data

Use UTF-8 CSV files. Required raw columns are `smiles`, `protein1_sequence` and `protein2_sequence`. Training and validation also require unique `triplet_id` values and binary `label` values (0 or 1); include both classes in each set. Keep the sets disjoint.

Optional `protein1_structure` and `protein2_structure` columns contain complete matching monomer PDB paths relative to the CSV. Optional chain columns are `protein1_chain` and `protein2_chain`.

Run `data_process.py` to create prepared tables and features. Training labels are retained. Prediction does not require labels.
