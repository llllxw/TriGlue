"""Frozen Uni-Mol2 features from molecule coordinates, with training-only perturbations."""
from functools import lru_cache
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem


@lru_cache(maxsize=256)
def conformer_features(smiles):
    from unimol_tools.data.conformer import mol2unimolv2

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid molecular SMILES: " + smiles)
    molecule = Chem.AddHs(molecule)
    # Fixed ETKDG initialization; the fallback changes only the initial placement.
    for random_start in (False, True):
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useRandomCoords = random_start
        if AllChem.EmbedMolecule(molecule, params) == 0:
            break
    else:
        raise ValueError("Could not build a 3D conformer for " + smiles)
    if AllChem.MMFFHasAllMoleculeParams(molecule):
        AllChem.MMFFOptimizeMolecule(molecule)
    conf = molecule.GetConformer()
    if not conf.Is3D() or not np.isfinite(conf.GetPositions()).all():
        raise ValueError("Invalid conformer for " + smiles)
    # Preserve bond orders and stereochemistry. Do not crop atoms at random.
    return mol2unimolv2(molecule, max_atoms=molecule.GetNumAtoms(), remove_hs=True)


class CoordinateEncoder:
    def __init__(self, directory, device):
        directory = Path(directory).resolve()
        checkpoint = directory / "modelzoo/84M/checkpoint.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(str(checkpoint))
        os.environ["UNIMOL_WEIGHT_DIR"] = str(directory)
        from unimol_tools.models.unimolv2 import UniMolV2Model
        self.encoder = UniMolV2Model(model_size="84m").requires_grad_(False).eval().to(device)
        if Path(self.encoder.pretrain_path).resolve() != checkpoint:
            raise ValueError("Uni-Mol2 loaded a different encoder directory")
        digest = hashlib.sha256()
        with checkpoint.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        self.checkpoint_sha256 = digest.hexdigest()
        self.device = device
        self.clean_cache = {}

    @torch.no_grad()
    def encode(self, smiles, sigma=0.0):
        if sigma < 0:
            raise ValueError("Coordinate noise sigma must be nonnegative")
        # Microbatching keeps frozen inference within a desktop GPU's memory budget.
        pending = list(smiles) if sigma else list(dict.fromkeys(s for s in smiles if s not in self.clean_cache))
        result = []
        for start in range(0, len(pending), 2):
            chunk = pending[start:start + 2]
            batch, _ = self.encoder.batch_collate_fn([(conformer_features(s), 0) for s in chunk])
            batch = {name: value.to(self.device) for name, value in batch.items()}
            if sigma:
                positions = batch["src_coord"]
                batch["src_coord"] = positions + torch.randn_like(positions) * sigma * batch["atom_mask"].unsqueeze(-1)
            output = self.encoder(**batch, return_repr=True)["cls_repr"].detach()
            if output.shape != (len(chunk), 768) or not torch.isfinite(output).all():
                raise ValueError("Invalid Uni-Mol2 representation")
            if sigma:
                result.extend(output.unbind())
            else:
                self.clean_cache.update(zip(chunk, output.cpu().unbind()))
        if not sigma:
            result = [self.clean_cache[s].to(self.device) for s in smiles]
        return torch.stack(result).unsqueeze(1)
