#!/usr/bin/env python
"""Atom-graph and residue-sequence occlusion for one prepared triplet."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from schema import audit_triplets, read_csv, sha256_file


def occlusion_scores(model, batch, lengths):
    """Match the atom/residue channel occlusion used in the case analyses."""
    import torch
    with torch.inference_mode():
        def logit(features):
            value = float(model(*features).reshape(-1)[0].cpu())
            if not np.isfinite(value):
                raise ValueError('nonfinite occlusion logit')
            return value
        baseline = logit(batch)
        atoms = []
        graph = batch[1]
        for i in range(graph.num_nodes()):
            perturbed = list(batch)
            copied = graph.clone()
            copied.ndata['feats'] = graph.ndata['feats'].clone()
            copied.ndata['feats'][i] = 0
            perturbed[1] = copied
            atoms.append(baseline-logit(perturbed))
        residues = []
        for index, length in zip((3,6), lengths):
            values = []
            for position in range(length):
                perturbed = list(batch)
                sequence = batch[index].clone()
                sequence[0,position,:] = 0
                perturbed[index] = sequence
                values.append(baseline-logit(perturbed))
            residues.append(np.asarray(values))
    return baseline, np.asarray(atoms), residues


def residue_plot(frame, destination):
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="175" viewBox="0 0 1000 175">',
             '<rect width="1000" height="175" fill="white"/>',
             '<text x="20" y="22" font-family="sans-serif" font-size="15">Residue one-hot occlusion: absolute change in logit</text>']
    maximum = max(float(frame.abs_delta_logit.max()), 1e-12)
    for row_index, side in enumerate(('protein1','protein2')):
        subset = frame.loc[frame.protein.eq(side)]
        y = 58+row_index*65
        parts.append(f'<text x="20" y="{y-8}" font-family="sans-serif">{side}</text>')
        width = 920/len(subset)
        for i,item in enumerate(subset.itertuples()):
            strength = item.abs_delta_logit/maximum
            pale = int(round(255*(1-strength)))
            title = html.escape(f'{side} {item.position_1based} {item.residue}: |delta logit|={item.abs_delta_logit:.6g}')
            parts.append(f'<rect x="{20+i*width:.3f}" y="{y}" width="{width:.3f}" height="22" fill="rgb(255,{pale},{pale})"><title>{title}</title></rect>')
        parts.append(f'<text x="20" y="{y+37}" font-family="sans-serif" font-size="11">1</text>')
        parts.append(f'<text x="925" y="{y+37}" font-family="sans-serif" font-size="11">{len(subset)}</text>')
    parts.append('</svg>')
    destination.write_text('\n'.join(parts),encoding='utf-8')


def explain_prepared(input_csv, feature_root, checkpoint, run_config, output_dir, *,
                     triplet_id=None, device='cpu', molformer_model):
    import torch
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from Dataset import PreparedFeatureDataset
    from data_process import compound_graph
    from inference import _score_one_checkpoint, _move_batch

    frame = read_csv(input_csv)
    if triplet_id is not None:
        frame = frame.loc[frame.triplet_id.eq(triplet_id)]
    if len(frame)!=1:
        raise ValueError('select exactly one row with --triplet-id')
    frame = frame.reset_index(drop=True)
    report = audit_triplets(frame, require_sequences=True, feature_root=feature_root)
    if report['status']!='PASS':
        raise ValueError(str(report['issues'][:5]))
    if 'preparation_status' in frame and not frame.preparation_status.eq('ready').all():
        raise ValueError('complete feature preparation before explanation')
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('use a new explanation output directory')
    root = Path(feature_root).resolve()
    dataset = PreparedFeatureDataset(frame,root)
    sample = dataset[0]
    mol = Chem.MolFromSmiles(frame.smiles.iloc[0])
    expected_graph = compound_graph(frame.smiles.iloc[0])
    actual_graph = sample['compound_graph']
    if mol.GetNumAtoms()!=actual_graph.num_nodes():
        raise ValueError('compound graph nodes do not match SMILES atoms')
    if (not torch.equal(actual_graph.ndata['feats'].cpu(),expected_graph.ndata['feats'].cpu())
        or any(not torch.equal(a.cpu(),b.cpu()) for a,b in zip(actual_graph.edges(),expected_graph.edges()))):
        raise ValueError('compound graph atom order/schema differs from the input SMILES')
    lengths = [len(frame[f'{side}_sequence'].iloc[0]) for side in ('protein1','protein2')]
    cache = {}
    _, metadata = _score_one_checkpoint(frame,root,Path(checkpoint).resolve(),Path(run_config).resolve(),
        batch_size=1,device_name=device,temperature=1.0,cache_size=8,
        molformer_model=molformer_model,runtime_cache=cache)
    model = next(iter(cache.values()))[0]
    batch = _move_batch(PreparedFeatureDataset.collate([sample]),next(model.parameters()).device)
    baseline, atom_delta, residue_delta = occlusion_scores(model,batch,lengths)
    atoms = pd.DataFrame(dict(atom_index_0based=np.arange(len(atom_delta)),
        element=[a.GetSymbol() for a in mol.GetAtoms()],delta_logit=atom_delta,
        abs_delta_logit=np.abs(atom_delta)))
    atoms['rank_by_abs_delta'] = atoms.abs_delta_logit.rank(method='min',ascending=False).astype(int)
    residues = []
    for side, values in zip(('protein1','protein2'),residue_delta):
        seq = frame[f'{side}_sequence'].iloc[0]
        part = pd.DataFrame(dict(protein=side,protein_id=frame[f'{side}_id'].iloc[0],
            position_1based=np.arange(1,len(seq)+1),residue=list(seq),delta_logit=values,abs_delta_logit=np.abs(values)))
        part['rank_within_protein'] = part.abs_delta_logit.rank(method='min',ascending=False).astype(int)
        residues.append(part)
    residues = pd.concat(residues,ignore_index=True)
    destination.mkdir(parents=True,exist_ok=True)
    atoms.to_csv(destination/'atom_importance.csv',index=False)
    residues.to_csv(destination/'residue_importance.csv',index=False)
    top = atoms.loc[atoms.abs_delta_logit>0].nlargest(6,'abs_delta_logit').atom_index_0based.tolist()
    drawer = rdMolDraw2D.MolDraw2DSVG(700,450)
    drawer.drawOptions().addAtomIndices = True
    rdMolDraw2D.PrepareAndDrawMolecule(drawer,mol,highlightAtoms=top,
        highlightAtomColors={int(i):(0.95,0.35,0.35) for i in top})
    drawer.FinishDrawing()
    (destination/'atom_top6.svg').write_text(drawer.GetDrawingText())
    residue_plot(residues,destination/'residue_importance.svg')
    manifest = dict(triplet_id=frame.triplet_id.iloc[0],baseline_logit=baseline,
        baseline_score=float(torch.sigmoid(torch.tensor(baseline,dtype=torch.float64))),
        atom_method='zero one compound graph-node feature vector; keep bonds, SMILES and Uni-Mol2 fixed',
        residue_method='zero one residue one-hot vector; keep protein graph and ESM-2 fixed',
        delta_definition='baseline_logit - occluded_logit',importance='abs(delta_logit)',
        scoring_context='one triplet, supplied protein order, one checkpoint, unscaled logit',
        structure_contacts='not inferred; experimental contact mapping requires a resolved ternary structure',
        input_sha256=sha256_file(Path(input_csv)),model=metadata)
    (destination/'explanation.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input','feature-root','checkpoint','run-config','output-dir','molformer-model'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--triplet-id')
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='cpu')
    args = p.parse_args(argv)
    result = explain_prepared(args.input,args.feature_root,args.checkpoint,args.run_config,args.output_dir,
        triplet_id=args.triplet_id,device=args.device,molformer_model=args.molformer_model)
    print(json.dumps(result,indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
