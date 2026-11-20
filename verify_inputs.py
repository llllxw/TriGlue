"""Check the pinned final dataset and reconstructed parent-grouped five-fold partitions."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))

import pandas as pd
from schema import _canonical_smiles


def verify(dataset, splits, provenance):
    record = json.loads(Path(provenance).read_text(encoding='utf-8'))
    expected = record['dataset_sha256']
    if hashlib.sha256(Path(dataset).read_bytes()).hexdigest() != expected:
        raise ValueError('Dataset differs from the pinned final Supplementary Table 1')
    frame = pd.read_csv(dataset, dtype=str, keep_default_na=False)
    if frame.sample_id.duplicated().any() or not frame.label.isin(['0', '1']).all():
        raise ValueError('Require unique sample_id and binary labels')
    ids = set(frame.sample_id)
    root=Path(provenance).parent
    prepared=pd.read_csv(root/'data/prepared_triplets.csv',dtype=str,keep_default_na=False)
    groups=pd.read_csv(root/'data/groups.csv',dtype=str,keep_default_na=False)
    links=pd.read_csv(root/'data/parent_links.csv',dtype=str,keep_default_na=False)
    sequences=pd.read_csv(root/'data/sequences.csv',dtype=str,keep_default_na=False)
    if set(prepared.triplet_id)!=ids or prepared.triplet_id.duplicated().any():
        raise ValueError('Prepared features do not cover the final dataset exactly once')
    if set(groups.sample_id)!=ids or groups.sample_id.duplicated().any():
        raise ValueError('Parent-group mapping does not cover the final dataset')
    truth=frame.set_index('sample_id')
    expected_groups=groups.set_index('sample_id').parent_group_id.to_dict()
    sequence_map=sequences.set_index('protein_id').sequence.to_dict()
    for row in sequences.itertuples():
        if hashlib.sha256(row.sequence.encode()).hexdigest()!=row.sequence_sha256:
            raise ValueError('Protein sequence digest mismatch')
    exact={}
    for row in prepared.itertuples():
        original=truth.loc[row.triplet_id]
        if row.label!=original.label or _canonical_smiles(row.smiles)[0]!=_canonical_smiles(original.SMILES)[0] or row.parent_group_id!=expected_groups[row.triplet_id]:
            raise ValueError('Prepared row differs from the final dataset or parent groups')
        if row.protein1_sequence!=sequence_map[row.protein1_id] or row.protein2_sequence!=sequence_map[row.protein2_id]:
            raise ValueError('Protein feature ID does not match its sequence')
        key=(_canonical_smiles(row.smiles)[0],*sorted((row.protein1_sequence,row.protein2_sequence)))
        if key in exact and exact[key]!=row.parent_group_id:
            raise ValueError('Exact sequence triplet occurs in different parent groups')
        exact[key]=row.parent_group_id
    if any(expected_groups[r.sample_id]!=expected_groups[r.parent_sample_id] for r in links.itertuples()):
        raise ValueError('Derived negative and contributing positive parent have different groups')
    tests = []
    group_members = {}
    for fold in range(1, 6):
        filename = Path(splits) / ('fold' + str(fold) + '.csv')
        if not filename.is_file():
            raise FileNotFoundError('Pinned final-data partition required: ' + str(filename))
        table = pd.read_csv(filename, dtype=str, keep_default_na=False)
        required = {'sample_id', 'partition', 'parent_group_id'}
        if not required.issubset(table.columns):
            raise ValueError(str(filename) + ': require ' + str(sorted(required)))
        if table.sample_id.duplicated().any() or set(table.sample_id) != ids:
            raise ValueError(str(filename) + ': partition IDs must cover the final dataset exactly once')
        if set(table.partition) != {'train', 'val', 'test'} or table.parent_group_id.eq('').any():
            raise ValueError(str(filename) + ': require three partitions and nonempty parent groups')
        if table.groupby('parent_group_id').partition.nunique().gt(1).any():
            raise ValueError(str(filename) + ': a parent group crosses partitions')
        if any(expected_groups[r.sample_id]!=r.parent_group_id for r in table.itertuples()):
            raise ValueError('Fold parent groups differ from the fixed parent-group mapping')
        for row in table.itertuples():
            if row.sample_id in group_members and group_members[row.sample_id] != row.parent_group_id:
                raise ValueError('Parent-group identity changes across folds')
            group_members[row.sample_id] = row.parent_group_id
        tests.extend(table.loc[table.partition.eq('test'), 'sample_id'])
    if len(tests) != len(ids) or set(tests) != ids:
        raise ValueError('Each sample must appear in exactly one outer test fold')
    for relative, digest in record.get('file_sha256', {}).items():
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned data/split artifact differs: ' + relative)
    return {'status': 'PASS', 'n_samples': len(frame), 'n_folds': 5,
            'scope': 'file identity, parent-group membership and fold coverage; requires authentic supplied parent links'}


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default=str(root / 'data' / 'dataset.csv'))
    parser.add_argument('--splits', default=str(root / 'splits'))
    args = parser.parse_args()
    print(json.dumps(verify(args.dataset, args.splits, root / 'provenance.json'), indent=2))
