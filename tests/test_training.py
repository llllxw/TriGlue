from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from train import TrainingBatches, parser, validate_split
from screen import enumerate_screen, enumerate_screen_stream


def test_training_defaults_and_batches():
    args = parser().parse_args(['--train','t.csv','--val','v.csv','--feature-root','f',
        '--output-dir','o','--molformer-model','m'])
    assert (args.lr,args.batch_size,args.epochs,args.patience)==(1e-4,16,50,20)
    for n in range(2,40):
        for batch_size in (2,3,16):
            sampler = TrainingBatches(n,batch_size,2025)
            batches = list(sampler)
            assert len(batches)==len(sampler)
            assert all(len(b)>=2 for b in batches)
            assert sorted(i for b in batches for i in b)==list(range(n))


def test_split_detects_reversed_duplicate():
    train = pd.DataFrame([dict(triplet_id='a',smiles='CCO',protein1_id='p1',protein2_id='p2')])
    val = pd.DataFrame([dict(triplet_id='b',smiles='CCO',protein1_id='p2',protein2_id='p1')])
    with pytest.raises(ValueError,match='same compound/protein'):
        validate_split(train,val)


@pytest.mark.parametrize('enumerate_fn',[enumerate_screen,enumerate_screen_stream])
def test_enumeration_preserves_structure_paths(tmp_path,enumerate_fn):
    source = tmp_path/'inputs'
    source.mkdir()
    pdb = source/'p.pdb'
    pdb.write_text('path fixture')
    pd.DataFrame([dict(compound_id='c',smiles='CCO')]).to_csv(source/'compounds.csv',index=False)
    pd.DataFrame([dict(pair_id='p',protein1_id='p1',protein2_id='p2',protein1_sequence='AC',
        protein2_sequence='AD',protein1_structure='p.pdb',protein2_structure='p.pdb')]).to_csv(source/'pairs.csv',index=False)
    target = tmp_path/'output/screen.csv'
    enumerate_fn(source/'compounds.csv',source/'pairs.csv',target)
    assert pd.read_csv(target).protein1_structure.iloc[0]==str(pdb)


def test_labelled_features_do_not_mutate_cache(tmp_path):
    torch=pytest.importorskip('torch')
    dgl=pytest.importorskip('dgl')
    from Dataset import LabelledFeatureDataset
    for folder in ['compound_graph','protein_graph','compound_3d_embedding',
                   'protein_embedding/one_hot','protein_3d_embedding']:
        (tmp_path/folder).mkdir(parents=True)
    for folder,name,dim in [('compound_graph','c',44),('protein_graph','p',41)]:
        graph=dgl.graph(([0,1],[1,0]),num_nodes=2)
        graph.ndata['feats']=torch.ones(2,dim)
        dgl.save_graphs(str(tmp_path/folder/f'{name}.bin'),[graph])
    np.save(tmp_path/'compound_3d_embedding/c.npy',np.ones((1,768),dtype=np.float32))
    np.save(tmp_path/'protein_embedding/one_hot/p.npy',np.ones((1200,25),dtype=np.float32))
    np.save(tmp_path/'protein_3d_embedding/p.npy',np.ones((2,1280),dtype=np.float32))
    frame=pd.DataFrame([dict(smiles='CCO',compound_id='c',protein1_id='p',protein2_id='p',label=1)])
    train=LabelledFeatureDataset(frame,tmp_path,training=True)
    val=LabelledFeatureDataset(frame,tmp_path,training=False)
    assert not np.array_equal(train[0]['compound_embedding'],val[0]['compound_embedding'])
    assert np.array_equal(val[0]['compound_embedding'],np.ones((1,768)))
    assert train[0]['compound_graph'].num_nodes()==2
