import numpy as np
import pandas as pd
import pytest

from uncertainty import ensemble_summary,aggregate_predictions


def test_single_model_uncertainty_is_unavailable():
    result = ensemble_summary([[0.7],[0.3]])
    assert result['uncertainty_status']=='single_model_not_available'
    for key in ('ensemble_sd','ensemble_q025','ensemble_q975','ensemble_interval_width'):
        assert np.isnan(result[key]).all()


def test_ensemble_spread():
    values = np.array([[0.1,0.4,0.8],[0.7,0.7,0.7]])
    result = ensemble_summary(values)
    np.testing.assert_allclose(result['ensemble_sd'],values.std(axis=1,ddof=1))
    np.testing.assert_allclose(result['ensemble_interval_width'],np.quantile(values,.975,axis=1)-np.quantile(values,.025,axis=1))


def test_aggregate_rejects_misaligned_entities(tmp_path):
    frame=pd.DataFrame([dict(triplet_id='t',compound_id='c',protein1_id='a',protein2_id='b',inducibility_score_mean=.5,n_models=1)])
    a=tmp_path/'a.csv';b=tmp_path/'b.csv'
    frame.to_csv(a,index=False)
    frame['protein1_id']='wrong'
    frame.to_csv(b,index=False)
    with pytest.raises(ValueError,match='identities differ'):
        aggregate_predictions([a,b],tmp_path/'out.csv')


def test_occlusion_is_signed_and_preserves_inputs():
    torch=pytest.importorskip('torch');dgl=pytest.importorskip('dgl')
    from explain import occlusion_scores
    graph=dgl.graph(([0,1],[1,2]),num_nodes=3)
    graph.ndata['feats']=torch.ones(3,2)
    batch=[None,graph,None,torch.ones(1,5,4),None,None,torch.ones(1,5,4)]
    class Model:
        def __call__(self,*x):
            return (2*x[1].ndata['feats'].sum()+3*x[3].sum()-5*x[6].sum()).reshape(1)
    baseline,atoms,residues=occlusion_scores(Model(),batch,[2,3])
    assert baseline==pytest.approx(-28)
    np.testing.assert_array_equal(atoms,[4,4,4])
    np.testing.assert_array_equal(residues[0],[12,12])
    np.testing.assert_array_equal(residues[1],[-20,-20,-20])
    assert torch.equal(graph.ndata['feats'],torch.ones(3,2))
    assert torch.equal(batch[3],torch.ones(1,5,4))
