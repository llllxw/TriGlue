import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from calibration import analyze, apply_temperature, reliability


def files(tmp_path, prefix, labelled):
    paths=[]
    for model in range(2):
        frame=pd.DataFrame(dict(triplet_id=[f'{prefix}{i}' for i in range(4)],
            compound_id=[f'{prefix}_c{i}' for i in range(4)],protein1_id='p1',protein2_id='p2',
            inducibility_score_mean=np.array([.1,.8,.7,.9])*(1-.1*model),n_models=1))
        if labelled:frame['label']=[0,1,0,1]
        path=tmp_path/f'{prefix}_model{model}.csv';frame.to_csv(path,index=False);paths.append(path)
    return paths


def test_labelled_outputs_and_validation_only_fit(tmp_path):
    validation=files(tmp_path,'val',True);test=files(tmp_path,'test',True)
    analyze(validation,test,tmp_path/'labelled')
    output=tmp_path/'labelled'
    assert len(list(output.glob('*.svg')))==4
    for p in output.glob('*.svg'):ET.parse(p)
    result=pd.read_csv(output/'calibrated_predictions.csv')
    params=json.loads((output/'temperatures.json').read_text())
    raw=np.column_stack([pd.read_csv(p).inducibility_score_mean for p in test])
    expected=np.column_stack([apply_temperature(raw[:,i],t) for i,t in enumerate(params['model_temperatures'])])
    np.testing.assert_allclose(result.ensemble_sd,expected.std(axis=1,ddof=1))
    np.testing.assert_allclose(result.calibrated_confidence,np.maximum(result.ensemble_mean_calibrated,1-result.ensemble_mean_calibrated))
    for p in test:
        f=pd.read_csv(p);f=f.drop(columns='label');f.to_csv(p,index=False)
    analyze(validation,test,tmp_path/'unlabelled')
    second=json.loads((tmp_path/'unlabelled/temperatures.json').read_text())
    assert params['model_temperatures']==second['model_temperatures']
    assert params['ensemble_temperature']==second['ensemble_temperature']
    assert not list((tmp_path/'unlabelled').glob('*.svg'))
    assert not (tmp_path/'unlabelled/calibration_metrics.csv').exists()


def test_reliability_uses_predicted_class_confidence():
    bins,metrics=reliability(np.array([0,1]),np.array([.1,.9]))
    assert bins.n.sum()==2
    assert metrics['ECE']==pytest.approx(.1)
    assert metrics['Brier']==pytest.approx(.01)
    assert metrics['NLL']==pytest.approx(-np.log(.9))


def test_rejects_validation_prediction_overlap(tmp_path):
    paths=files(tmp_path,'val',True)
    with pytest.raises(ValueError,match='must be separate'):
        analyze(paths,paths,tmp_path/'output')
