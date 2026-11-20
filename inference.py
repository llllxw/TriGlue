"""Load matching trained checkpoints and score fixed prepared features."""
from pathlib import Path
import gc,hashlib,json
import numpy as np
import pandas as pd
from schema import audit_triplets,read_csv,sha256_file
from uncertainty import ensemble_summary


def settings_digest():
    settings=json.loads(Path(__file__).with_name('config.json').read_text(encoding='utf-8'))
    return hashlib.sha256(json.dumps(settings,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def source_hashes():
    return {name:hashlib.sha256(Path(__file__).with_name(name).read_text(encoding='utf-8').encode()).hexdigest()
            for name in ['model.py','Dataset.py']}


def _resolve_device(torch,requested):
    if requested=='auto':requested='cuda' if torch.cuda.is_available() else 'cpu'
    if requested=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
    return torch.device(requested)


def freeze_molformer_random_features(model):
    changed=0
    for module in model.molf.modules():
        if hasattr(module,'orthogonal_random_weights') and hasattr(module,'deterministic'):
            module.deterministic=True;changed+=1
    if not changed:raise RuntimeError('Use the pinned MoLFormer snapshot from the resource archive')
    return changed


def _load_config(path):
    from model import MODEL_ID
    config=json.loads(Path(path).read_text(encoding='utf-8'))
    if config.get('model_id')!=MODEL_ID:raise ValueError('Checkpoint requires training with this manuscript reference model')
    if config.get('settings_sha256')!=settings_digest():raise ValueError('Training settings differ from config.json')
    if config.get('model_source_sha256')!=source_hashes():raise ValueError('Model implementation differs from training')
    if config.get('temperature_fit_partition')!='validation':raise ValueError('Validation-fitted temperature is missing')
    return config


def _move_batch(batch,device):
    return (batch[0],)+tuple(value.to(device) for value in batch[1:])


def predict_prepared(input_csv,feature_root,checkpoints,run_configs,output_csv,*,
                     batch_size=16,device='cpu',temperatures=None,molformer_model=None):
    import torch
    from torch.utils.data import DataLoader
    from Dataset import PreparedFeatureDataset
    from model import TriComplexClassifier,SETTINGS
    if batch_size<1 or not checkpoints or len(checkpoints)!=len(run_configs):
        raise ValueError('Supply a positive batch size and one run config per checkpoint')
    if not molformer_model or not Path(molformer_model).is_dir():raise FileNotFoundError('Local MoLFormer snapshot required')
    hashes=[sha256_file(Path(p)) for p in checkpoints]
    if len(set(hashes))!=len(hashes):raise ValueError('Repeated weights cannot estimate ensemble uncertainty')
    configs=[_load_config(Path(p)) for p in run_configs]
    for digest,config in zip(hashes,configs):
        if digest!=config.get('checkpoint_sha256'):raise ValueError('Checkpoint and run configuration do not match')
    if temperatures is None:temperatures=[c['temperature'] for c in configs]
    if len(temperatures)!=len(checkpoints) or any(not np.isfinite(t) or t<=0 for t in temperatures):
        raise ValueError('Supply one finite positive temperature per checkpoint')
    frame=read_csv(input_csv)
    audit=audit_triplets(frame,feature_root=feature_root,require_sequences=True)
    if audit['status']!='PASS':raise ValueError(str(audit['issues'][:5]))
    dataset=PreparedFeatureDataset(frame,Path(feature_root))
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,collate_fn=dataset.collate,num_workers=0)
    target=_resolve_device(torch,device)
    scores=[];raw=[]
    TriComplexClassifier.MOLFORMER=str(Path(molformer_model).resolve())
    for checkpoint,temperature in zip(checkpoints,temperatures):
        model=TriComplexClassifier(dropout=SETTINGS['dropout'],enable_smiles_cache=True)
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        model.to(target).eval();freeze_molformer_random_features(model)
        logits=[]
        with torch.inference_mode():
            for batch in loader:logits.extend(model(*_move_batch(batch,target)).cpu().tolist())
        values=torch.tensor(logits,dtype=torch.float64)
        if not torch.isfinite(values).all():raise ValueError('Model returned nonfinite logits')
        scores.append(torch.sigmoid(values/temperature).numpy());raw.append(torch.sigmoid(values).numpy())
        del model;gc.collect()
    output=frame.copy()
    values=np.stack(scores,axis=1)
    for name,value in ensemble_summary(values).items():output[name]=value
    output['raw_score_mean']=np.mean(raw,axis=0)
    for i,value in enumerate(scores):output['score_model_'+str(i+1)]=value
    destination=Path(output_csv);destination.parent.mkdir(parents=True,exist_ok=True)
    output.to_csv(destination,index=False,encoding='utf-8')
    manifest={'status':'PASS','input_sha256':sha256_file(Path(input_csv)),
              'n_triplets':len(frame),'n_models':len(checkpoints),'output':str(destination.resolve()),
              'checkpoint_sha256':hashes,'seeds':[c['seed'] for c in configs],'temperatures':temperatures}
    destination.with_suffix(destination.suffix+'.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return output,manifest
