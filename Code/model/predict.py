"""现行 HGB-ECR：历史输入推理入口。仅同步名称，不改变预测公式或权重。"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Code/revision_step4_matched_teacher_20261014_run13'))
import audit_release as a
from _locations import CURRENT, relocate_loaded_modules
relocate_loaded_modules()
a.DEST = CURRENT
import argparse,json
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from contracts import sha,write_json,write_csv,npz_new

def main(held,out,sample_key=None):
    registry=json.loads((a.DEST/'冻结模型注册表.json').read_text(encoding='utf-8'))
    locks=json.loads((a.DEST/'输入与实现锁.json').read_text(encoding='utf-8'))
    entry_lock=json.loads((a.DEST/'现行入口代码锁.json').read_text(encoding='utf-8'))
    for path,h in entry_lock.items():
        if sha(Path(path))!=h:raise ValueError('Current HGB-ECR entry changed: '+path)
    for path,h in locks['code_sha256'].items():
        if sha(a.r.Path(path))!=h:raise ValueError('Frozen code changed: '+path)
    row=next(x for x in registry['models'] if x['held']==held);asset=locks['inputs'][held]
    for name in ['base','core']:
        assert sha(a.r.Path(row[name+'_path']))==row[name+'_sha256']
    assert sha(a.r.Path(asset['history_path']))==asset['history_sha256']
    assert sha(a.r.Path(asset['feature_path']))==asset['feature_sha256']
    samples=joblib.load(asset['history_path'])
    with np.load(asset['feature_path'],allow_pickle=False) as z:
        assert set(z.files)=={'x','keys'};x=z['x'];keys=z['keys']
    assert keys.tolist()==[s.key for s in samples]
    if sample_key:
        ix=[i for i,s in enumerate(samples) if s.key==sample_key]
        if len(ix)!=1:raise ValueError('Expected one known sample key')
        samples=[samples[ix[0]]];x=x[ix]
    for s in samples:
        assert set(vars(s))=={'key','history'}
        assert s.history.timestamp_s.max()<=float(s.key.split(':t0')[-1])+1e-9
    parent=joblib.load(row['base_path']).base;core=joblib.load(row['core_path'])
    assert set(parent.training_days)==set(row['training_days']) and held not in parent.training_days
    policy=row['policy']
    raw=a.r.h.l.damp(a.r.h.g.generate(parent,samples,x,policy['fallback_method'],0.),policy['lateral_tau_s'],samples)
    generated=a.r.h.corrected(core,raw,x,row['beta'],samples)
    target=a.r.Path(out).resolve();target.mkdir(parents=True,exist_ok=False)
    npz_new(target/'predictions_120s.npz',**{k:v for k,v in generated.items() if isinstance(v,np.ndarray)})
    parts=[]
    for i,s in enumerate(samples):
        t=generated['offsets_s'];p=generated['positions'][i]
        parts.append(pd.DataFrame(dict(sample_key=s.key,offset_s=t,timestamp_s=generated['initial_origin_s'][i]+t,pred_x_m=p[:,0],pred_y_m=p[:,1])))
    write_csv(target/'predictions_120s.csv',pd.concat(parts,ignore_index=True))
    write_json(target/'identity.json',dict(model='HGB-ECR', model_status='现行主模型', implementation_id='matched_teacher_full_horizon_20261014', held=held,N=len(samples),training_days=parent.training_days,
        no_future_label_files_loaded=True,no_scoring_references_loaded=True,
        registry_sha256=sha(a.DEST/'冻结模型注册表.json'),input_lock_sha256=sha(a.DEST/'输入与实现锁.json'),
        artifacts={p.name:sha(p) for p in target.iterdir() if p.is_file()}))
    print('HGB-ECR PAST-ONLY PREDICTION COMPLETE',held,len(samples),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--held',required=True,choices=a.r.h.s.DAYS);p.add_argument('--out',required=True);p.add_argument('--sample-key');args=p.parse_args()
    with threadpool_limits(limits=2):main(args.held,args.out,args.sample_key)
