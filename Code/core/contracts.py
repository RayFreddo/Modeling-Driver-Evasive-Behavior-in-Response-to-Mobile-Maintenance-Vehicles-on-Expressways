"""Exclusive artifacts and CURRENT per-fold training contracts; no old prior fallback."""
from pathlib import Path
import hashlib,json,platform,importlib.metadata
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DAYS=('0403','0407','0409','0410','0416')
SEEDS=tuple(range(20261012,20261017))
HORIZONS=np.arange(.5,3.01,.5)
FOLD_ROOT=ROOT/'五折独立实验_20261012'
CURRENT_STEP3=ROOT/'Code/revision_step3_20261012'
ACTIVE_HELD=None

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,set)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,(np.integer,np.bool_)):return v.item()
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    if isinstance(v,Path):return str(v.resolve())
    return v
def digest(v):return hashlib.sha256(json.dumps(clean(v),sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def write_json(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(clean(v),f,ensure_ascii=False,allow_nan=False,indent=2)
def write_csv(p,v):
    with Path(p).open('x',encoding='utf-8-sig',newline='') as f:v.to_csv(f,index=False,float_format='%.17g')
def dump_new(p,v):
    import joblib
    with Path(p).open('xb') as f:joblib.dump(v,f,compress=3)
def npz_new(p,**v):
    with Path(p).open('xb') as f:np.savez_compressed(f,**v)
def versions():
    import torch
    return {'python':platform.python_version(),**{p:importlib.metadata.version(p) for p in
      ['numpy','pandas','scipy','scikit-learn','torch','joblib','openpyxl','threadpoolctl']},
      'cuda_available':torch.cuda.is_available(),'cuda_runtime':torch.version.cuda,
      'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
def sources():return {str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*.py') if '__pycache__' not in p.parts}
def check_originals():
    old=json.loads((HERE/'originals_before.json').read_text(encoding='utf-8'))
    changed=[name for name,h in old.items() if not (ROOT/name).is_file() or sha(ROOT/name)!=h]
    return {'passed':not changed,'protected_files':len(old),'changed_or_missing':changed}
def set_fold(held):
    global ACTIVE_HELD
    if held not in DAYS:raise ValueError('Non-whitelist held day')
    ACTIVE_HELD=held
def analysis_path(held):return FOLD_ROOT/f'holdout_{held}'/'01_特征分析/复核材料'
def load_prior(train_days,forbidden):
    from full_dataset import load_full_prior
    return load_full_prior(train_days,forbidden)
