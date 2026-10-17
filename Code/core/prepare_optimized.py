"""Join detector boxes to full corrected histories; never use raw trajectory positions."""
from pathlib import Path
import json,gc
import numpy as np
import pandas as pd
import joblib
from contracts import ROOT,DAYS,sha,write_json,write_csv,dump_new,npz_new
from full_dataset import read_day,COUNTS
from optimized_features import from_history,BOX_COLUMNS
OUT=ROOT/'outputs/step4_optimized_full4890_dataset_20261014_run2'

def json_frames(path):
    decoder=json.JSONDecoder();buf='';pos=0;eof=False;opened=False
    with path.open('r',encoding='utf-8-sig') as f:
        while True:
            if pos>0:buf=buf[pos:];pos=0
            if not eof:
                chunk=f.read(1024*1024);buf+=chunk;eof=not chunk
            while True:
                while pos<len(buf) and buf[pos] in ' \r\n\t,':pos+=1
                if not opened:
                    if pos==len(buf):break
                    if buf[pos]!='[':raise ValueError('Expected JSON array')
                    opened=True;pos+=1;continue
                if pos<len(buf) and buf[pos]==']':return
                if pos==len(buf):break
                try:value,end=decoder.raw_decode(buf,pos)
                except json.JSONDecodeError:
                    if eof:raise
                    break
                pos=end;yield value
            if eof:raise ValueError('Truncated JSON')

def read_optimized(day):
    folder=OUT/day;proof=json.loads((folder/'identity.json').read_text(encoding='utf-8'))
    for rel,d in proof['input_sha256'].items():
        if sha(ROOT/rel)!=d:raise ValueError('Optimized data input changed: '+rel)
    for name,d in proof['artifact_sha256'].items():
        if sha(folder/name)!=d:raise ValueError('Optimized data cache changed: '+name)
    ss=joblib.load(folder/'dataset.joblib')
    with np.load(folder/'arrays.npz',allow_pickle=False) as f:z={k:f[k].copy() for k in f.files}
    assert len(ss)==len(z['keys'])==COUNTS[day]
    return ss,z,proof

def main():
    OUT.mkdir(exist_ok=False);records=[]
    for day in DAYS:
        folder=OUT/day;folder.mkdir();ss=read_day(day);needed={}
        for s in ss:
            h=s.history
            for frame in h.loc[~h.model_extended,'frame']:
                needed[(int(s.manifest['object_id']),int(frame))]=None
        raw=ROOT/'Data'/f'output_json_{day}';before=sha(raw)
        for item in json_frames(raw):
            for det in item.get('detections',[]):
                key=(int(det['object_id']),int(det.get('frame',item['frame'])))
                if key in needed:
                    assert needed[key] is None
                    # Only these detector fields are read. x_offset/y_offset are excluded.
                    needed[key]=[*map(float,det['bbox']),float(det['class']),float(det['confidence'])]
        assert sha(raw)==before
        arrays=[];targets=[];keys=[];counts=[]
        for s in ss:
            h=s.history.copy();a=np.full((len(h),6),np.nan)
            for j,record in enumerate(h.itertuples(index=False)):
                if not record.model_extended:
                    v=needed.get((int(record.object_id),int(record.frame)))
                    if v is not None:a[j]=v
            for j,col in enumerate(BOX_COLUMNS):h[col]=a[:,j]
            h['confidence']=a[:,5];s.history=h;t0=float(h.timestamp_s.iloc[-1]);x=from_history(h,t0)
            arrays.append(x);targets.append(s.truth_delta+h[['x_m','y_m']].iloc[-1].to_numpy())
            keys.append(s.key);counts.append(int(np.isfinite(a[:,:4]).all(1).sum()))
            s.manifest['observed_box_rows_in_history']=counts[-1]
            s.manifest['box_missing_rows_in_history']=len(h)-counts[-1]
        x=np.array(arrays);y=np.array(targets);assert x.shape==(COUNTS[day],310) and np.isfinite(y).all()
        dump_new(folder/'dataset.joblib',ss);npz_new(folder/'arrays.npz',x=x,y=y,keys=np.array(keys),days=np.repeat(day,len(ss)))
        write_csv(folder/'manifest.csv',pd.DataFrame([s.manifest for s in ss]))
        info={'day':day,'N':len(ss),'all_corrected_objects_included':True,'coordinates_and_targets':'five corrected Excel only',
          'raw_fields_used':['bbox','class','confidence','object_id','frame'],'raw_coordinate_fields_used':False,
          'missing_boxes':'NaN, fitted imputation within the training scope only; no invented detector boxes',
          'objects_with_any_observed_boxes':int(np.sum(np.array(counts)>0)),
          'input_sha256':{str((ROOT/'Data'/f'data_{day}_修正.xlsx').relative_to(ROOT)):sha(ROOT/'Data'/f'data_{day}_修正.xlsx'),str(raw.relative_to(ROOT)):before},
          'artifact_sha256':{name:sha(folder/name) for name in ['dataset.joblib','arrays.npz','manifest.csv']}}
        write_json(folder/'identity.json',info);records.append(info);print('OPTIMIZED FULL DATA',day,len(ss),'boxes available',info['objects_with_any_observed_boxes'],flush=True)
        del ss,needed;gc.collect()
    assert sum(r['N'] for r in records)==4890
    write_json(OUT/'completion.json',{'passed':True,'N':4890,'records':records,'no_model_fitting':True})
if __name__=='__main__':main()
