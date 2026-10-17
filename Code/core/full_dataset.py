"""All 4890 corrected Excel objects; processed extensions included as instructed."""
from dataclasses import dataclass
import json,gc
import numpy as np
import pandas as pd
import joblib
from contracts import *
from upstream3.io_utils import load_day
from upstream3.features import history_features,FEATURE_NAMES
from upstream3.scenes import NeighborIndex
from upstream3.config import Rules
from upstream2.kinematics import interpolate
from causal_inputs import causal_motion,VERSION as MOTION_VERSION
CACHE=ROOT/'outputs/step4_full4890_dataset_20261014_run1'
COUNTS={'0403':418,'0407':1725,'0409':1202,'0410':1229,'0416':316}
ACTIVE_SAMPLES=None
ACTIVE_OUT=None

@dataclass
class Sample:
    key:str
    day:str
    history:pd.DataFrame
    truth_delta:np.ndarray
    manifest:dict
    features:np.ndarray
    snapshot:object
    action:str

def build_day(day):
    dest=CACHE/day;dest.mkdir(parents=True,exist_ok=False)
    tables,info=load_day(ROOT,day,CURRENT_STEP3/'parsed_cache',allow_create=False)
    main=tables['main'];neighbors=NeighborIndex(tables['neighbors']);samples=[]
    for obj,object_frame in main.groupby('object_id',sort=True):
        choice=None
        for seg,g in object_frame.groupby('segment_id',sort=True):
            g=g.sort_values('timestamp_s').reset_index(drop=True)
            valid=np.isfinite(g[['timestamp_s','x_m','y_m','follower_speed_mps','rel_velocity_mps']]).all(axis=1)
            g=g.loc[valid].reset_index(drop=True);t=g.timestamp_s.to_numpy(float)
            if not len(t):continue
            cuts=np.r_[0,np.flatnonzero(np.diff(t)>.10000001)+1,len(t)]
            for run,(left,right) in enumerate(zip(cuts[:-1],cuts[1:])):
                part=g.iloc[left:right].reset_index(drop=True);tt=part.timestamp_s.to_numpy(float)
                j=int(np.searchsorted(tt,tt[0]+3.+1e-9,side='right')-1)
                if j>=2 and tt[j]-tt[0]>=2.98-1e-9 and tt[-1]>=tt[j]+3.-1e-9:
                    choice=(seg,run,part,j);break
            if choice is not None:break
        if choice is None:raise ValueError(f'No nominal 3s/precise future 3s window: {day}:{obj}')
        seg,run,g,j=choice;t=g.timestamp_s.to_numpy(float);t0=float(t[j]);h=causal_motion(g.iloc[:j+1].copy())
        snap,_,_=neighbors.snapshot(obj,t0,h.frame.iloc[-1],Rules(),float(t[0]))
        fdict=history_features(h,snap,Rules());f=np.array([fdict[c] for c in FEATURE_NAMES],float)
        y=interpolate(t,g[['x_m','y_m']].to_numpy(float),t0+HORIZONS)-h[['x_m','y_m']].iloc[-1].to_numpy(float)
        right=np.minimum(np.searchsorted(t,t0+HORIZONS),len(t)-1);left=np.maximum(right-1,0)
        assert right.max()<len(t) and np.all(t[right]-t[left]<=.10000001)
        extended=g.model_extended.to_numpy(bool)
        future=g[(g.timestamp_s>=t0)&(g.timestamp_s<=t0+3.+1e-9)]
        acc=future.follower_acc_mps2.to_numpy(float);ft=future.timestamp_s.to_numpy(float);dec=False;begin=None
        for k,flag in enumerate(np.isfinite(acc)&(acc<=-1.2)):
            if flag:
                if begin is None:begin=k
                if ft[k]-ft[begin]>=4/60.-1e-9:dec=True
            else:begin=None
        lateral=abs(float(y[-1,0]))>=.8
        action='coupled' if dec and lateral else ('deceleration_only' if dec else ('lateral_only' if lateral else 'no_threshold_action'))
        key=f'{day}:{int(obj)}:corrected_full4890:t0{t0:.9f}'
        row={**fdict,'sample_key':key,'day':day,'object_id':int(obj),'segment_id':int(seg),'run_id':run,'t0_s':t0,'max_input_time_s':t0,
             'actual_history_s':float(t0-t[0]),'nominal_history_s':3.,'clock_quantization_shortfall_s':float(max(0.,3.-(t0-t[0]))),
             'history_extended_fraction':float(h.model_extended.mean()),'target_bracket_extended_fraction':float((extended[left]|extended[right]).mean()),
             'action_type':action,'target_delta_x_3s_m':float(y[-1,0]),'target_delta_y_3s_m':float(y[-1,1]),
             'input_source':'complete corrected Excel including processed extension','reference':'corrected reconstructed coordinates'}
        samples.append(Sample(key,day,h,y,row,f,snap,action))
    assert len(samples)==COUNTS[day] and len({s.key for s in samples})==len(samples)
    dump_new(dest/'dataset.joblib',samples);write_csv(dest/'manifest.csv',pd.DataFrame([s.manifest for s in samples]))
    write_json(dest/'dataset_audit.json',{'samples':len(samples),'input_files':[info],'input_sha256':info['sha256'],
      'dataset_sha256':sha(dest/'dataset.joblib'),'manifest_sha256':sha(dest/'manifest.csv'),'source_code_sha256':sources(),
      'all_objects_included':True,'source_supported_only':False,'maximum_clock_quantization_shortfall_s':max(s.manifest['clock_quantization_shortfall_s'] for s in samples),
      'measurement_validated':False,'prediction_scope':'full processed reconstruction; exact 3s future, nominal 3s past with <=0.02s clock quantization'})
    print('FULL CORRECTED PREPARED',day,len(samples),flush=True)
    del main,tables,neighbors;gc.collect()

def read_day(day,include_samples=True):
    dest=CACHE/day;info=json.loads((dest/'dataset_audit.json').read_text(encoding='utf-8'))
    if sha(ROOT/'Data'/f'data_{day}_修正.xlsx')!=info['input_sha256']:raise ValueError('Corrected Excel fingerprint differs')
    for name in ['dataset','manifest']:
        ext='joblib' if name=='dataset' else 'csv'
        if sha(dest/(name+'.'+ext))!=info[name+'_sha256']:raise ValueError('Full-cohort cache changed')
    return joblib.load(dest/'dataset.joblib') if include_samples else pd.read_csv(dest/'manifest.csv',dtype={'day':str},float_precision='round_trip')

def validate_analysis(held):
    c=json.loads((FOLD_ROOT/f'holdout_{held}'/'02_模型训练/训练接入协议.json').read_text(encoding='utf-8'))
    assert set(c['train_days'])==set(DAYS)-{held} and c['outer_validation_day']==held
    for day,h in c['analysis_input_sha256'].items():
        if sha(ROOT/'Data'/f'data_{day}_修正.xlsx')!=h:raise ValueError('Input differs from user fold protocol')
    frame=pd.concat([read_day(d,False) for d in DAYS if d!=held],ignore_index=True)
    assert len(frame)==4890-COUNTS[held] and not frame.sample_key.duplicated().any()
    return frame,c,{'full4890':True}

def save_scope(samples,out):
    dump_new(out/'dataset.joblib',samples);write_csv(out/'manifest.csv',pd.DataFrame([s.manifest for s in samples]))
    infos=[json.loads((CACHE/d/'dataset_audit.json').read_text(encoding='utf-8')) for d in sorted({s.day for s in samples})]
    write_json(out/'dataset_audit.json',{'samples':len(samples),'counts_by_day':{d:sum(s.day==d for s in samples) for d in sorted({s.day for s in samples})},
      'input_files':[i['input_files'][0] for i in infos],'all_objects_included':True,'source_supported_only':False,'measurement_validated':False,
      'prediction_scope':'prediction error relative to complete corrected trajectories including processed extension'})
    write_json(out/'causality_boundary.json',{'past_only_derivatives':True,'whole_track_reconstructed_coordinates_causal_validated':False,
      'processing_extension_included_per_user':True,'strict_raw_online_no_leakage_proven':False,'reference':'corrected Excel'})

def prepare_training(held,out):
    global ACTIVE_SAMPLES,ACTIVE_OUT
    validate_analysis(held);samples=[s for d in DAYS if d!=held for s in read_day(d)]
    assert len(samples)==4890-COUNTS[held]
    ACTIVE_SAMPLES=samples;ACTIVE_OUT=out;save_scope(samples,out)
    write_json(out/'current_training_contract.json',{'train_days':[d for d in DAYS if d!=held],'outer_validation_day':held,'all_objects_included':True,
      'training_n':len(samples),'validation_expected_n':COUNTS[held],'total':4890,'reference':'complete corrected Excel'})
    return samples

def prepare_validation(held,out,frozen):
    assert frozen['status']=='frozen_before_outer_input' and frozen['held']==held
    for name,h in frozen['model_sha256'].items():
        if sha(Path(frozen['train_output'])/name)!=h:raise ValueError('Frozen weights changed')
    samples=read_day(held);assert len(samples)==COUNTS[held];save_scope(samples,out);return samples

def load_full_prior(train_days,forbidden):
    import contracts
    train_days=set(train_days);forbidden=set(forbidden)
    if ACTIVE_SAMPLES is None or contracts.ACTIVE_HELD not in forbidden or train_days&forbidden:raise ValueError('Invalid prior scope')
    ss=[s for s in ACTIVE_SAMPLES if s.day in train_days];assert {s.day for s in ss}==train_days
    quantiles={}
    for j,c in enumerate(FEATURE_NAMES):
        a=np.array([s.features[j] for s in ss]);a=a[np.isfinite(a)]
        quantiles[c]={'q05_q25_q50_q75_q95':np.quantile(a,[.05,.25,.5,.75,.95]).tolist() if len(a) else None}
    prototypes={}
    for action in sorted({s.action for s in ss}):
        y=np.array([s.truth_delta for s in ss if s.action==action])
        prototypes[action]={'valid_n':len(y),'median_delta_xy_m':np.median(y,axis=0).tolist(),'time_from_prediction_t0_s':HORIZONS.tolist()}
    v={'train_days':sorted(train_days),'excluded_days':sorted(forbidden),'model_training_sample_keys':[s.key for s in ss],
       'model_training_feature_quantiles':quantiles,'training_label_prototypes':prototypes,'version':'complete_corrected_4890_v1',
       'training_input_hashes':{d:sha(ROOT/'Data'/f'data_{d}_修正.xlsx') for d in sorted(train_days)}}
    v['fingerprint']=digest(v);folder=ACTIVE_OUT/'full_cohort_priors';folder.mkdir(exist_ok=True);path=folder/('_'.join(sorted(train_days))+'.json')
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==clean(v)
    else:write_json(path,v)
    return v,{'path':str(path),'sha256':sha(path),'fingerprint':v['fingerprint'],'train_days':sorted(train_days),'excluded_days':sorted(forbidden)}

def input_arrays(samples,span=3.):
    rows=[]
    for s in samples:
        h=s.history;t=h.timestamp_s.to_numpy(float);t0=t[-1];xy=h[['x_m','y_m']].to_numpy(float)
        actual=min(float(span),float(t0-t[0]));seq=interpolate(t,xy,t0+np.linspace(-actual,0,21))-xy[-1]
        f=s.features.copy();mask=t>=t0-actual-1e-9
        for axis,k in [('x',0),('y',1)]:
            f[FEATURE_NAMES.index('history_'+axis+'_slope_mps')]=np.polyfit(t[mask]-t0,xy[mask,k],1)[0]
            for lag in [.5,1.,2.,3.]:f[FEATURE_NAMES.index(f'{axis}_delta_{lag:g}s_m')]=xy[-1,k]-np.interp(t0-min(lag,actual),t,xy[:,k])
        rows.append(np.r_[f,seq.ravel()])
    x=np.array(rows)
    if np.isinf(x).any():raise ValueError('Infinite predictor')
    return x

def feature_schema():
    return {'predictor_columns':FEATURE_NAMES+[f'centered_{axis}_{i}' for i in range(21) for axis in ['x','y']],
      'upstream_predictors':FEATURE_NAMES,'dimensions':80,'history_candidates_s':[1.,2.,3.],'sequence_points':21,
      'history_lag_definition':'min(lag, selected span, actual observed duration)','sequence_time_definition':'linspace(-min(selected_span,actual_duration),0,21)',
      'targets':[f'delta_{axis}_{h:g}s_m' for h in HORIZONS for axis in ['x','y']],
      'forbidden':['date','action_type','future labels','event_end','final_lane_changed'],
      'corrector_meta':'X80, anchor12, base12, residual12, specialist2, specialist-minus-base2',
      'task':'complete corrected 4890 objects, six fixed future horizons; nominal 3s history with <=0.02s clock quantization',
      'generalization':'internal date generalization relative to corrected reconstruction including extensions','motion_input_version':MOTION_VERSION,
      'strict_raw_online_no_leakage_proven':False,'measurement_validated':False}
