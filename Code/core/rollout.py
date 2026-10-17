"""Three-second inputs, successive predicted segments, separate endpoint scoring."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from upstream2.state_update import freeze_ego,update_states,EgoState
from upstream2.kinematics import interpolate

HORIZONS=np.arange(.5,3.01,.5)
BOX_COLUMNS=['bbox_left','bbox_top','bbox_right','bbox_bottom','confidence']

def _history(h,end):
    # Only the preceding 3s enter the next block. The left endpoint is continuity.
    q=h[(h.timestamp_s<=end+1e-10)&(h.timestamp_s>=end-3.-1e-10)].copy()
    q.attrs.update(h.attrs)
    if len(q)<3 or q.timestamp_s.iloc[-1]-q.timestamp_s.iloc[0]<2.98-1e-8:
        raise ValueError('Insufficient nominal three-second history; no fabricated history')
    return q

def generate(history,model,*,generation_duration_s,lane_centers=None):
    """Generation limit is predeclared from training policy, not the true test endpoint.

    No full trajectory, future observations or reference terminal time are accepted.
    Every later input uses the model's own coordinates and consistent derived states.
    Missing future image detections remain missing; never read test image futures.
    """
    if not np.isfinite(generation_duration_s) or generation_duration_s<=0:
        raise ValueError('Positive generation duration required')
    if any(c in history for c in ['event_end_s','lane_completion_confirm_s','truth_delta']):
        raise ValueError('Scoring labels cannot enter predictor input')
    t0=float(history.timestamp_s.iloc[-1]);current=_history(history,t0)
    # Match optimized physical initialization: the available t0 ego projection.
    ego=(EgoState(t0,float(current.ego_projection_mps.iloc[-1]),'observed_t0_ego_projection')
         if 'ego_projection_mps' in current else freeze_ego(current))
    paths=[];logs=[];elapsed=0.;block=0
    origin=current[['x_m','y_m']].iloc[-1].to_numpy(float)
    while elapsed<generation_duration_s-1e-10:
        block+=1;tcur=float(current.timestamp_s.iloc[-1]);current.attrs['ego_state']=ego
        prediction=np.asarray(model.predict_history(current,tcur),float)
        if prediction.shape!=(6,2) or not np.isfinite(prediction).all():
            raise ValueError('Model does not produce a complete finite segment; no fallback')
        duration=min(3.,generation_duration_s-elapsed)
        dense=np.arange(.05,duration-1e-10,.05)
        fixed=np.r_[HORIZONS[HORIZONS<duration-1e-10],duration]
        dense=dense[np.min(np.abs(dense[:,None]-fixed[None,:]),axis=1)>1e-10]
        offsets=np.sort(np.r_[dense,fixed])
        xy=interpolate(np.r_[0.,HORIZONS],np.vstack([current[['x_m','y_m']].iloc[-1].to_numpy(),prediction]),offsets)
        new=update_states(current,tcur+offsets,xy,ego=ego,lane_centers=lane_centers)
        new['ego_projection_mps']=ego.longitudinal_speed_mps
        for c in BOX_COLUMNS:new[c]=np.nan
        # Detector class is immutable tracked identity, not a synthetic image observation.
        new['detector_class']=float(current.detector_class.iloc[-1]) if 'detector_class' in current else np.nan
        new['model_extended']=False;new['is_prediction']=True;new['source_supported']=False
        new['block_index']=block;new['forecast_origin_s']=t0
        logs.append(dict(block=block,input_first_s=float(current.timestamp_s.iloc[0]),input_last_s=tcur,
            output_last_s=float(new.timestamp_s.iloc[-1]),observations_added=False,
            input_predicted_fraction=float(current.get('is_prediction',pd.Series(False,index=current.index)).fillna(False).mean())))
        paths.append(new)
        joined=pd.concat([current,new],ignore_index=True);joined.attrs.update(current.attrs)
        elapsed+=duration;current=_history(joined,t0+elapsed)
    path=pd.concat(paths,ignore_index=True)
    path.attrs.update(forecast_origin_s=t0,origin_xy=origin.tolist(),generation_duration_s=generation_duration_s)
    return path,logs

def score_to_lane_completion(path,reference,*,t0,lane_completion_s):
    """Reference completion exists only in this scorer; include the exact last point."""
    if not np.isfinite(lane_completion_s) or lane_completion_s<=t0:
        raise ValueError('No confirmed lane completion after input; cannot invent FDE')
    duration=lane_completion_s-t0
    offsets=np.r_[np.arange(.5,duration-1e-10,.5),duration]
    times=np.r_[t0,path.timestamp_s.to_numpy(float)]
    positions=np.vstack([path.attrs['origin_xy'],path[['x_m','y_m']].to_numpy(float)])
    pred=interpolate(times,positions,t0+offsets)
    truth=interpolate(reference.timestamp_s.to_numpy(float),reference[['x_m','y_m']].to_numpy(float),t0+offsets)
    err=np.linalg.norm(pred-truth,axis=1)
    points=pd.DataFrame(dict(timestamp_s=t0+offsets,offset_s=offsets,pred_x_m=pred[:,0],pred_y_m=pred[:,1],
        truth_x_m=truth[:,0],truth_y_m=truth[:,1],error_m=err))
    return points,dict(ADE_m=float(err.mean()),FDE_m=float(err[-1]),MSE_m2=float(np.mean(err**2)),
        RMSE_m=float(np.sqrt(np.mean(err**2))),completion_s=float(lane_completion_s),duration_s=float(duration),K=len(err))
