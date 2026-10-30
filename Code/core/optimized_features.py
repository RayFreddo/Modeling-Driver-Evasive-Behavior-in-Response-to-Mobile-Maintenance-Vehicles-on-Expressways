"""Past-only motion/image features plus TTC, DRAC and adjacent-lane gaps.

The appended input schema requires refitting; legacy 310-column weights and
cached arrays cannot be used as if they had learned these additional inputs.
"""
import numpy as np
from optimized_features_base import features,transform as _motion_transform,anchor_basis,HORIZONS
BOX_COLUMNS=['bbox_left','bbox_top','bbox_right','bbox_bottom','detector_class']
RISK_FEATURE_NAMES=('ttc_s','drac_mps2',
    'left_front_gap_m','left_rear_gap_m','left_available_gap_m',
    'right_front_gap_m','right_rear_gap_m','right_available_gap_m')
FEATURE_DIM=310+len(RISK_FEATURE_NAMES)
FEATURE_SCHEMA='motion310-risk8-v1'

def risk_features(past,t0):
    """Use only the supplied prefix and its time-matched neighbor snapshot.

    y is positive rearward, so closing speed is minus relative y velocity.
    Gaps are longitudinal reference-point distances, not bumper clearances.
    Both adjacent lanes are retained; no future maneuver-side label is used.
    """
    row=past.iloc[-1];spacing=float(row.y_m)
    if 'rel_velocity_mps' in past:
        closing=-float(row.rel_velocity_mps)
    else:
        recent=past.loc[past.timestamp_s>=t0-.5-1e-9]
        closing=-float(np.polyfit(recent.timestamp_s.to_numpy(float)-t0,
                                 recent.y_m.to_numpy(float),1)[0])
    valid=np.isfinite(spacing) and np.isfinite(closing) and spacing>0 and closing>.5
    values=[spacing/closing if valid else np.nan,
            closing**2/(2*spacing) if valid else np.nan]
    snapshot=past.attrs.get('neighbor_snapshot')
    for side in ('left','right'):
        gaps=[]
        for role in ('front','rear'):
            column=f'{side}_{role}_gap_m'
            if snapshot is not None:
                from upstream3.neighbors import observe_slot
                ego=float(row.ego_projection_mps)
                observation=observe_slot(snapshot,f'{side}_{role}',float(row.x_m),
                    spacing,ego,t0,.1,row.get('object_id'))
                gap=abs(observation.dy_m) if observation.observed else np.nan
            else:
                gap=float(row.get(column,np.nan))
            gaps.append(gap if np.isfinite(gap) and gap>0 else np.nan)
        # Missing detections stay missing; they are not zero or an infinite gap.
        values.extend([*gaps,sum(gaps) if np.isfinite(gaps).all() else np.nan])
    return np.asarray(values,float)

def transform(x):
    """Pass the appended risk inputs into the same fitted model transformer."""
    x=np.asarray(x,float)
    if x.ndim!=2 or x.shape[1]!=FEATURE_DIM:
        raise ValueError('Expected motion310-risk8-v1 inputs; rebuild features and refit the model')
    return np.column_stack([_motion_transform(x[:,:310]),x[:,310:]])

def box_features(t,boxes,classes,t0):
    t=np.asarray(t,float);keep=(t<=t0+1e-10)&(t>=t0-3.10000001)
    t=t[keep];b=np.asarray(boxes,float)[keep];classes=np.asarray(classes)[keep]
    if len(t)<3 or abs(t[-1]-t0)>1e-8 or t[-1]-t[0]<2.98-1e-8 or np.any(np.diff(t)<=0):raise ValueError('Unsupported history')
    wh=b[:,2:]-b[:,:2];valid=np.isfinite(b).all(1)&(wh>0).all(1)
    z=np.full((len(t),7),np.nan)
    z[valid]=np.column_stack([(b[valid,0]+b[valid,2])/2,(b[valid,1]+b[valid,3])/2,wh[valid],b[valid,3],1/wh[valid,0],1/wh[valid,1]])
    u=t-t0;values=list(z[-1])
    for edge in np.arange(-3.,-.01,.25):
        m=(u>=edge-1e-9)&(u<=edge+.25+1e-9)&valid
        values.extend(np.median(z[m],axis=0) if m.any() else np.full(7,np.nan))
    for span in (1.,2.,3.):
        m=(u>=-span-1e-9)&valid;a=z[m];uu=u[m]
        if len(uu)>=3:
            slope=np.linalg.lstsq(np.column_stack([np.ones(len(uu)),uu]),a,rcond=None)[0][1]
            values.extend(np.median(a,axis=0));values.extend(slope);values.extend(np.quantile(a,.75,axis=0)-np.quantile(a,.25,axis=0))
        else:values.extend(np.full(21,np.nan))
    values.extend([float(classes[-1]==code) for code in (2,5,7)]);values.append(float(classes[-1] not in (2,5,7)))
    return np.array(values,float)

def from_history(h,t0):
    past=h.loc[h.timestamp_s<=t0+1e-10].copy()
    a=features(past.timestamp_s.to_numpy(),past[['x_m','y_m']].to_numpy(),past.ego_projection_mps.to_numpy(),past.confidence.to_numpy(),t0)
    b=box_features(past.timestamp_s.to_numpy(),past[BOX_COLUMNS[:4]].to_numpy(),past.detector_class.to_numpy(),t0)
    x=np.r_[a,b,risk_features(past,t0)]
    if x.shape!=(FEATURE_DIM,) or np.isinf(x).any():raise ValueError('Invalid optimized features')
    return x
