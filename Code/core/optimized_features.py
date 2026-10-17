"""Original 310-feature optimized schema; unknown box observations remain missing."""
import numpy as np
from optimized_features_base import features,transform,anchor_basis,HORIZONS
BOX_COLUMNS=['bbox_left','bbox_top','bbox_right','bbox_bottom','detector_class']

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
    x=np.r_[a,b]
    if x.shape!=(310,) or np.isinf(x).any():raise ValueError('Invalid optimized features')
    return x
