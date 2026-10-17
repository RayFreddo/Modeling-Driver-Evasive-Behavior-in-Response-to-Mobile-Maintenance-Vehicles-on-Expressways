import numpy as np
def observations(h):
    t=h.timestamp_s.to_numpy(float);height=(h.bbox_bottom-h.bbox_top).to_numpy(float)
    xy=h[['x_m','y_m']].to_numpy(float)
    if len(t)<3 or not np.isfinite(np.column_stack([t,xy,height])).all() or np.any(np.diff(t)<=0) or np.any(height<=0) or np.any(xy[:,1]<=0):raise ValueError('Unsupported geometric history')
    # Apparent-height inverse-range assumption. Calibration is past-only per object.
    k=float(np.median(xy[:,1]*height))
    return t,np.column_stack([xy[:,0],1./height]),k
