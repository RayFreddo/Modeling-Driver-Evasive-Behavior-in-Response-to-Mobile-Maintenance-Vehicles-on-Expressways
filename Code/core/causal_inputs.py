"""Prefix-only motion estimates conditional on supplied reconstructed coordinates.

This does NOT make the preceding whole-track reconstruction causal.  Ego speed
is an explicitly declared reconstruction proxy unless an independent ego column
is provided. No acceleration, lateral speed or relative speed column is trusted.
"""
import numpy as np

VERSION = 'prefix-local-polynomial-0.5s-v2'
WINDOW_S = .5

def causal_motion(frame):
    h = frame.copy()
    t = h.timestamp_s.to_numpy(float)
    xy = h[['x_m','y_m']].to_numpy(float)
    if len(t)<3 or not np.isfinite(t).all() or not np.isfinite(xy).all() or np.any(np.diff(t)<=0):
        raise ValueError('Causal motion needs ordered finite positions')
    if 'ego_projection_mps' in h:
        ego = h.ego_projection_mps.to_numpy(float)
        provenance = 'explicit supplied ego projection; independently validate upstream'
    else:
        ego = (h.follower_speed_mps+h.rel_velocity_mps).to_numpy(float)
        provenance = 'offline reconstruction ego proxy; not independently causal-validated'
    if not np.isfinite(ego).all() or (ego<0).any():
        raise ValueError('Invalid ego speed proxy')
    slopes = np.empty_like(xy); accelerations = np.empty_like(xy)
    # The first unavailable derivatives use a zero-order initialization. They
    # are not future-filled. The production t0 has at least three seconds history.
    slopes[0]=0.;accelerations[0]=0.
    for i in range(1,len(t)):
        start=int(np.searchsorted(t,t[i]-WINDOW_S,side='left'))
        start=min(start,i-1)
        u=t[start:i+1]-t[i]
        degree=2 if len(u)>=3 else 1
        design=np.column_stack([u**j for j in range(degree+1)])
        coef=np.linalg.lstsq(design,xy[start:i+1]-xy[i],rcond=None)[0]
        slopes[i]=coef[1]
        accelerations[i]=2*coef[2] if degree==2 else 0.
    # Differentiate the ego proxy with the same prefix policy; no central diff.
    ego_acc=np.r_[0.,np.diff(ego)/np.diff(t)]
    h['rel_velocity_mps']=slopes[:,1]
    h['lat_speed_mps']=slopes[:,0]
    h['follower_speed_mps']=ego-slopes[:,1]
    h['follower_acc_mps2']=ego_acc-accelerations[:,1]
    h['lat_acc_mps2']=accelerations[:,0]
    h['ego_projection_mps']=ego
    h.attrs['motion_input_version']=VERSION
    h.attrs['ego_provenance']=provenance
    return h

def history_at(frame,t0):
    """Select the prefix BEFORE any transformation, rejecting an absent endpoint."""
    if not np.isfinite(t0):raise ValueError('Finite prediction time required')
    h=frame.loc[frame.timestamp_s<=t0+1e-10].copy()
    if len(h)<3 or abs(float(h.timestamp_s.iloc[-1])-t0)>1e-9:
        raise ValueError('Prediction time must be an observed endpoint')
    h=h.loc[h.timestamp_s>=t0-3.-.05].copy()
    return causal_motion(h)
