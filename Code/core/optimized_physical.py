"""Optimized four-state/time-weighted Kalman and physical initialization adapters."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
from weighted_core import weighted_states
H=np.arange(.5,3.01,.5)

def observation(h,span=2.):
    t0=float(h.timestamp_s.iloc[-1]);h=h.loc[h.timestamp_s>=t0-span-1e-9].copy()
    width=(h.bbox_right-h.bbox_left).to_numpy(float);height=(h.bbox_bottom-h.bbox_top).to_numpy(float)
    size=np.sqrt(np.maximum(width*height,0.));ok=np.isfinite(size)&(size>0)&(h.y_m.to_numpy()>0)
    if ok[-1] and ok.sum()>=3:
        g=h.loc[ok];t=g.timestamp_s.to_numpy(float)
        if t[-1]-t[0]>=.5 and np.max(np.diff(t))<=.10000001:
            k=float(np.median(g.y_m.to_numpy()*size[ok]));return t,np.column_stack([g.x_m.to_numpy(),1/size[ok]]),k,True
    # Missing image measurements are not synthesized. Same state estimator uses
    # the actual corrected-position observation channel, explicitly flagged.
    return h.timestamp_s.to_numpy(),h[['x_m','y_m']].to_numpy(),1.,False

def slope_state(h,geometry_weight=1.,span=2.,tau=.5):
    t0=float(h.timestamp_s.iloc[-1]);t,z,k,geometric=observation(h,span)
    w=np.exp((t-t0)/tau);w/=w.sum();u=t-np.sum(t*w);den=np.sum(w*u*u)
    if den<=0:raise ValueError('Unsupported weighted slope')
    estimate=float(k*np.sum(w*u*(z[:,1]-np.sum(w*z[:,1])))/den)
    past=h.loc[h.timestamp_s>=t0-span-1e-9];pt=past.timestamp_s.to_numpy(float);pw=np.exp((pt-t0)/tau);pw/=pw.sum();pu=pt-np.sum(pt*pw)
    point=float(np.sum(pw*pu*(past.y_m.to_numpy()-np.sum(pw*past.y_m.to_numpy())))/np.sum(pw*pu*pu))
    estimate=geometry_weight*estimate+(1-geometry_weight)*point
    changed=h.copy();ego=float(h.ego_projection_mps.iloc[-1]);idx=changed.index[-1]
    changed.loc[idx,'rel_velocity_mps']=estimate;changed.loc[idx,'follower_speed_mps']=ego-estimate
    return changed,geometric

class OptimizedPhysical:
    def __init__(self,core,days,geometry_weight=1.,lateral_span=2.):
        self.core=core;self.training_days=list(days);self.geometry_weight=geometry_weight;self.lateral_span=lateral_span
    def predict_at(self,h,t0):
        h=h.loc[h.timestamp_s<=t0+1e-10].copy().sort_values('timestamp_s')
        if abs(float(h.timestamp_s.iloc[-1])-t0)>1e-9:raise ValueError('Observed endpoint required')
        changed,_=slope_state(h,self.geometry_weight);p=self.core.predict(changed)
        g=h.loc[h.timestamp_s>=t0-self.lateral_span-1e-9];v=np.polyfit(g.timestamp_s.to_numpy()-t0,g.x_m.to_numpy(),1)[0]
        p[:,0]=H*v;return p+h[['x_m','y_m']].iloc[-1].to_numpy()
    def predict_history(self,h,t0):return self.predict_at(h,t0)

class OptimizedKalman:
    def __init__(self,days,scale,q,tau=.5,geometry_weight=1.):
        self.training_days=list(days);self.scale={k:np.asarray(v) for k,v in scale.items()};self.q=float(q);self.tau=tau;self.geometry_weight=geometry_weight
    def predict_histories(self,histories):
        normalized=[];klist=[];origin=[];last=[];point_hist=[];scales=[]
        for h in histories:
            t,z,k,geometric=observation(h);scale=self.scale['geometry' if geometric else 'point'];zz=z/scale;scales.append(scale)
            normalized.append(pd.DataFrame(dict(timestamp_s=t,x_m=zz[:,0],y_m=zz[:,1])));klist.append(k)
            origin.append(h[['x_m','y_m']].iloc[-1].to_numpy());last.append(zz[-1])
            t0=float(h.timestamp_s.iloc[-1]);g=h.loc[h.timestamp_s>=t0-2.-1e-9]
            gh=g[['timestamp_s','x_m','y_m']].copy();gh[['x_m','y_m']]=gh[['x_m','y_m']]/self.scale['point'];point_hist.append(gh)
        p=(np.array(last)[:,None]+weighted_states(normalized,{'q':self.q,'r':1.},self.tau))*np.array(scales)[:,None,:]
        p[:,:,1]*=np.array(klist)[:,None]
        velocity=(p[:,1]-p[:,0])/.5
        if self.geometry_weight<1:
            direct=weighted_states(point_hist,{'q':self.q,'r':1.},self.tau)*self.scale['point'];dv=(direct[:,1]-direct[:,0])/.5
            velocity=self.geometry_weight*velocity+(1-self.geometry_weight)*dv
        return np.array(origin)[:,None]+H[None,:,None]*velocity[:,None]
    def predict_history(self,h,t0):
        past=h.loc[h.timestamp_s<=t0+1e-10].sort_values('timestamp_s')
        if abs(float(past.timestamp_s.iloc[-1])-t0)>1e-9:raise ValueError('Observed endpoint required')
        return self.predict_histories([past])[0]

def estimate_scale(samples):
    byday={'geometry':[],'point':[]}
    for day in sorted({s.day for s in samples}):
        residual={'geometry':[],'point':[]}
        for s in samples:
            if s.day!=day:continue
            t,z,_,geometric=observation(s.history)
            observations=[('geometry' if geometric else 'point',t,z)]
            h=s.history;t0=float(h.timestamp_s.iloc[-1]);g=h.loc[h.timestamp_s>=t0-2.-1e-9]
            if geometric:observations.append(('point',g.timestamp_s.to_numpy(),g[['x_m','y_m']].to_numpy()))
            for kind,t,z in observations:
                l=np.diff(t)[:-1];r=np.diff(t)[1:];w=r/(l+r)
                residual[kind].append((z[1:-1]-w[:,None]*z[:-2]-(1-w)[:,None]*z[2:])/np.sqrt(1+w*w+(1-w)**2)[:,None])
        for kind in residual:
            if residual[kind]:
                e=np.concatenate(residual[kind]);byday[kind].append(np.median(np.abs(e-np.median(e,axis=0)),axis=0)/.6744897501960817)
    return {kind:np.maximum(np.sqrt(np.mean(np.array(rows)**2,axis=0)) if rows else [0.,0.],[.001,1e-7] if kind=='geometry' else [.001,.001]) for kind,rows in byday.items()}
