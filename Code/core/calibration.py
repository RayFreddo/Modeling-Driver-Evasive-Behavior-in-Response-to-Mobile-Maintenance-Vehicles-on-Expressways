"""Vectorized training-only physical calibration, with new Step3 prior initializations."""
import numpy as np
from scipy.optimize import least_squares
from upstream2.physical_models import PhysicalModel
from upstream2.kinematics import interpolate

def rows(samples,tau=0.,return_weights=False):
    allrows=[];owners=[]
    for s in samples:
        h=s.history;t=h.timestamp_s.to_numpy(float)
        states=h[['follower_speed_mps','rel_velocity_mps','y_m','follower_acc_mps2']].to_numpy(float,copy=True)
        states[:,1]+=states[:,0]
        idx=np.arange(0,len(h),6);idx=idx[t[idx]-tau>=t[0]+.5-1e-9]
        if not len(idx):continue
        delayed=interpolate(t,states,t[idx]-tau)
        r=np.column_stack([delayed[:,:3],states[idx,3]])
        ok=np.isfinite(r).all(1)&(r[:,:2]>=0).all(1)&(r[:,2]>0)
        if ok.any():allrows.append(r[ok]);owners.append(s.day)
    if not allrows or not sum(map(len,allrows)):raise ValueError('No valid physical calibration rows')
    data=np.vstack(allrows)
    if not return_weights:return data
    dates=sorted(set(owners))
    weights=np.concatenate([np.full(len(r),np.sqrt(len(data)/(len(dates)*owners.count(day)*len(r)))) for day,r in zip(owners,allrows)])
    return data,weights

def fit_acceleration(kind,samples,prior,tau=0.):
    data,weights=rows(samples,tau,True);v,l,g,target=data.T
    q=prior['model_training_feature_quantiles']['absolute_speed_mps']['q05_q25_q50_q75_q95']
    desired=float(np.clip(q[-1],5,45)) if q else 20.
    if kind=='Helly':
        names=['ks','kv','s0','T'];bounds=([0,0,0,.1],[2,3,30,4])
        starts=[[.02,.18,2.,1.5],[.02,.18,min(20.,np.quantile(g,.05)),1.]]
        def calculate(p):ks,kv,s0,T=p;return kv*(l-v)+ks*(g-s0-T*v)
    elif kind=='IDM':
        names=['v0','T','s0','a','b'];bounds=([1,.1,.1,.05,.1],[50,4,30,5,10])
        starts=[[desired,1.5,2.,1.4,3.],[20.,1.,1.,1.,2.]]
        def calculate(p):
            v0,T,s0,a,b=p;sg=s0+v*T+v*(v-l)/(2*np.sqrt(a*b))
            return a*(1-(v/v0)**4-(sg/g)**2)
    else:raise ValueError(kind)
    trials=[];fits=[]
    for init in starts:
        fit=least_squares(lambda p:(calculate(p)-target)*weights,init,bounds=bounds,loss='soft_l1',max_nfev=600)
        trials.append({'initial':init,'success':bool(fit.success),'message':fit.message,'nfev':fit.nfev,'cost':fit.cost,'params':fit.x})
        if fit.success:fits.append(fit)
    if not fits:raise RuntimeError(kind+' calibration failed')
    fit=min(fits,key=lambda f:f.cost);params=dict(zip(names,map(float,fit.x)))
    errors=calculate(fit.x)-target
    report={'kind':kind,'training_days':sorted({s.day for s in samples}),'sample_keys':[s.key for s in samples],
      'rows':len(data),'row_stride':6,'loss':'soft_l1','bounds':dict(zip(names,zip(*bounds))),
      'weighting':'equal dates, then equal eligible vehicles within date, then equal retained rows within vehicle',
      'prefix_initialization_excluded_s':.5,
      'at_bound':dict(zip(names,np.isclose(fit.x,bounds[0],atol=1e-4)|np.isclose(fit.x,bounds[1],atol=1e-4))),
      'trials':trials,'params':params,'training_acceleration_rmse_mps2':float(np.sqrt(np.mean(errors**2))),
      'prior_fingerprint':prior['fingerprint'],'tau_s':tau,'point_spacing_proxy':True,
      'equation':'Helly instantaneous/delayed paper Eq36; no acceleration term' if kind=='Helly' else 'Treiber2000 IDM, delta=4, s1=0'}
    return PhysicalModel(kind,params,reaction_time_s=tau),report

def fit_physical(kind,samples,prior,tau=0.,method='acceleration'):
    from physical_training import fit_calibrated
    return fit_calibrated(kind,samples,prior,tau,method)
