import numpy as np
HORIZONS=np.arange(.5,3.01,.5)
def features(t,xy,ego,confidence,t0):
    t=np.asarray(t,float);xy=np.asarray(xy,float)
    keep=(t<=t0+1e-10)&(t>=t0-3.10000001)
    t=t[keep];xy=xy[keep];ego=np.asarray(ego)[keep];confidence=np.asarray(confidence)[keep]
    if len(t)<3 or abs(t[-1]-t0)>1e-8 or t[-1]-t[0]<2.98-1e-8:
        raise ValueError('Missing observed t0 or complete 3s history')
    if not np.isfinite(t).all() or not np.isfinite(xy).all() or np.any(np.diff(t)<=0) or np.max(np.diff(t))>.10000001:
        raise ValueError('Nonfinite/duplicate/unsupported history')
    u=t-t0
    med=[];spread=[]
    for edge in np.arange(-3.,-.001,.1):
        m=(u>=edge-1e-9)&(u<edge+.1+1e-9)
        if not m.any():raise ValueError('Empty historical bin')
        a=xy[m];med.append(np.median(a,axis=0));spread.append(np.std(a,axis=0))
    med=np.array(med);center=np.median(xy[u>=-.3],axis=0)
    result=[*center,*(xy[-1]-center),*np.asarray(med).ravel(),*np.asarray(spread).ravel()]
    for span in (.5,1.,2.,3.):
        m=u>=-span-1e-9;a=xy[m];z=u[m]
        c=np.linalg.lstsq(np.column_stack([np.ones(len(z)),z]),a,rcond=None)[0]
        result.extend(c.ravel());result.extend(np.quantile(a,.9,axis=0)-np.quantile(a,.1,axis=0))
    for a in (ego,confidence):
        finite=a[np.isfinite(a)]
        result.extend([a[-1],np.median(finite) if len(finite) else np.nan])
    return np.asarray(result,float)
def coarse_transform(x):
    if x.shape[1]!=310:raise ValueError('Expected existing past-only 310 feature schema')
    origin=x[:,:2]+x[:,2:4]
    positions=np.median(x[:,4:64].reshape(-1,6,5,2),axis=2)
    slopes=np.stack([x[:,132:134],x[:,138:140],x[:,144:146]],axis=1)
    boxes=np.median(x[:,159:243].reshape(-1,6,2,7),axis=2)[:,:,:5]
    summaries=np.column_stack([x[:,264:269],x[:,271:276],x[:,285:290],x[:,292:297]])
    inverse=np.column_stack([1/np.maximum(x[:,266:268],1e-6),1/np.maximum(x[:,287:289],1e-6)])
    return np.column_stack([x[:,:4],positions.reshape(len(x),-1),(positions-origin[:,None,:]).reshape(len(x),-1),
      slopes.reshape(len(x),-1),np.diff(slopes,axis=1).reshape(len(x),-1),x[:,146:152],
      boxes.reshape(len(x),-1),summaries,inverse,x[:,299:304],x[:,306:310]])
def transform(x):
    # OLS coefficients over 0.5s and 1s are past-only averages, unlike frame-to-frame derivatives.
    extra=np.column_stack([x[:,126:128],x[:,126:128]-x[:,132:134],x[:,243:248],x[:,250:255]])
    return np.column_stack([coarse_transform(x),extra])
def anchor_basis(x):
    # First four features encode recent median then raw-origin-minus-median.
    origin=x[:,:2]+x[:,2:4];velocity=x[:,144:146]
    a=origin[:,None,:]+velocity[:,None,:]*HORIZONS[None,:,None]
    norm=np.linalg.norm(velocity,axis=1);u=velocity/np.maximum(norm,1e-12)[:,None];u[norm<1e-6]=[0.,-1.]
    basis=np.stack([u,np.column_stack([-u[:,1],u[:,0]])],axis=2)
    return a,basis
