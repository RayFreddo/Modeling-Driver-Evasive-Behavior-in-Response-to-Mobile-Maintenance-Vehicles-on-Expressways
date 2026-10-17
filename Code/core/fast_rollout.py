"""Aligned batched rolling histories, with the original optimized model equations."""
import numpy as np
from fast_history_features import batch_features
from physical_training import batch_predict

H=np.arange(.5,3.01,.5)
DENSE=np.arange(61,dtype=float)/20.

def predicted_completion(samples,paths,offsets):
    """Output completion diagnostics from predictions only; no reference labels.

    Numerical lane grid follows the corrected-data construction (3.75m, +2).
    A timeout or model failure never becomes an alleged successful lane change.
    """
    initial_lane=np.array([float(s.history.lane_index.iloc[-1]) for s in samples])
    initial_x=paths[:,0,0];dx=np.diff(paths[:,:,0],axis=1)/np.diff(offsets)[None,:]
    lane=np.floor(paths[:,:,0]/3.75+.5)+2
    finite=np.isfinite(paths).all(2)
    stable=(np.abs(dx)<=.1)&finite[:,1:]&finite[:,:-1]
    changed=(lane[:,1:]!=initial_lane[:,None])&(lane[:,1:]==lane[:,:-1])
    retained=(np.abs(paths[:,1:,0]-initial_x[:,None])>=.8)&(np.abs(paths[:,:-1,0]-initial_x[:,None])>=.8)
    qualified=stable&changed&retained
    completion=np.full(len(paths),np.nan)
    for i in range(len(paths)):
        # The rate is over the preceding 0.5s interval, so one complete stable
        # interval is the predeclared .5s confirmation, not merely one point.
        ix=np.flatnonzero(qualified[i])
        if len(ix):completion[i]=offsets[ix[0]+1]
    return completion

def dense_segment(origin,prediction):
    whole=np.concatenate([origin[:,None,:],prediction],axis=1)
    k=np.minimum(np.arange(61)//10,5);w=np.arange(61)/10-k
    return whole[:,k,:]*(1-w)[None,:,None]+whole[:,k+1,:]*w[None,:,None]

def kalman_velocity(times,xy,scale,q,tau):
    """Exact weighted_core equations, vectorizing only its input assembly."""
    z=xy/scale[None,None,:];n=len(z);dt=times[:,1]-times[:,0]
    state=np.concatenate([z[:,1],(z[:,1]-z[:,0])/dt[:,None]],axis=1)
    r0=np.exp((times[:,-1]-times[:,0])/tau);r1=np.exp((times[:,-1]-times[:,1])/tau)
    cov=np.zeros((n,4,4));identity=np.eye(4)
    for pos,vel in [(0,2),(1,3)]:
        cov[:,pos,pos]=r1;cov[:,pos,vel]=cov[:,vel,pos]=r1/dt
        cov[:,vel,vel]=(r0+r1)/dt**2
    for j in range(2,times.shape[1]):
        dt=np.maximum(times[:,j]-times[:,j-1],1e-4)
        f=np.tile(identity,(n,1,1));f[:,0,2]=dt;f[:,1,3]=dt
        process=np.zeros_like(f)
        for pos,vel in [(0,2),(1,3)]:
            process[:,pos,pos]=q*dt**4/4
            process[:,pos,vel]=process[:,vel,pos]=q*dt**3/2
            process[:,vel,vel]=q*dt**2
        p=f@cov@f.transpose(0,2,1)+process;s=(f@state[:,:,None])[:,:,0]
        noise=np.exp((times[:,-1]-times[:,j])/tau)[:,None,None]*np.eye(2)[None]
        gain=np.linalg.solve(p[:,:2,:2]+noise,p[:,:2,:]).transpose(0,2,1)
        s+=(gain@(z[:,j]-s[:,:2])[:,:,None])[:,:,0]
        a=np.tile(identity,(n,1,1));a[:,:,:2]-=gain
        cov=a@p@a.transpose(0,2,1)+gain@noise@gain.transpose(0,2,1);state=s
    return state[:,2:]*scale[None,:]

def point_prediction(model,times,xy,ego,preserve_prefix=False):
    """Post-first-block missing camera channel; same explicit position fallback."""
    name=type(model).__name__
    if name=='OptimizedKalman':
        # Dense grid last 2s. All predicted camera observations are missing.
        velocity=kalman_velocity(times[:,-41:],xy[:,-41:],model.scale['point'],model.q,model.tau)
        return xy[:,-1,None]+H[None,:,None]*velocity[:,None]
    if name=='OptimizedPhysical':
        t=times[:,-41:];y=xy[:,-41:,1];w=np.exp((t-t[:,-1,None])/.5);w/=w.sum(1)[:,None]
        u=t-(t*w).sum(1)[:,None]
        vy=np.sum(w*u*(y-(y*w).sum(1)[:,None]),axis=1)/np.sum(w*u*u,axis=1)
        span=model.lateral_span;k=int(round(span/.05))+1
        lt=times[:,-k:];lx=xy[:,-k:,0];lu=lt-lt.mean(1)[:,None]
        vx=np.sum(lu*(lx-lx.mean(1)[:,None]),axis=1)/np.sum(lu*lu,axis=1)
        speed=ego-vy
        if preserve_prefix:speed=np.where((speed<0)&(speed>=-1e-8),0.,speed)
        initial=np.column_stack([speed,ego,xy[:,-1,1],vx])
        ok=np.isfinite(initial).all(1)&(initial[:,:2]>=0).all(1)&(initial[:,2]>0)
        delta=batch_predict(model.core.kind,model.core.params,(initial,None,ok),model.core.reaction_time_s,preserve_prefix=preserve_prefix)
        return xy[:,-1,None]+delta
    raise ValueError('Unknown physical array adapter '+name)

def generate_batch(model,samples,initial_predictions,*,blocks=40,physical_prefix=False):
    """Fixed predeclared 120s generation, independent of all scoring endpoints.

    Returned paths include model failures and completion metadata separately; the
    scorer must never silently replace a missing endpoint with an earlier point.
    """
    if not isinstance(blocks,int) or blocks<1:raise ValueError('Positive integer block count required')
    n=len(samples);origin=np.array([s.history[['x_m','y_m']].iloc[-1].to_numpy(float) for s in samples])
    t0=np.array([float(s.history.timestamp_s.iloc[-1]) for s in samples])
    ego=np.array([float(s.history.ego_projection_mps.iloc[-1]) for s in samples])
    cls=np.array([float(s.history.detector_class.iloc[-1]) for s in samples])
    first_box=np.array([s.history[['bbox_left','bbox_top','bbox_right','bbox_bottom']].iloc[-1].to_numpy(float) for s in samples])
    first_conf=np.array([float(s.history.confidence.iloc[-1]) for s in samples])
    p=np.asarray(initial_predictions,float).copy()
    if p.shape!=(n,6,2):raise ValueError('Initial prediction shape differs')
    paths=np.full((n,6*blocks+1,2),np.nan);paths[:,0]=origin
    failed_block=np.zeros(n,dtype=int);valid=np.isfinite(p).all((1,2));failed_block[~valid]=1
    history_xy=None
    for block in range(blocks):
        if block:
            ix=np.flatnonzero(valid)
            if not len(ix):break
            times=t0[ix,None]+3.*(block-1)+DENSE[None]
            if hasattr(model,'predict_arrays'):
                boxes=np.full((len(ix),61,4),np.nan);conf=np.full((len(ix),61),np.nan)
                if block==1:boxes[:,0]=first_box[ix];conf[:,0]=first_conf[ix]
                x=batch_features(times,history_xy[ix],np.broadcast_to(ego[ix,None],times.shape),conf,
                    boxes,np.broadcast_to(cls[ix,None],times.shape))
                nextp=model.predict_arrays(x)
            else:nextp=point_prediction(model,times,history_xy[ix],ego[ix],preserve_prefix=physical_prefix)
            p=np.full((n,6,2),np.nan);p[ix]=nextp
            newly_failed=valid&~np.isfinite(p).all((1,2));failed_block[newly_failed]=block+1
            valid&=np.isfinite(p).all((1,2))
        paths[:,block*6+1:(block+1)*6+1]=p
        history_xy=dense_segment(paths[:,block*6],p)
        print('BATCH ROLLOUT BLOCK',block+1,blocks,'valid',int(valid.sum()),flush=True)
    offsets=np.arange(6*blocks+1,dtype=float)*.5
    completion=predicted_completion(samples,paths,offsets)
    return dict(offsets_s=offsets,positions=paths,failed_block=failed_block,predicted_completion_offset_s=completion,
        generated_duration_s=3.*blocks,initial_origin_s=t0,keys=np.array([s.key for s in samples]),
        generation_policy='fixed before outer evaluation; scoring endpoints not accepted by this generator')
