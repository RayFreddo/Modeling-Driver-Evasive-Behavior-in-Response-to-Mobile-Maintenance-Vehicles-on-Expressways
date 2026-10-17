import numpy as np
def weighted_states(histories,params,tau):
    n=len(histories);lengths=np.array([len(h) for h in histories]);L=int(lengths.max())
    values=np.empty((n,L,3));origin=np.empty((n,2));start=np.zeros(n,int)
    state=np.empty((n,4));cov=np.empty((n,4,4));q,r=params['q'],params['r']
    for i,h in enumerate(histories):
        v=h[['timestamp_s','x_m','y_m']].to_numpy(float)
        if len(v)<3 or not np.isfinite(v).all() or np.any(np.diff(v[:,0])<=0):raise ValueError('Invalid past history')
        values[i,:len(v)]=v;values[i,len(v):]=v[-1];origin[i]=v[-1,1:]
        # Initialize at the SECOND observation and assimilate each observation once.
        # For independent position noise R at z0/z1:
        # Var(z1)=R, Cov(z1,(z1-z0)/dt)=R/dt, Var(v)=2R/dt^2.
        dt=float(v[1,0]-v[0,0]);start[i]=1
        state[i]=np.r_[v[1,1:],(v[1,1:]-v[0,1:])/dt]
        r0=r*np.exp((v[-1,0]-v[0,0])/tau)
        r1=r*np.exp((v[-1,0]-v[1,0])/tau)
        c=np.array([[r1,r1/dt],[r1/dt,(r0+r1)/dt**2]])
        cov[i]=np.zeros((4,4))
        for pos,vel in ((0,2),(1,3)):cov[i][np.ix_([pos,vel],[pos,vel])]=c
    identity=np.eye(4)
    for j in range(1,L):
        ix=np.flatnonzero((lengths>j)&(start<j))
        if not len(ix):continue
        dt=np.maximum(values[ix,j,0]-values[ix,j-1,0],1e-4)
        f=np.tile(identity,(len(ix),1,1));f[:,0,2]=dt;f[:,1,3]=dt
        process=np.zeros_like(f)
        for pos,vel in ((0,2),(1,3)):
            process[:,pos,pos]=q*dt**4/4
            process[:,pos,vel]=process[:,vel,pos]=q*dt**3/2
            process[:,vel,vel]=q*dt**2
        p=f@cov[ix]@f.transpose(0,2,1)+process
        s=(f@state[ix,:,None])[:,:,0]
        age=values[ix,lengths[ix]-1,0]-values[ix,j,0]
        noise=r*np.exp(age/tau)[:,None,None]*np.eye(2)[None]
        gain=np.linalg.solve(p[:,:2,:2]+noise,p[:,:2,:]).transpose(0,2,1)
        s+=(gain@(values[ix,j,1:]-s[:,:2])[:,:,None])[:,:,0]
        a=np.tile(identity,(len(ix),1,1));a[:,:,:2]-=gain
        cov[ix]=a@p@a.transpose(0,2,1)+gain@noise@gain.transpose(0,2,1)
        state[ix]=s
    return state[:,None,:2]-origin[:,None,:]+np.arange(.5,3.01,.5)[None,:,None]*state[:,None,2:]
