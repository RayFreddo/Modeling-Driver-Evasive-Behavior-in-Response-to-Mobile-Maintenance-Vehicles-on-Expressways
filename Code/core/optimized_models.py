from pathlib import Path
import json,time,joblib
import numpy as np
import torch
from torch import nn
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor,GradientBoostingRegressor
from optimized_features import transform,from_history,anchor_basis,HORIZONS
from contracts import sha,write_json as write,dump_new as dump
SEED=20261012
class Head:
    def __init__(self,days):self.training_days=list(days)
    def fit(self,x,y,days):
        if set(days)!=set(self.training_days):raise ValueError('Scope mismatch')
        self.imputer=SimpleImputer(keep_empty_features=True,add_indicator=True)
        z=self.imputer.fit_transform(transform(x));w=np.array([1/np.sum(days==d) for d in days]);w*=len(w)/w.sum()
        self.estimators=[]
        for j in range(y.shape[1]):
            # Three views per original example; both L2 and minimum leaf count scale by three.
            est=HistGradientBoostingRegressor(max_iter=960,max_depth=4,max_leaf_nodes=15,min_samples_leaf=60,
              learning_rate=.04,l2_regularization=60.,early_stopping=False,random_state=SEED)
            est.fit(z,y[:,j],sample_weight=w);self.estimators.append(est)
        return self
    def predict(self,x):
        z=self.imputer.transform(transform(x));return np.column_stack([m.predict(z) for m in self.estimators])

class RobustHistoryHGB:
    def __init__(self,days):self.training_days=list(days)
    def fit(self,x,y,days):
        a,_=anchor_basis(x)
        self.base=Head(self.training_days).fit(x,((y-a)/HORIZONS[None,:,None]).reshape(-1,12),days)
        return self
    def predict_arrays(self,x):
        a,_=anchor_basis(x);return a+self.base.predict(x).reshape(-1,6,2)*HORIZONS[None,:,None]
    def predict_history(self,h,t0):return self.predict_arrays(from_history(h,t0)[None])[0]
class MotionHead(Head):
    def fit(self,x,y,days):
        if set(days)!=set(self.training_days):raise ValueError('Scope mismatch')
        self.imputer=SimpleImputer(keep_empty_features=True,add_indicator=True)
        z=self.imputer.fit_transform(transform(x));w=np.array([1/np.sum(days==d) for d in days]);w*=len(w)/w.sum()
        self.estimators=[]
        for j in range(y.shape[1]):
            est=HistGradientBoostingRegressor(max_iter=960,max_depth=4,max_leaf_nodes=15,min_samples_leaf=60,
              learning_rate=.04,l2_regularization=60.,early_stopping=False,random_state=SEED)
            est.fit(z,y[:,j],sample_weight=w);self.estimators.append(est)
        return self
    def predict(self,x):
        z=self.imputer.transform(transform(x));return np.column_stack([m.predict(z) for m in self.estimators])

class RobustMotionHGB(RobustHistoryHGB):
    def fit(self,x,y,days):
        a,_=anchor_basis(x)
        self.base=MotionHead(self.training_days).fit(x,((y-a)/HORIZONS[None,:,None]).reshape(-1,12),days)
        return self
class WindowHead(MotionHead):
    def fit(self,x,y,days,weights):
        if set(days)!=set(self.training_days):raise ValueError('Scope mismatch')
        self.imputer=SimpleImputer(keep_empty_features=True,add_indicator=True)
        z=self.imputer.fit_transform(transform(x));self.estimators=[]
        if np.any(weights<=0) or not np.isfinite(weights).all():raise ValueError('Invalid vehicle weights')
        for j in range(y.shape[1]):
            est=HistGradientBoostingRegressor(max_iter=960,max_depth=4,max_leaf_nodes=15,min_samples_leaf=140,
              learning_rate=.04,l2_regularization=20.,early_stopping=False,random_state=SEED)
            est.fit(z,y[:,j],sample_weight=weights);self.estimators.append(est)
        return self

class WindowRegularizedHGB(RobustMotionHGB):
    def fit(self,x,y,days,weights):
        a,_=anchor_basis(x)
        self.base=WindowHead(self.training_days).fit(x,((y-a)/HORIZONS[None,:,None]).reshape(-1,12),days,weights)
        return self
class OriginHGB:
    def __init__(self,days):self.training_days=list(days)
    def predict_arrays(self,x):
        origin=x[:,:2]+x[:,2:4]
        return origin[:,None,:]+self.base.predict(x).reshape(-1,6,2)*HORIZONS[None,:,None]
    def predict_history(self,h,t0):return self.predict_arrays(from_history(h,t0)[None])[0]
GBR_CONFIG=dict(n_estimators=240,learning_rate=.05,max_depth=3,max_leaf_nodes=7,min_samples_leaf=140,loss="huber",alpha=.9,subsample=.75,max_features=.75,n_iter_no_change=None,random_state=20261012)
class RobustWindowGBR:
    def __init__(self,days):self.training_days=list(days)
    def fit(self,x,y,days,weights,checkpoint_folder):
        assert set(days)==set(self.training_days)
        self.imputer=SimpleImputer(keep_empty_features=True,add_indicator=True);z=self.imputer.fit_transform(transform(x))
        origin=x[:,:2]+x[:,2:4];target=(y-origin[:,None,:]).reshape(-1,12);self.estimators=[]
        for j in range(12):
            model=GradientBoostingRegressor(**GBR_CONFIG).fit(z,target[:,j],sample_weight=weights)
            self.estimators.append(model);dump(checkpoint_folder/('head_'+str(j)+'.joblib'),model)
            print('EXISTING ROBUST GBR HEAD FROZEN',j+1,12,flush=True)
        return self
    def predict_arrays(self,x):
        z=self.imputer.transform(transform(x));delta=np.column_stack([m.predict(z) for m in self.estimators]).reshape(-1,6,2)
        return (x[:,:2]+x[:,2:4])[:,None,:]+delta
    def predict_history(self,h,t0):return self.predict_arrays(from_history(h,t0)[None])[0]
def pack(x):
    origin=x[:,:2]+x[:,2:4]
    positions=x[:,4:64].reshape(-1,30,2)-origin[:,None,:]
    boxes=x[:,159:243].reshape(-1,12,7)[:,:,2:4]
    old=np.r_[np.arange(-2.875,0,.25),0.];new=np.arange(-2.95,0,.1)
    wh=np.empty((len(x),30,2))
    for i in range(len(x)):
        for j in range(2):wh[i,:,j]=np.interp(new,old,np.r_[boxes[i,:,j],x[i,154+j]])
    return np.column_stack([origin,x[:,144:146],positions.reshape(len(x),-1),wh.reshape(len(x),-1),transform(x)])

class FixedPool(nn.Module):
    def forward(self,x):return x.reshape(x.shape[0],x.shape[1],6,5).mean(-1)

class CNN(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv1d(4,24,5,padding=2),nn.SiLU(),nn.Conv1d(24,24,5,padding=2),nn.SiLU(),nn.Dropout1d(.1),FixedPool())
        self.head=nn.Sequential(nn.Linear(144+width-120,64),nn.SiLU(),nn.Dropout(.2),nn.Linear(64,12))
    def forward(self,x):
        seq=torch.cat([x[:,4:64].reshape(-1,30,2),x[:,64:124].reshape(-1,30,2)],2).transpose(1,2)
        z=self.conv(seq).flatten(1);return self.head(torch.cat([z,x[:,:4],x[:,124:]],1))

class ExistingCNN:
    def __init__(self,days):self.training_days=list(days)
    def fit_statistics(self,x,y,w,days):
        if set(days)!=set(self.training_days):raise ValueError('Scope mismatch')
        px=pack(x);self.width=px.shape[1];self.center=np.nanmedian(px,0)
        self.scale=(np.nanquantile(px,.75,axis=0)-np.nanquantile(px,.25,axis=0))/1.349
        # Shared scale per temporal channel, rather than a different scale at every time bin.
        for left in (4,64):
            a=px[:,left:left+60].reshape(-1,30,2)
            c=np.nanmedian(a,axis=(0,1));s=(np.nanquantile(a,.75,axis=(0,1))-np.nanquantile(a,.25,axis=(0,1)))/1.349
            self.center[left:left+60]=np.tile(c,30);self.scale[left:left+60]=np.tile(s,30)
        self.center=np.where(np.isfinite(self.center),self.center,0);self.scale=np.where(np.isfinite(self.scale)&(self.scale>1e-5),self.scale,1)
        a,_=anchor_basis(x);r=(y-a)/HORIZONS[None,:,None]
        self.ymean=np.average(r,axis=0,weights=w)
        self.yscale=np.maximum(np.sqrt(np.average((r-self.ymean)**2,axis=0,weights=w)),[.1,1.])
    def clean(self,x):
        a=pack(x);return np.clip((np.where(np.isfinite(a),a,self.center)-self.center)/self.scale,-12,12)
    def predict_arrays(self,x):
        net=CNN(self.width);net.load_state_dict({k:torch.from_numpy(v) for k,v in self.state.items()});net.eval()
        with torch.no_grad():r=net(torch.tensor(self.clean(x),dtype=torch.float32)).numpy().reshape(-1,6,2)
        a,_=anchor_basis(x);return a+(r*self.yscale+self.ymean)*HORIZONS[None,:,None]
    def predict_history(self,h,t0):return self.predict_arrays(from_history(h,t0)[None])[0]
def geometry_rates(x):
    origin_y=x[:,1]+x[:,3]
    inverse_now=x[:,157:159]
    slopes=np.stack([x[:,255:257],x[:,276:278],x[:,297:299]],axis=1)
    # If apparent size is proportional to1/range, dy/dt=y*(d inverse_size/dt)/inverse_size.
    values=origin_y[:,None,None]*slopes/inverse_now[:,None,:]
    values=np.where(np.isfinite(values),values,np.nan)
    return values.reshape(len(x),6)

class GeometryCNN(ExistingCNN):
    def fit_statistics(self,x,y,w,days):
        super().fit_statistics(x,y,w,days)
        z=geometry_rates(x);self.geometry_center=np.nanmedian(z,axis=0);self.geometry_center=np.where(np.isfinite(self.geometry_center),self.geometry_center,0.)
        self.geometry_scale=np.maximum((np.nanquantile(z,.75,axis=0)-np.nanquantile(z,.25,axis=0))/1.349,1e-5)
        self.geometry_scale=np.where(np.isfinite(self.geometry_scale),self.geometry_scale,1.);self.width+=6
    def clean(self,x):
        z=np.clip((geometry_rates(x)-self.geometry_center)/self.geometry_scale,-12,12)
        z=np.where(np.isfinite(z),z,0.);return np.column_stack([super().clean(x),z])
CONFIG=dict(base_iterations=960,base_leaf_rows=140,base_depth=4,base_leaf_nodes=15,specialist_iterations=240,corrector_iterations=80,error_iterations=80,meta_leaf_rows=20,meta_depth=3,meta_leaf_nodes=7,L2=20.,learning_rate=.04,seed=SEED,alpha=.5,gamma_m=.1)
def meta(x,a,p,s):
    origin=(x[:,:2]+x[:,2:4])[:,None,:]
    return np.column_stack([transform(x),(a-origin).reshape(-1,12),(p-origin).reshape(-1,12),
        (p-a).reshape(-1,12),s-origin[:,0],s-p[:,-1]])

class Heads:
    def __init__(self,stage,days):self.stage=stage;self.training_days=list(days)
    def fit(self,z,y,days,weights=None):
        assert set(days)==set(self.training_days)
        self.imputer=SimpleImputer(keep_empty_features=True,add_indicator=True);z=self.imputer.fit_transform(z)
        if weights is None:
            weights=np.array([1/np.sum(days==d) for d in days]);weights*=len(weights)/weights.sum()
        self.estimators=[]
        for j in range(y.shape[1]):
            est=HistGradientBoostingRegressor(loss=('absolute_error' if self.stage=='error' else 'squared_error'),
                max_iter=CONFIG[self.stage+'_iterations'],max_depth=3,max_leaf_nodes=7,
                min_samples_leaf=(140 if self.stage=='specialist' else 20),l2_regularization=20.,
                learning_rate=.04,early_stopping=False,random_state=SEED)
            est.fit(z,y[:,j],sample_weight=weights);self.estimators.append(est)
        return self
    def predict(self,z):return np.column_stack([m.predict(self.imputer.transform(z)) for m in self.estimators])

class Upstream:
    def __init__(self,base,specialist,days):self.base=base;self.specialist=specialist;self.training_days=list(days)
    def predict(self,x):
        a,b=anchor_basis(x);p=self.base.predict_arrays(x);s=a[:,-1]+self.specialist.predict(transform(x))
        return a,b,p,s

class ECR:
    def __init__(self,up,corr,error,days):
        self.upstream=up;self.corrector=corr;self.errors=error;self.training_days=list(days);self.alpha=.5;self.gamma=.1
    def predict_with_policy(self,x,mode='full'):
        a,b,p,s=self.upstream.predict(x);p=p.copy()
        if mode=='residual':return p,dict(accepted=np.zeros(len(x),bool))
        z=meta(x,a,p,s);unit=np.einsum('ni,nji->nj',self.corrector.predict(z),b)
        error=np.maximum(self.errors.predict(z),0)
        accepted=error[:,1]+self.gamma<error[:,0]
        if mode=='always':accepted=np.ones(len(x),bool)
        p[accepted,-1]+=self.alpha*unit[accepted]
        return p,dict(accepted=accepted,predicted_base_error_m=error[:,0],predicted_corrected_error_m=error[:,1],unit_correction=unit)
    def predict_arrays(self,x):return self.predict_with_policy(x)[0]
    def predict_history(self,h,t0):return self.predict_arrays(from_history(h,t0)[None])[0]

class Trainer:
    def __init__(self,folder):self.folder=folder;self.up={};self.cors={};self.pipes={};self.logs=[]
    def checkpoint(self,stage,days,fit):
        path=self.folder/(stage+'_'+'_'.join(days));modelpath=path.with_suffix('.joblib');proofpath=path.with_suffix('.json')
        if proofpath.exists():
            proof=json.loads(proofpath.read_text(encoding='utf-8'))
            assert proof['config']==CONFIG and sha(modelpath)==proof['model_sha256']
            assert all(sha(p)==v for p,v in proof['source_hashes'].items())
            self.logs.extend(proof['logs']);return joblib.load(modelpath)
        if modelpath.exists():raise RuntimeError('Incomplete checkpoint retained for inspection: '+str(modelpath))
        start=len(self.logs);model,hashes=fit();dump(modelpath,model)
        write(proofpath,dict(config=CONFIG,stage=stage,training_days=list(days),source_hashes=hashes,
            model_sha256=sha(modelpath),logs=self.logs[start:],external_validation_access=False))
        return model
    def record(self,stage,days,keys,deps):
        assert all(set(dep)<=set(days) for dep in deps)
        self.logs.append(dict(stage=stage,training_days=list(days),training_keys=list(keys),dependency_training_scopes=[list(d) for d in deps]))
    def upstream(self,days):
        days=tuple(sorted(days))
        if days not in self.up:
            def fit():
                data=scope_data(days);a,_=anchor_basis(data['x'])
                base=WindowRegularizedHGB(days).fit(data['x'],data['y'],data['days'],data['weights'])
                specialist=Heads('specialist',days).fit(transform(data['x']),data['y'][:,-1]-a[:,-1],data['days'],data['weights'])
                self.record('window_base_specialist',days,data['keys'],[])
                print('ROBUST ECR UPSTREAM FROZEN',days,len(data['actors']),flush=True)
                return Upstream(base,specialist,days),data['hashes']
            self.up[days]=self.checkpoint('upstream',days,fit)
        return self.up[days]
    def corrector(self,days):
        days=tuple(sorted(days));assert len(days)>=2
        if days not in self.cors:
            def fit():
                zz=[];yy=[];ds=[];keys=[];deps=[];hashes={}
                for held in days:
                    td=tuple(d for d in days if d!=held);data=scope_data((held,));x=data['ox'];y=data['oy']
                    a,b,p,s=self.upstream(td).predict(x)
                    zz.append(meta(x,a,p,s));yy.append(np.einsum('ni,nij->nj',y[:,-1]-p[:,-1],b))
                    ds.extend([held]*len(x));keys.extend(data['keys']);deps.append(td);hashes.update(data['hashes'])
                model=Heads('corrector',days).fit(np.concatenate(zz),np.concatenate(yy),np.array(ds))
                self.record('corrector_date_OOF',days,keys,deps);return model,hashes
            self.cors[days]=self.checkpoint('corrector',days,fit)
        return self.cors[days]
    def pipeline(self,days):
        days=tuple(sorted(days));assert len(days)>=3
        if days not in self.pipes:
            def fit():
                zz=[];ee=[];ds=[];keys=[];deps=[];hashes={}
                for held in days:
                    td=tuple(d for d in days if d!=held);data=scope_data((held,));x=data['ox'];y=data['oy']
                    a,b,p,s=self.upstream(td).predict(x);z=meta(x,a,p,s)
                    unit=np.einsum('ni,nji->nj',self.corrector(td).predict(z),b)
                    zz.append(z);ee.append(np.column_stack([np.linalg.norm(y[:,-1]-p[:,-1],axis=1),
                        np.linalg.norm(y[:,-1]-p[:,-1]-.5*unit,axis=1)]))
                    ds.extend([held]*len(x));keys.extend(data['keys']);deps.append(td);hashes.update(data['hashes'])
                error=Heads('error',days).fit(np.concatenate(zz),np.concatenate(ee),np.array(ds))
                up=self.upstream(days);corr=self.corrector(days)
                self.record('gate_nested_date_OOF',days,keys,deps)
                return ECR(up,corr,error,days),hashes
            self.pipes[days]=self.checkpoint('pipeline',days,fit)
        return self.pipes[days]
