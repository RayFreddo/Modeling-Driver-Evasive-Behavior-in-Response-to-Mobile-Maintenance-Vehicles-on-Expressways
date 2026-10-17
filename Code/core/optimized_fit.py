"""Fresh fitting of retained optimized architecture/configs, with exact training scopes."""
import numpy as np
import torch
from torch import nn
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from optimized_models import OriginHGB,WindowHead,GeometryCNN,CNN
from optimized_features import transform,from_history,anchor_basis,HORIZONS
from contracts import dump_new,write_json
SEEDS=(20261012,20261013,20261014)
HGB_CONFIG=dict(max_iter=960,max_depth=4,max_leaf_nodes=15,min_samples_leaf=140,learning_rate=.04,l2_regularization=20.,early_stopping=False,random_state=20261012,max_features=1.)
CNN_CONFIG=dict(epochs=40,batch_size=256,learning_rate=.0005,weight_decay=.02,dropout=[.1,.2],seeds=SEEDS)

def training_scope(data,days):
    xs=[];ys=[];ws=[];ds=[];ox=[];oy=[];od=[];keys=[];hashes={};actors=[]
    for day in sorted(days):
        samples,z,proof=data[day];n=len(samples);views=[]
        rng=np.random.default_rng(20261014+int(day))
        for i,s in enumerate(samples):
            h=s.history;t0=float(h.timestamp_s.iloc[-1]);variants=[h]
            # Exact source noise recipe: labels never perturbed; no smoothing.
            for _ in range(2):
                g=h.copy()
                g[['x_m','y_m']]=h[['x_m','y_m']].to_numpy()+rng.normal(0,.02,(len(h),2))
                cols=['bbox_left','bbox_top','bbox_right','bbox_bottom']
                g[cols]=h[cols].to_numpy()+np.clip(rng.normal(0,1.,(len(h),4)),-2.,2.)
                # Genuine missing observations stay missing. Never invent boxes.
                if np.any(g.bbox_right<=g.bbox_left) or np.any(g.bbox_bottom<=g.bbox_top):
                    raise ValueError('Invalid source-recipe noisy box; stop, do not filter')
                variants.append(g)
            actual=[from_history(g,t0) for g in variants]
            np.testing.assert_allclose(actual[0],z['x'][i],rtol=0,atol=0,equal_nan=True)
            views.extend(actual)
        xs.append(np.array(views));ys.append(np.repeat(z['y'],3,axis=0));ws.append(np.full(3*n,1/(len(days)*n*3)))
        ds.extend([day]*(3*n));ox.append(z['x']);oy.append(z['y']);od.extend([day]*n);keys.extend(z['keys'])
        actors.extend(day+':vehicle:'+str(s.manifest['object_id']) for s in samples)
        hashes.update({k:v for k,v in proof['input_sha256'].items()})
    weights=np.concatenate(ws);weights*=len(weights)/weights.sum()
    return dict(x=np.concatenate(xs),y=np.concatenate(ys),weights=weights,w=weights,days=np.array(ds),
      ox=np.concatenate(ox),oy=np.concatenate(oy),od=np.array(od),keys=np.array(keys),actors=actors,hashes=hashes,td=list(sorted(days)))

def fit_origin(data,folder):
    model=OriginHGB(data['td']);model.base=WindowHead(data['td']);model.base.imputer=SimpleImputer(keep_empty_features=True,add_indicator=True)
    z=model.base.imputer.fit_transform(transform(data['x']));origin=data['x'][:,:2]+data['x'][:,2:4]
    y=((data['y']-origin[:,None])/HORIZONS[None,:,None]).reshape(-1,12);model.base.estimators=[]
    for j in range(12):
        est=HistGradientBoostingRegressor(**HGB_CONFIG).fit(z,y[:,j],sample_weight=data['weights']);model.base.estimators.append(est)
        dump_new(folder/f'head_{j}.joblib',est);print('OPTIMIZED ORIGIN HEAD',j+1,12,flush=True)
    model.config=HGB_CONFIG.copy();return model

class CNNEnsemble:
    def __init__(self,models):self.models=models;self.training_days=models[0].training_days
    def predict_arrays(self,x):return np.mean([m.predict_arrays(x) for m in self.models],axis=0)
    def predict_history(self,h,t0):return self.predict_arrays(from_history(h,t0)[None])[0]

def fit_cnn(data,folder,device='cuda'):
    models=[];logs=[];torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    x,y,w=data['x'],data['y'],data['weights'];a,_=anchor_basis(x)
    for seed in SEEDS:
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);m=GeometryCNN(data['td']);m.fit_statistics(x,y,w,data['days']);net=CNN(m.width).to(device)
        xx=torch.tensor(m.clean(x),dtype=torch.float32,device=device);yy=torch.tensor(y,dtype=torch.float32,device=device);aa=torch.tensor(a,dtype=torch.float32,device=device)
        ww=torch.tensor(w/w.sum()*len(w),dtype=torch.float32,device=device);ym=torch.tensor(m.ymean,dtype=torch.float32,device=device);ys=torch.tensor(m.yscale,dtype=torch.float32,device=device)
        hh=torch.tensor(HORIZONS,dtype=torch.float32,device=device);hw=torch.tensor([1,1,1,1,1,2],dtype=torch.float32,device=device)/7
        opt=torch.optim.AdamW(net.parameters(),lr=CNN_CONFIG['learning_rate'],weight_decay=CNN_CONFIG['weight_decay']);rng=np.random.default_rng(seed)
        for epoch in range(1,CNN_CONFIG['epochs']+1):
            net.train();order=rng.permutation(len(x));total=0.
            for start in range(0,len(x),CNN_CONFIG['batch_size']):
                ix=torch.tensor(order[start:start+CNN_CONFIG['batch_size']],device=device);opt.zero_grad(set_to_none=True)
                p=aa[ix]+(net(xx[ix]).reshape(-1,6,2)*ys+ym)*hh[None,:,None];sq=(p-yy[ix]).square().sum(-1)
                loss=(((torch.sqrt(sq+.01)+.05*sq)*hw).sum(1)*ww[ix]).mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite optimized CNN loss')
                loss.backward();nn.utils.clip_grad_norm_(net.parameters(),10);opt.step();total+=float(loss.detach().cpu())*len(ix)/len(x)
            if epoch%10==0:logs.append(dict(seed=seed,epoch=epoch,loss=total));print('OPTIMIZED CNN',seed,epoch,total,flush=True)
        m.state={k:v.detach().cpu().numpy().copy() for k,v in net.state_dict().items()};m.seed=seed;m.epoch=CNN_CONFIG['epochs'];m.config=CNN_CONFIG.copy()
        dump_new(folder/f'seed_{seed}.joblib',m);models.append(m)
    write_json(folder/'loss_history.json',logs);return CNNEnsemble(models)
