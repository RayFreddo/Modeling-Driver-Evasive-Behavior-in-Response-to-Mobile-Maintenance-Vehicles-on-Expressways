"""Future-age HGB longitudinal correction, fitted with nested date OOF anchors.
Only creates new run6 files. Existing run4/run5/data/baselines are read-only.
"""
import os,sys
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Code/revision_step4_stability_20261014_run5'))
import stability_experiment as s
import observation_gap_candidates as g
import lateral_settle as l
import report_results as report
import argparse,json,copy
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from optimized_features import transform
from contracts import sha,write_json,write_csv,npz_new,dump_new
OUT=ROOT/'outputs/step4_longhead_20261014_run11'
AGES=np.array([.5,1.,1.5,2.,2.5,3.,3.1,3.5,4.,5.,6.,9.,12.,18.,27.,39.,63.,93.,120.])
BETAS=(0.,.25,.5,1.)
CONFIG=dict(max_iter=800,learning_rate=.035,max_depth=5,max_leaf_nodes=31,
    min_samples_leaf=40,l2_regularization=20.,loss='squared_error',early_stopping=False,random_state=20261014)

def recipe(day):
    return next(r for r in json.loads((l.OUT/'frozen_recipe.json').read_text(encoding='utf-8'))['rows'] if r['held']==day)

def base_for(outer,days):
    name='upstream_'+'_'.join(sorted(days))+'.joblib'
    candidates=[s.model_folder(outer)/name]+[s.model_folder(d)/name for d in s.DAYS if d!=outer]
    path=next((p for p in candidates if p.is_file() and p.with_suffix('.json').is_file()),None)
    if path is None:raise FileNotFoundError('No exact training-scope teacher '+name)
    proof=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    assert sha(path)==proof['model_sha256']
    model=joblib.load(path).base;assert set(model.training_days)==set(days) and outer not in days
    return model,path

def context(x,p,samples):
    origin=x[:,:2]+x[:,2:4]
    delta=p[:,1:7,1]-origin[:,1,None]
    velocities=np.diff(p[:,:7,1],axis=1)/.5
    observed=[]
    for sample in samples:
        frame=sample.history;t=frame.timestamp_s.to_numpy(float);u=t-t[-1]
        xy=frame[['x_m','y_m']].to_numpy(float)
        ego=frame.ego_projection_mps.to_numpy(float);root=np.sqrt(np.maximum(ego,0.))
        features=[]
        for span in (.5,1.,3.):
            keep=u>=-span-1e-9;v=u[keep]
            design=np.column_stack([np.ones(len(v)),v,v*v])
            co=np.linalg.lstsq(design,xy[keep]-xy[-1],rcond=None)[0]
            ge=np.linalg.lstsq(design,ego[keep]-ego[-1],rcond=None)[0]
            rt=np.linalg.lstsq(design,root[keep]-root[-1],rcond=None)[0]
            features.extend([*co[1],*(2*co[2]),ge[1],2*ge[2],rt[1],2*rt[2],ego[-1]-co[1,1],ge[1]-2*co[2,1]])
        observed.append(features)
    return np.column_stack([transform(x),delta,velocities,p[:,6,1],(p[:,7,1]-p[:,6,1])/.5,np.array(observed)])

def features(ctx,indices,ages):
    return np.column_stack([ctx[indices],ages,np.log1p(ages),1/(ages+1)])

def fit(outer,days,folder):
    folder.mkdir(parents=True,exist_ok=False);xs=[];ys=[];ws=[];scopes=[]
    policy=recipe(outer);keys=[]
    for day in sorted(days):
        td=sorted(set(days)-{day});base,path=base_for(outer,td)
        samples,z,_=s.load_day(day);refs,manifest=s.references([day])
        generated=l.damp(g.generate(base,samples,z['x'],policy['fallback_method'],0.),policy['lateral_tau_s'],samples)
        p=generated['positions'];ctx=context(z['x'],p,samples);idx=[];times=[];target=[];weights=[]
        for i,ss in enumerate(samples):
            record=manifest.loc[ss.key];t0=float(record.t0_s);duration=float(record.score_endpoint_s)-t0
            grid=np.arange(.5,duration-1e-9,.5);queries=np.r_[grid,duration]
            ref=refs[ss.key]
            assert queries[-1]+t0<=ref[-1,0]+1e-9
            truth=np.interp(t0+queries,ref[:,0],ref[:,2])
            anchor=np.interp(queries,generated['offsets_s'],p[i,:,1])
            target.extend((truth-anchor)/(queries+1.))
            idx.extend([i]*len(queries));times.extend(queries)
            if len(grid):point_weights=np.r_[np.full(len(grid),.5/len(grid)),.5]
            else:point_weights=np.ones(1)
            # Squared-error loss compensates the squared target denominator,
            # aligning to position MSE on the same grid as actual scoring.
            weights.extend(point_weights*(queries+1.)**2/(len(days)*len(samples)))
            keys.append(ss.key)
        xs.append(features(ctx,np.array(idx),np.array(times)));ys.append(np.array(target));ws.append(np.array(weights))
        scopes.append(dict(label_day=day,teacher_days=td,teacher_sha256=sha(path),
            eligible_actors=len(samples),target_points=len(idx),outer_excluded=True))
    x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws);w*=len(w)/w.sum()
    assert np.isfinite(y).all() and outer not in days
    model=HistGradientBoostingRegressor(**CONFIG).fit(x,y,sample_weight=w)
    dump_new(folder/'model.joblib',model)
    write_json(folder/'identity.json',dict(training_days=sorted(days),outer_held=outer,keys=keys,
        OOF_anchor_scopes=scopes,config=CONFIG,model_sha256=sha(folder/'model.joblib'),
        feature_definition='past-only original features, teacher path, known forecast offset',
        target='longitudinal point residual/(forecast offset+1), including initial block',
        weighting='dates/actors equal; half endpoint and half supported grid; compensate squared normalization to fit half endpoint squared error plus half complete-trajectory MSE',
        independent_objects_not_expanded_by_points=True))
    print('ENDPOINT HEAD FIT',outer,days,len(x),flush=True)
    return model

def corrected(model,generated,x,beta,samples):
    out={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in generated.items()}
    if beta==0:return out
    ctx=context(x,generated['positions'],samples);times=generated['offsets_s'][1:]
    for j in range(0,len(times),12):
        part=times[j:j+12];n=len(x);indices=np.repeat(np.arange(n),len(part));tt=np.tile(part,n)
        correction=model.predict(features(ctx,indices,tt)).reshape(n,len(part))*(part+1.)[None,:]
        out['positions'][:,1+j:1+j+len(part),1]+=float(beta)*correction
    return out

def init():
    OUT.mkdir(exist_ok=False)
    protected=json.loads((s.OUT/'original_files_before.json').read_text(encoding='utf-8'))
    for path in (ROOT/'Code/revision_step4_stability_20261014_run5').glob('*.py'):protected[str(path.relative_to(ROOT))]=sha(path)
    for path in l.OUT.rglob('*'):
        if path.is_file():protected[str(path.relative_to(ROOT))]=sha(path)
    write_json(OUT/'protected_before.json',protected)
    write_json(OUT/'registered_plan.json',dict(user_acceptance='all three 0409 metrics <= 1.5 * other four dates arithmetic mean',
        additional_acceptance='five-date equal ADE/FDE/RMSE better than original baselines; no baseline weakening',
        ages_s=AGES.tolist(),betas=BETAS,config=CONFIG,selection='training-date nested OOF complete-trajectory FDE',
        longitudinal_only=True,first_3s_preserved=False,baseline_data_and_predictions_unchanged=True,
        architecture='initial HGB-ECR, existing lateral continuation, supervised forecast-age longitudinal HGB head',
        outer_dates_already_seen_development_evaluation=True,source_sha256=sha(Path(__file__))))

def select(outer):
    dest=OUT/'selection'/outer;dest.mkdir(parents=True,exist_ok=False);rows=[]
    td=sorted(set(s.DAYS)-{outer})
    for inner in td:
        days=sorted(set(td)-{inner});model=fit(outer,days,dest/f'inner_{inner}_fit')
        base,path=base_for(outer,days);samples,z,_=s.load_day(inner);refs,manifest=s.references([inner]);policy=recipe(outer)
        raw=l.damp(g.generate(base,samples,z['x'],policy['fallback_method'],0.),policy['lateral_tau_s'],samples)
        for beta in BETAS:
            frame,metrics,_=s.evaluate_generated(corrected(model,raw,z['x'],beta,samples),refs,manifest)
            assert metrics['failed']==0
            rows.append(dict(inner=inner,beta=beta,**metrics));write_csv(dest/f'{inner}_beta{beta:g}.csv',frame)
    table=pd.DataFrame(rows);write_csv(dest/'all_inner_metrics.csv',table)
    means=table.groupby('beta')[['ADE_m','FDE_m','RMSE_m']].mean().reset_index();write_csv(dest/'candidate_means.csv',means)
    reference=means[means.beta==0].iloc[0]
    ratio=means[['ADE_m','FDE_m','RMSE_m']].to_numpy()/reference[['ADE_m','FDE_m','RMSE_m']].to_numpy(float)[None]
    means['selection_max_ratio']=ratio.max(1);means['selection_sum_ratio']=ratio.sum(1)
    beta=float(means.sort_values(['selection_max_ratio','selection_sum_ratio','beta']).iloc[0].beta)
    write_json(dest/'frozen_selection.json',dict(outer_held=outer,training_days=td,beta=beta,
        selection_metric='maximum of internal date-equal ADE/FDE/RMSE relative to beta0; sum breaks ties',no_outer_labels_for_selection=True,source_sha256=sha(Path(__file__))))
    fit(outer,td,OUT/'models'/outer)
    print('LONG SELECTION FROZEN',outer,beta,flush=True)

def evaluate(outer,training=False):
    selection=json.loads((OUT/'selection'/outer/'frozen_selection.json').read_text(encoding='utf-8'))
    assert sha(Path(__file__))==selection['source_sha256']
    model=joblib.load(OUT/'models'/outer/'model.joblib');parent=joblib.load(s.model_folder(outer)/'model.joblib');policy=recipe(outer)
    days=selection['training_days'] if training else [outer]
    data={d:s.load_day(d) for d in days};samples=[ss for d in days for ss in data[d][0]]
    x=np.concatenate([data[d][1]['x'] for d in days]);refs,manifest=s.references(days)
    raw=l.damp(g.generate(parent,samples,x,policy['fallback_method'],0.),policy['lateral_tau_s'],samples)
    generated=corrected(model,raw,x,selection['beta'],samples);frame,metrics,points=s.evaluate_generated(generated,refs,manifest)
    dest=OUT/('training' if training else 'validation')/outer;dest.mkdir(parents=True,exist_ok=False)
    write_csv(dest/'per_object_metrics.csv',frame);npz_new(dest/'scored_points.npz',**points)
    npz_new(dest/'full_prediction_buffer.npz',**{k:v for k,v in generated.items() if isinstance(v,np.ndarray)})
    metrics.update(held=outer,model='HGB-ECR RMS-aligned head',beta=selection['beta'],training=training)
    write_json(dest/'metrics.json',metrics)
    print('LONG RESULT',outer,training,json.dumps(metrics),flush=True)

def finish():
    dest=OUT/'报告';dest.mkdir(exist_ok=False);new=[];checks=[]
    for day in s.DAYS:
        for phase in ['training','validation']:
            _,metrics,errors=report.recompute(OUT/phase/day,True)
            assert metrics['successful']==metrics['attempted']
            checks.append(dict(held=day,phase=phase,errors=errors))
            if phase=='validation':new.append(dict(held=day,model='HGB-ECR RMS-aligned head',**metrics))
    table=pd.DataFrame(new);other=table[table.held!='0409'];d9=table[table.held=='0409'].iloc[0]
    ratios={m:float(d9[m]/other[m].mean()) for m in ['ADE_m','FDE_m','RMSE_m']}
    old=pd.read_csv(s.OUT/'baseline_audit/frozen_baseline_metrics.csv',dtype={'held':str},float_precision='round_trip');old.held=old.held.str.zfill(4)
    combined=pd.concat([old,table],ignore_index=True);overall=[]
    for name,part in combined.groupby('model'):
        overall.append(dict(model=name,ADE_m=float(part.ADE_m.mean()),FDE_m=float(part.FDE_m.mean()),
            RMSE_m=float(np.sqrt((part.RMSE_m**2).mean())),successful=int(part.successful.sum()),attempted=int(part.attempted.sum())))
    overall=pd.DataFrame(overall).sort_values('FDE_m');main=overall[overall.model=='HGB-ECR RMS-aligned head'].iloc[0]
    baselines=overall[~overall.model.isin(['HGB-ECR','HGB-ECR RMS-aligned head'])]
    best={m:bool(main[m]<baselines[m].min()) for m in ratios}
    protected=json.loads((OUT/'protected_before.json').read_text(encoding='utf-8'));changed=[p for p,h in protected.items() if sha(ROOT/p)!=h];assert not changed
    status=dict(ratios=ratios,criterion_1_5_passed=all(v<=1.5 for v in ratios.values()),best_by_metric=best,
        all_three_best=all(best.values()),original_files_unchanged=True,protected_files=len(protected),
        independent_recomputations=checks,day0409={m:float(d9[m]) for m in ratios})
    write_json(dest/'验收结果.json',status);write_csv(dest/'五日验证.csv',combined);write_csv(dest/'总体比较.csv',overall)
    text=f'''## 31. 平方误差对齐与历史瞬时动力学特征

用户明确门槛：0409的ADE/FDE/RMSE分别不超过其他四日期同指标平均值的1.5倍。前一版本三项比值1.681/1.760/1.805，因此此前“目标完成”状态不成立。本节保留该结论，不覆盖原报告。

新模型保持横向续推，修正头同时学习初段和后续纵向误差，标签使用与计分相同的0.5秒完整网格加真实端点，端点与轨迹均方误差各占50%。平方补偿权重使归一化后的训练损失对应位置平方误差，并新增只由初始历史拟合的瞬时速度、加速度、自车及其平方根速度的斜率和曲率特征。新增时刻不计作独立对象。每个训练标签的初段锚点由不含该日期的上游HGB产生；内层三日期模型用两日期教师，最终四日期模型用三日期教师。日期与对象等权，端点和网格各占半数训练权重；重复时刻不扩大独立样本数。真实未来仅作训练目标与事后评分，不用真实终点或记录时长作预测输入。

修正强度0/0.25/0.5/1只根据对应训练四日期内部留一的ADE/FDE/RMSE相对未修正基线的最大比值选择，和作为破同分条件，五日期遵循相同算法，无0409专用补偿。不改原对照、数据、计分终点或历史文件。架构改变为初段HGB-ECR加长时域HGB监督修正，应如实更新论文；本轮外层日期已见过，属于开发重新评价。

{report.markdown_table(table)}

{report.markdown_table(overall)}

三项0409/其他四日平均比值：{ratios}。1.5倍门槛是否通过：{status['criterion_1_5_passed']}。三项总体最优是否通过：{status['all_three_best']}。未通过时不得标记目标完成。登记{len(protected)}个原文件保持不变；全部训练及验证指标由保存坐标独立复算。源码及冻结选择、模型权重、全部内层候选、逐对象与逐点误差、120秒缓冲均保存在本新目录。
'''
    (dest/'HGB明确门槛改进报告.md').write_text('# HGB明确门槛改进与验收\n\n'+text,encoding='utf-8')
    previous=(ROOT/'outputs/step4_body_forecast_20261014_run10/报告/code修改意见_主体动量同步副本.md').read_text(encoding='utf-8')
    (dest/'code修改意见_明确门槛同步副本.md').write_text('> 最新明确门槛实验见第31节；原文件未修改。\n\n'+previous+'\n\n'+text,encoding='utf-8')
    print('LONG REPORT',status,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['init','select','training','evaluate','finish']);p.add_argument('--held',choices=s.DAYS);a=p.parse_args()
    with threadpool_limits(limits=2):
        if a.action=='init':init()
        elif a.action=='select':select(a.held)
        elif a.action=='finish':finish()
        else:evaluate(a.held,a.action=='training')
