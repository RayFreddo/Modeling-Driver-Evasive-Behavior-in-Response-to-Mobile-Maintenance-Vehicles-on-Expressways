"""Replay all frozen holdout coordinates and generate the single final error report."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from contracts import ROOT,DAYS,FOLD_ROOT,sha,write_json,write_csv
from train_optimized import RUN,OUT
from evaluate_holdout import PUBLIC
from full_training_check import references

def verify_model(held,name,folder,refs,manifest):
    identity=json.loads((folder/'identity.json').read_text(encoding='utf-8'))
    metrics=json.loads((folder/'validation_metrics.json').read_text(encoding='utf-8'))
    frame=pd.read_csv(folder/'per_object_metrics.csv',dtype={'day':str},float_precision='round_trip')
    assert identity['held']==held and set(identity['training_days'])==set(DAYS)-{held}
    assert len(frame)==len(manifest) and set(frame.sample_key)==set(manifest.index) and not frame.sample_key.duplicated().any()
    assert set(frame.day)=={held} and metrics['validation'] and metrics['attempted']==len(frame)
    for filename,value in identity['artifact_sha256'].items():assert sha(folder/filename)==value
    bykey=frame.set_index('sample_key');independent={};maximum=0.
    with np.load(folder/'scored_points.npz') as archive:
        values=archive['values'];offsets=archive['offsets'];keys=archive['keys']
    assert set(keys)==set(frame.loc[frame.score_success,'sample_key'])
    for i,key in enumerate(keys):
        key=str(key);row=bykey.loc[key];a=values[offsets[i]:offsets[i+1]];ref=refs[key];m=manifest.loc[key]
        assert len(a)==int(row.score_point_count) and abs(a[-1,0]-float(m.score_endpoint_s))<1e-9
        assert np.all(np.diff(a[:,0])>0) and a[0,0]>=ref[0,0]-1e-10 and a[-1,0]<=ref[-1,0]+1e-10
        q=np.r_[np.arange(.5,float(m.score_duration_s)-1e-10,.5),float(m.score_duration_s)]
        np.testing.assert_allclose(a[:,0],float(m.t0_s)+q,rtol=0,atol=1e-9)
        truth=np.column_stack([np.interp(a[:,0],ref[:,0],ref[:,j]) for j in [1,2]])
        np.testing.assert_allclose(a[:,4:6],truth,rtol=0,atol=1e-8)
        error=np.linalg.norm(a[:,2:4]-truth,axis=1);np.testing.assert_allclose(a[:,6],error,rtol=0,atol=1e-8)
        values_i=dict(ADE_m=float(error.mean()),FDE_m=float(error[-1]),MSE_m2=float(np.mean(error**2)))
        for metric,value in values_i.items():
            maximum=max(maximum,abs(float(row[metric])-value));assert abs(float(row[metric])-value)<1e-8
        independent[key]=values_i
    valid=frame.score_success
    assert metrics['successful']==int(valid.sum()) and metrics['failed']==int((~valid).sum())
    replay=dict(ADE_m=float(frame.loc[valid,'ADE_m'].mean()),FDE_m=float(frame.loc[valid,'FDE_m'].mean()),
        RMSE_m=float(np.sqrt(frame.loc[valid,'MSE_m2'].mean())))
    for metric,value in replay.items():assert abs(value-float(metrics[metric]))<1e-8
    return frame,dict(held=held,model=name,**replay,N_attempted=len(frame),N_successful=int(valid.sum()),N_failed=int((~valid).sum()),
        failure_rate=float((~valid).mean()),predicted_completed=int(frame.model_predicted_completed.sum()),
        predicted_unconfirmed=int((~frame.model_predicted_completed).sum()),coordinate_replay_max_difference_m=maximum,
        exact_all_endpoints=True,source_truth_replayed=True,source_folder=str(folder))

def intervals(frames,seed=20261014,draws=1000):
    rng=np.random.default_rng(seed);result=[]
    for model in PUBLIC:
        byday=[frames[(d,model)].loc[frames[(d,model)].score_success,['ADE_m','FDE_m','MSE_m2']].to_numpy(float) for d in DAYS]
        mean=np.mean([a.mean(0) for a in byday],axis=0);bootstrap=[]
        for _ in range(draws):
            a=np.mean([z[rng.integers(0,len(z),len(z))].mean(0) for z in byday],axis=0)
            bootstrap.append([a[0],a[1],np.sqrt(a[2])])
        b=np.array(bootstrap);ci=np.quantile(b,[.025,.975],axis=0)
        result.append(dict(model=model,ADE_m=float(mean[0]),FDE_m=float(mean[1]),RMSE_m=float(np.sqrt(mean[2])),
            ADE_ci95_low=float(ci[0,0]),ADE_ci95_high=float(ci[1,0]),FDE_ci95_low=float(ci[0,1]),FDE_ci95_high=float(ci[1,1]),
            RMSE_ci95_low=float(ci[0,2]),RMSE_ci95_high=float(ci[1,2]),N_attempted=4890,
            N_successful=sum(int(frames[(d,model)].score_success.sum()) for d in DAYS),
            CI_scope='vehicle resampling within five fixed observed dates, conditional on score success; not external-date uncertainty'))
    return result

def paired(frames,draws=1000):
    rng=np.random.default_rng(20261015);result=[]
    for baseline in [m for m in PUBLIC if m!='HGB-ECR']:
        days=[];counts=[]
        for day in DAYS:
            a=frames[(day,'HGB-ECR')].set_index('sample_key');b=frames[(day,baseline)].set_index('sample_key')
            common=a.index[a.score_success&b.loc[a.index,'score_success']]
            delta=b.loc[common,'FDE_m'].to_numpy()-a.loc[common,'FDE_m'].to_numpy();days.append(delta);counts.append(len(common))
        estimate=float(np.mean([a.mean() for a in days]));boot=[]
        for _ in range(draws):boot.append(np.mean([a[rng.integers(0,len(a),len(a))].mean() for a in days]))
        ci=np.quantile(boot,[.025,.975]);result.append(dict(baseline=baseline,mean_baseline_minus_HGB_ECR_FDE_m=estimate,
            ci95_low=float(ci[0]),ci95_high=float(ci[1]),paired_common_N=sum(counts),paired_common_N_by_date=dict(zip(DAYS,counts)),
            full_cohort_not_removed=True,interpretation='positive means HGB-ECR lower FDE on these paired successful objects; failures separately reported'))
    return result

def main():
    status=json.loads((OUT/'all_holdout_predictions_complete.json').read_text(encoding='utf-8'));assert status['passed'] and len(status['results'])==5
    dest=OUT/'交付_五折完整避险结果_20261014';dest.mkdir(exist_ok=False);rows=[];frames={};folds=[]
    for held in DAYS:
        folder=FOLD_ROOT/f'holdout_{held}'/'03_验证评价'/RUN
        proof=json.loads((folder/'validation_complete.json').read_text(encoding='utf-8'));assert proof['passed']
        refs,manifest=references([held]);n=len(manifest)
        for name in PUBLIC+['HGB-origin-ablation']:
            frame,row=verify_model(held,name,folder/name,refs,manifest);frames[(held,name)]=frame;rows.append(row)
            print('INDEPENDENT HOLDOUT COORDINATE REPLAY PASS',held,name,row['FDE_m'],flush=True)
        reg=json.loads((folder/'validation_registration.json').read_text(encoding='utf-8'))
        assert reg['validation_n']==n and reg['training_and_validation_total']==4890 and reg['training_n']+n==4890
        train_keys=json.loads((FOLD_ROOT/f'holdout_{held}'/'02_模型训练'/RUN/'training_identity.json').read_text(encoding='utf-8'))['training_keys']
        assert not set(train_keys)&set(manifest.index)
        folds.append(dict(held=held,N_training=reg['training_n'],N_validation=n,train_test_keys_disjoint=True))
    assert sum(r['N_validation'] for r in folds)==4890
    table=pd.DataFrame(rows);write_csv(dest/'五折各模型验证误差与覆盖.csv',table)
    global_rows=intervals(frames);write_csv(dest/'五折模型总体误差与置信区间.csv',pd.DataFrame(global_rows))
    comparisons=paired(frames);write_csv(dest/'共同成功样本配对比较.csv',pd.DataFrame(comparisons))
    training=json.loads((OUT/'独立训练核验_final_prefix/verification_and_training_gates.json').read_text(encoding='utf-8'))
    write_csv(dest/'五折各模型完整训练误差.csv',pd.DataFrame(training['rows']))
    texts=['# 五折完整避险轨迹预测误差','',
        '使用指定五份修正Excel的全部4890个日期/对象，四天训练、一天留出。下列为留出日验证误差，单位为米；训练误差另表保存。',
        '输入前约3秒历史，每轮预测后3秒并用自身预测递推。确认换道完成则在精确完成时刻评分；没有记录到完成则按最后有效记录评分。超出记录的预测不补造真值。',
        '误差相对于离线平滑、重建及部分延伸的修正参考，不代表独立现场测量真值准确度。预测完成诊断与评分覆盖是两件事，未确认完成或物理失效的输出均保留状态。','',
        '| 留出日 | 训练对象 | 验证对象 |','|---|---:|---:|']
    texts.extend(f"| {r['held']} | {r['N_training']} | {r['N_validation']} |" for r in folds)
    for held in DAYS:
        texts.extend(['',f'## 留出 {held}','', '| 模型 | ADE | FDE | RMSE | 成功评分/全部 | 失败 | 预测确认完成 |','|---|---:|---:|---:|---:|---:|---:|'])
        for model in PUBLIC:
            r=next(q for q in rows if q['held']==held and q['model']==model)
            texts.append(f"| {model} | {r['ADE_m']:.3f} | {r['FDE_m']:.3f} | {r['RMSE_m']:.3f} | {r['N_successful']}/{r['N_attempted']} | {r['N_failed']} | {r['predicted_completed']} |")
        texts.append('')
        texts.append('失败对象没有被删除；该模型误差均值基于能覆盖精确评分终点的对象。模型比较还应阅读共同成功样本配对表。')
    texts.extend(['','## 统计与复核','',
        '总体误差先按日、日内按对象等权。RMSE是各对象均方欧氏误差再按日平均后开根号。置信区间对各固定日期内的车辆重采样，不能当作从未观察日期的外部泛化区间。',
        '所有验证数值已从保存的绝对预测/参考坐标独立重算，终点时刻与全对象清单逐条核对。HGB-origin-ablation是关闭ECR的原点对照，不代替主HGB-ECR。',
        '验证开始后没有根据留出误差更新模型、归一化、参数或评分规则。',
        '逐点坐标、逐对象误差、完整生成缓冲、预测完成状态和模型指纹位于各holdout的03_验证评价/step4_paper_full4890_20261014_run4。'])
    with (dest/'五折各模型完整避险误差报告.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(texts)+'\n')
    write_json(dest/'独立复核与全量覆盖.json',dict(passed=True,N=4890,folds=folds,models=PUBLIC,rows=rows,global_results=global_rows,paired=comparisons,
        all_exact_endpoints_verified=True,all_reference_coordinates_replayed=True,training_and_validation_separate=True,model_parameters_not_updated=True))

if __name__=='__main__':main()
