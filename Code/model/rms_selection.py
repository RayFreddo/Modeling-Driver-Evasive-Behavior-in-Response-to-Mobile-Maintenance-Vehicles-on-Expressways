"""Select RMS improvement subject to non-worsening internal ADE and FDE."""
import os,sys
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Code/revision_step4_longhead_20261014_run11'))
import longhead as h
import argparse,json
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from contracts import sha,write_json,write_csv,npz_new
OUT=ROOT/'outputs/step4_matched_teacher_20261014_run13'
BETAS=(0.,.25,.5,.75,1.)

def init():
    OUT.mkdir(exist_ok=False)
    write_json(OUT/'registered_plan.json',dict(betas=BETAS,
        core_model_folder=str(h.OUT),core_code_sha256=sha(Path(h.__file__)),code_sha256=sha(Path(__file__)),
        selection='min internal date-equal RMSE among candidates with internal ADE and FDE no greater than beta0; FDE then beta break ties',
        reason='all-metric max relative improvement over-prioritizes slight ADE improvements while the user RMSE criterion remains most difficult',
        original_weights_reused_no_refit=True,outer_labels_not_used_for_selection=True,
        user_acceptance='0409 each metric <= 1.5 * other four arithmetic means, and all three overall best vs original baselines',
        development_re_evaluation=True))

def raw(outer,day,base):
    samples,z,_=h.s.load_day(day);policy=h.recipe(outer)
    generated=h.l.damp(h.g.generate(base,samples,z['x'],policy['fallback_method'],0.),policy['lateral_tau_s'],samples)
    return samples,z,generated

def select(outer):
    dest=OUT/'selection'/outer;dest.mkdir(parents=True,exist_ok=False);rows=[];td=sorted(set(h.s.DAYS)-{outer})
    for inner in td:
        scope=sorted(set(td)-{inner});base,_=h.base_for(outer,scope)
        path=h.OUT/'selection'/outer/f'inner_{inner}_fit/model.joblib';proof=json.loads(path.with_name('identity.json').read_text(encoding='utf-8'))
        assert set(proof['training_days'])==set(scope) and sha(path)==proof['model_sha256']
        model=joblib.load(path);samples,z,generated=raw(outer,inner,base);refs,manifest=h.s.references([inner])
        full=h.corrected(model,generated,z['x'],1.,samples)
        for beta in BETAS:
            result=dict(generated);result['positions']=generated['positions']+beta*(full['positions']-generated['positions'])
            frame,metrics,_=h.s.evaluate_generated(result,refs,manifest);metrics.update(inner=inner,beta=beta);rows.append(metrics)
            write_csv(dest/f'{inner}_beta{beta:g}.csv',frame)
        print('RMS INNER COMPLETE',outer,inner,flush=True)
    table=pd.DataFrame(rows);write_csv(dest/'all_inner_metrics.csv',table)
    means=table.groupby('beta')[['ADE_m','FDE_m','RMSE_m']].mean().reset_index();reference=means[means.beta==0].iloc[0]
    means['internal_ADE_nonworse']=means.ADE_m<=reference.ADE_m+1e-12
    means['internal_FDE_nonworse']=means.FDE_m<=reference.FDE_m+1e-12
    candidates=means[means.internal_ADE_nonworse&means.internal_FDE_nonworse]
    best=candidates.sort_values(['RMSE_m','FDE_m','beta']).iloc[0]
    write_csv(dest/'candidate_means.csv',means)
    write_json(dest/'frozen_selection.json',dict(outer=outer,training_days=td,beta=float(best.beta),
        code_sha256=sha(Path(__file__)),core_code_sha256=sha(Path(h.__file__)),no_outer_metrics_for_selection=True))
    print('RMS FROZEN',outer,float(best.beta),flush=True)

def evaluate(outer,training=False):
    selection=json.loads((OUT/'selection'/outer/'frozen_selection.json').read_text(encoding='utf-8'))
    assert sha(Path(__file__))==selection['code_sha256'] and sha(Path(h.__file__))==selection['core_code_sha256']
    parent,_=h.base_for(outer,sorted(set(h.s.DAYS)-{outer}));path=h.OUT/'models'/outer/'model.joblib';model=joblib.load(path)
    proof=json.loads(path.with_name('identity.json').read_text(encoding='utf-8'))
    assert set(proof['training_days'])==set(h.s.DAYS)-{outer} and sha(path)==proof['model_sha256']
    days=selection['training_days'] if training else [outer];samples=[];zs=[];generations=[]
    for day in days:
        ss,z,gg=raw(outer,day,parent);samples.extend(ss);zs.append(z['x']);generations.append(gg)
    x=np.concatenate(zs);generated=dict(generations[0])
    for name in ['positions','keys','failed_block','predicted_completion_offset_s','initial_origin_s']:generated[name]=np.concatenate([gg[name] for gg in generations])
    result=h.corrected(model,generated,x,selection['beta'],samples);refs,manifest=h.s.references(days)
    frame,metrics,points=h.s.evaluate_generated(result,refs,manifest)
    dest=OUT/('training' if training else 'validation')/outer;dest.mkdir(parents=True,exist_ok=False)
    write_csv(dest/'per_object_metrics.csv',frame);npz_new(dest/'scored_points.npz',**points)
    npz_new(dest/'full_prediction_buffer.npz',**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
    metrics.update(held=outer,model='HGB matched-teacher correction',training=training,beta=selection['beta']);write_json(dest/'metrics.json',metrics)
    print('RMS RESULT',outer,training,json.dumps(metrics),flush=True)

def finish():
    dest=OUT/'报告';dest.mkdir(exist_ok=False);rows=[];checks=[]
    for day in h.s.DAYS:
        for phase in ['training','validation']:
            _,metrics,errors=h.report.recompute(OUT/phase/day,True);assert metrics['successful']==metrics['attempted']
            checks.append(dict(held=day,phase=phase,errors=errors))
            if phase=='validation':rows.append(dict(held=day,model='HGB matched-teacher correction',**metrics))
    table=pd.DataFrame(rows);other=table[table.held!='0409'];d9=table[table.held=='0409'].iloc[0]
    ratios={m:float(d9[m]/other[m].mean()) for m in ['ADE_m','FDE_m','RMSE_m']}
    old=pd.read_csv(h.s.OUT/'baseline_audit/frozen_baseline_metrics.csv',dtype={'held':str},float_precision='round_trip');old.held=old.held.str.zfill(4)
    combined=pd.concat([old,table],ignore_index=True);overall=[]
    for name,part in combined.groupby('model'):
        overall.append(dict(model=name,ADE_m=float(part.ADE_m.mean()),FDE_m=float(part.FDE_m.mean()),RMSE_m=float(np.sqrt((part.RMSE_m**2).mean())),successful=int(part.successful.sum()),attempted=int(part.attempted.sum())))
    overall=pd.DataFrame(overall).sort_values('FDE_m');main=overall[overall.model=='HGB matched-teacher correction'].iloc[0];base=overall[~overall.model.isin(['HGB-ECR','HGB matched-teacher correction'])]
    best={m:bool(main[m]<base[m].min()) for m in ratios}
    status=dict(ratios=ratios,criterion_1_5_passed=all(v<=1.5 for v in ratios.values()),best_by_metric=best,all_three_best=all(best.values()),
        independent_recomputations=checks,day0409={m:float(d9[m]) for m in ratios})
    write_json(dest/'验收结果.json',status);write_csv(dest/'五日验证.csv',combined);write_csv(dest/'总体比较.csv',overall)
    text=f'''## 33. 对齐训练外教师与部署底座的纠错架构

复用第31节平方误差HGB纠错头，第32节训练内修正强度。纠错头训练使用未启用旧ECR门控的HGB教师，本版部署也使用同类型四日期HGB底座，移除旧门控以避免教师/部署输入不一致。这是明确的架构更改，不能称作原条件门控ECR。选参只使用训练四日期的留一预测，要求候选内层平均ADE和FDE都不超过不修正基线，再选RMSE最小者。外层0409成绩不选参，所有日期同一算法。原对照和标签/计分协议保持，架构是新增长时域监督头，不能混写成原run4模型。

{h.report.markdown_table(table)}

{h.report.markdown_table(overall)}

0409三项与其他四日期平均值的比值{ratios}；1.5倍通过={status['criterion_1_5_passed']}；三项总体最优={status['all_three_best']}。两者同时通过才达标。全部训练与验证坐标独立复算；本轮是已经见过外层日期后的开发重新评价。对照没有被削弱。新增全时域监督使主模型的训练目标发生变化，结论限定为对原冻结对照的比较，正式论文应说明这一差异。
'''
    (dest/'HGB_RMSE选择验收报告.md').write_text('# HGB_RMSE选择与明确验收\n\n'+text,encoding='utf-8')
    previous=(ROOT/'outputs/step4_rms_selection_20261014_run12/报告/code修改意见_RMSE选择同步副本.md').read_text(encoding='utf-8')
    (dest/'code修改意见_RMSE选择同步副本.md').write_text('> 最新试验见第33节；原报告保持。\n\n'+previous+'\n\n'+text,encoding='utf-8')
    print('RMS REPORT',status,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['init','select','training','evaluate','finish']);p.add_argument('--held',choices=h.s.DAYS);a=p.parse_args()
    with threadpool_limits(limits=2):
        if a.action=='init':init()
        elif a.action=='select':select(a.held)
        elif a.action=='finish':finish()
        else:evaluate(a.held,a.action=='training')
