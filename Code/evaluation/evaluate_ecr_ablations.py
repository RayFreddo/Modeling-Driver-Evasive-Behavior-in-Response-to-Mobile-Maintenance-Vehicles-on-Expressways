"""Same frozen HGB-ECR teachers, controlled policies, complete event scoring."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import json,contextlib,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from contracts import ROOT,DAYS,FOLD_ROOT,sha,write_json,write_csv,npz_new
from train_optimized import RUN,OUT
from prepare_optimized import read_optimized
from optimized_features import anchor_basis
from fast_rollout import generate_batch
from full_training_check import references,evaluate_generated

class Policy:
    def __init__(self,parent,mode):self.parent=parent;self.mode=mode;self.training_days=parent.training_days
    def predict_arrays(self,x):
        if self.mode=='cv':return anchor_basis(x)[0]
        if self.mode=='specialist':
            _,_,p,s=self.parent.upstream.predict(x);p=p.copy();p[:,-1]=s;return p
        return self.parent.predict_with_policy(x,self.mode)[0]

def evaluate(held):
    gate=json.loads((OUT/'reviewed_complete_training_gate.json').read_text(encoding='utf-8'));assert gate['outer_evaluation_allowed']
    folder=FOLD_ROOT/f'holdout_{held}'/'02_模型训练'/RUN/'ECR-enabled-comparison'
    proof=json.loads((folder/'frozen_ecr.json').read_text(encoding='utf-8'));path=folder/'model.joblib';expected=proof['model_sha256'];assert sha(path)==expected
    parent=joblib.load(path);assert set(parent.training_days)==set(DAYS)-{held}
    samples,z,_=read_optimized(held);refs,manifest=references([held])
    dest=FOLD_ROOT/f'holdout_{held}'/'03_验证评价'/RUN/'ECR_受控消融';dest.mkdir(parents=True,exist_ok=False);summary=[]
    for name,mode in [('CV-anchor','cv'),('ECR-same-base-residual','residual'),('ECR-always-correct','always'),('ECR-specialist-endpoint','specialist')]:
        model=Policy(parent,mode);first=model.predict_arrays(z['x']);generated=generate_batch(model,samples,first,blocks=40)
        frame,metrics,points=evaluate_generated(generated,refs,manifest);child=dest/name;child.mkdir()
        frame['model_predicted_completed']=np.isfinite(generated['predicted_completion_offset_s'])
        metrics.update(held=held,model=name,mode=mode,model_predicted_completed=int(frame.model_predicted_completed.sum()),validation=True)
        write_csv(child/'per_object_metrics.csv',frame);write_json(child/'validation_metrics.json',metrics);npz_new(child/'scored_points.npz',**points)
        npz_new(child/'full_prediction_buffer.npz',positions=generated['positions'],offsets_s=generated['offsets_s'],keys=generated['keys'],predicted_completion_offset_s=generated['predicted_completion_offset_s'],initial_origin_s=generated['initial_origin_s'])
        write_json(child/'identity.json',dict(held=held,mode=mode,N=len(samples),training_days=parent.training_days,same_frozen_parent_model_sha256=expected,no_refit=True,
            comparisons='residual/always/full differ only in the ECR policy; specialist keeps the same five earlier horizons; CV equals the internal anchor',
            artifact_sha256={n:sha(child/n) for n in ['scored_points.npz','full_prediction_buffer.npz','per_object_metrics.csv','validation_metrics.json']}))
        summary.append(metrics);print('CONTROLLED COMPLETE-TRAJECTORY ABLATION',held,name,metrics,flush=True)
    assert sha(path)==expected;write_json(dest/'ablation_summary.json',summary)
    write_json(dest/'ablation_complete.json',dict(passed=True,held=held,N=len(samples),policies=len(summary),model_parameters_unchanged=True))

def worker(held):
    with (OUT/'logs'/f'{held}_ecr_ablations.log').open('x',encoding='utf-8',buffering=1) as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream),threadpool_limits(limits=2):
        try:evaluate(held);return dict(held=held,passed=True)
        except Exception:traceback.print_exc();raise

def main():
    results=[];errors=[]
    with ProcessPoolExecutor(max_workers=1) as pool:
        jobs={pool.submit(worker,d):d for d in DAYS}
        for future in as_completed(jobs):
            try:results.append(future.result());print('ECR ABLATION COMPLETE',jobs[future],flush=True)
            except Exception as e:errors.append(dict(held=jobs[future],error=str(e)));print('ECR ABLATION FAILED',jobs[future],str(e),flush=True)
    write_json(OUT/'all_controlled_ablations_complete.json',dict(passed=not errors,results=results,errors=errors,parameters_not_updated=True))
    if errors:raise RuntimeError('Ablation failure logs retained')

if __name__=='__main__':main()
