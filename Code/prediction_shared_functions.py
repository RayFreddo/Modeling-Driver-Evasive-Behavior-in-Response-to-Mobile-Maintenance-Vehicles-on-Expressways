"""六模型整条避险轨迹累计预测的共享实现。

这里集中放数据加载、特征工程、六类预测器、滚动预测、评估和绘图逻辑。
批量入口见 ``预测_主程序.py``。
"""

from __future__ import annotations

import importlib.util, json, math, re, sys, time, types, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(r"C:\Users\Administrator\Desktop")
SRC_DIR = BASE / "毕业论文"
DATA_DIR = SRC_DIR / "数据处理"
MODEL_ROOT = SRC_DIR / "多模型预测结果"
SCRIPT_DIR = MODEL_ROOT / "复现实验脚本"
OUT_DIR = BASE / "毕业论文修改输出"
FORMAL_DATA_DIR = OUT_DIR / "00_结果数据" / "正式完整轨迹结果"
FIG_DIR = OUT_DIR / "01_图表" / "01_整条轨迹指标图"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FORMAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = FORMAL_DATA_DIR / "单模型预测缓存"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILES = [DATA_DIR / f"{d}_断点插值补全.xlsx" for d in ["0403","0407","0410","0416"]]
FULL_0409 = DATA_DIR / "0409_断点插值补全.xlsx"
FEATURE_PARAM_FILE = SRC_DIR / "特征分析参数" / "trajectory_priors_0403_0407_0410_0416.json"
HGB_ARTIFACT = MODEL_ROOT / "HGB-ECR" / "HGB-ECR_artifacts.joblib"
HGB_FIXED_PRED = MODEL_ROOT / "HGB-ECR" / "HGB-ECR_0409_prediction.csv"
SCENE_STATS_XLSX = OUT_DIR / "00_结果数据" / "重跑统计结果.xlsx"

MAIN_COLS = ["object_id","segment_id","frame","timestamp_s","dt_s_fixed","distance_m","x_m","y_m",
    "rel_velocity_mps","follower_speed_mps","follower_acc_mps2","ttc_s","lane_index",
    "lat_speed_mps","lat_acc_mps2","lateral_shift_total_m","lateral_shift_abs_total_m",
    "final_lane_changed","final_significant_lateral_shift","trajectory_lanechange_evidence","model_min_length_flag"]
HGB_HORIZONS = np.array([0.5,1.0,1.5,2.0,2.5,3.0], dtype=float)
DT_DENSE = 0.05; MAX_TOTAL_DURATION_S = 10.0; MIN_AVOIDANCE_AFTER_INPUT_S = 1.0
LANE_END_RATIO = 0.94; LANE_END_STABLE_WINDOW = 5

MODEL_ORDER = ["HGB-ECR","Kalman-CV","Helly","IDM","GBR","CNN"]
MODEL_LABELS = {"HGB-ECR":"HGB-ECR","Kalman-CV":"Kalman-CV","Helly":"Helly线性跟驰","IDM":"IDM避险状态机","GBR":"GBR一步递推","CNN":"CNN一维卷积"}
COLORS = {"HGB-ECR":"#E45756","Kalman-CV":"#4C78A8","Helly":"#72B7B2","IDM":"#54A24B","GBR":"#F58518","CNN":"#B279A2"}

@dataclass
class Predictor:
    model_id: str; model_label: str
    fn: Callable[[pd.DataFrame,float,pd.DataFrame,str], pd.DataFrame]

def now_msg(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def clean_id(v):
    try:
        x=float(v)
        if abs(x-round(x))<1e-9: return str(int(round(x)))
    except: pass
    return str(v)

def load_module_from_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError(f"Cannot load: {path}")
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod

def load_patched_model_module(name,path):
    text=path.read_text(encoding="utf-8"); text=re.sub(r"^SMOOTH\s*=\s*load_smoothing_module\(\)\s*$","SMOOTH = None",text,flags=re.M)
    mod=types.ModuleType(name); mod.__file__=str(path); sys.modules[name]=mod
    exec(compile(text,str(path),"exec"),mod.__dict__)
    mod.TRAIN_FILES=TRAIN_FILES; mod.PREDICT_FILE=FULL_0409; mod.REFERENCE_HISTORY_FILE=FULL_0409; mod.FEATURE_PARAM_FILE=FEATURE_PARAM_FILE
    if hasattr(mod,"_FEATURE_PARAMS"): mod._FEATURE_PARAMS=None
    return mod

def read_0409_full():
    df=pd.read_excel(FULL_0409,sheet_name="Sheet1",usecols=lambda c:c in MAIN_COLS)
    if "segment_id" not in df.columns: df["segment_id"]=1
    for c in MAIN_COLS:
        if c in {"object_id","segment_id"} or c not in df.columns: continue
        df[c]=pd.to_numeric(df[c],errors="coerce")
    df["object_clean"]=df["object_id"].map(clean_id); df["segment_clean"]=df["segment_id"].map(clean_id)
    df["sample_key"]=df["object_clean"]+"__seg"+df["segment_clean"]
    return df.sort_values(["object_clean","segment_clean","timestamp_s"]).reset_index(drop=True)

def validation_keys_and_t0():
    pred=pd.read_csv(HGB_FIXED_PRED,usecols=["sample_key","object_clean","segment_clean","source","timestamp_s"],low_memory=False)
    obs=pred[pred["source"].eq("observed")].copy()
    t0=obs.groupby(["sample_key","object_clean","segment_clean"],as_index=False)["timestamp_s"].max()
    return t0.rename(columns={"timestamp_s":"t0"})

def load_scene_map():
    if not SCENE_STATS_XLSX.exists(): return pd.DataFrame()
    detail=pd.read_excel(SCENE_STATS_XLSX,sheet_name="五天动作与场景明细")
    detail=detail[pd.to_numeric(detail["day"],errors="coerce").eq(409)].copy()
    if detail.empty: return pd.DataFrame()
    detail["object_clean"]=detail["object_clean"].map(clean_id); detail["sample_key"]=detail["object_clean"]+"__seg1"
    keep=["sample_key","scene_type","action","avoidance_duration_s"]
    return detail[[c for c in keep if c in detail.columns]].drop_duplicates("sample_key")

def strict_avoidance_end_time(group,t0):
    g=group.sort_values("timestamp_s").dropna(subset=["timestamp_s","x_m","y_m"]).reset_index(drop=True)
    if g.empty: return float(t0)
    times=pd.to_numeric(g["timestamp_s"],errors="coerce").to_numpy(float)
    x=pd.to_numeric(g["x_m"],errors="coerce").to_numpy(float)
    acc=pd.to_numeric(g.get("follower_acc_mps2",pd.Series(np.nan,index=g.index)),errors="coerce").to_numpy(float)
    finite=np.isfinite(times)&np.isfinite(x)
    if finite.sum()<3: return float(min(np.nanmax(times),t0+MAX_TOTAL_DURATION_S))
    times,x,acc=times[finite],x[finite],acc[finite]
    start_idx=max(0,min(int(np.searchsorted(times,t0,side="left")),len(times)-1))
    x0=float(np.interp(t0,times,x)); shift=np.abs(x-x0)
    final_shift=float(np.nanmax(shift[start_idx:])) if np.isfinite(shift[start_idx:]).any() else 0.0
    candidates=[]
    if final_shift>=0.8:
        threshold=max(0.8,LANE_END_RATIO*final_shift)
        stable_lat=np.abs(np.gradient(x,times))<=0.35
        for idx in np.flatnonzero((np.arange(len(times))>=start_idx)&(shift>=threshold)):
            stop=min(idx+LANE_END_STABLE_WINDOW,len(times))
            if stop-idx<2 or bool(np.nanmean(stable_lat[idx:stop])>=0.6): candidates.append(int(idx)); break
    decel=np.isfinite(acc)&(acc<=-1.2)
    if decel[start_idx:].any():
        decel_idx=int(np.flatnonzero(decel&(np.arange(len(times))>=start_idx))[0])
        for idx in range(decel_idx+3,len(times)):
            recent_acc=acc[idx:min(idx+5,len(times))]
            if len(recent_acc)>=3 and np.isfinite(recent_acc).any() and np.nanmean(recent_acc>-0.45)>=0.8: candidates.append(idx); break
    cap=min(float(times[-1]),float(t0+MAX_TOTAL_DURATION_S)); min_end=float(t0+MIN_AVOIDANCE_AFTER_INPUT_S)
    return float(min(max(float(times[min(candidates)]),min_end),cap)) if candidates else float(min(max(float(times[-1]),min_end),cap))

def interp_arr(t,arr_t,arr_v):
    mask=np.isfinite(arr_t)&np.isfinite(arr_v); tt=arr_t[mask]; vv=arr_v[mask]
    if len(tt)<2: return float("nan")
    order=np.argsort(tt); tt=tt[order]; vv=vv[order]
    if t<tt[0]-1e-6 or t>tt[-1]+1e-6: return float("nan")
    return float(np.interp(t,tt,vv))

def numeric_history(hist):
    data=hist[["timestamp_s","x_m","y_m"]].apply(pd.to_numeric,errors="coerce").dropna().sort_values("timestamp_s")
    return data["timestamp_s"].to_numpy(float), data["x_m"].to_numpy(float), data["y_m"].to_numpy(float)

def recent_mask(t,window_s,min_points):
    if len(t)<=min_points: return np.ones(len(t),dtype=bool)
    mask=t>=t[-1]-window_s; return mask if int(mask.sum())>=min_points else np.ones(len(t),dtype=bool)

def line_velocity(t,values,window_s=0.8):
    if len(t)<2: return 0.0
    mask=recent_mask(t,window_s,4); rel=t[mask]-t[-1]; vals=values[mask]
    if len(vals)<2 or np.ptp(rel)<1e-6: dt=max(float(t[-1]-t[-2]),1e-6); return float((values[-1]-values[-2])/dt)
    return float(np.polyfit(rel,vals,1)[0])

def clipped_xy(x0,y0,vx,vy,dt,ax=0.0,ay=0.0):
    vx=float(np.clip(vx,-3.0,3.0)); vy=float(np.clip(vy,-22.0,22.0))
    ax=float(np.clip(ax,-4.0,4.0)); ay=float(np.clip(ay,-8.0,8.0))
    return x0+vx*dt+0.5*ax*dt*dt, y0+vy*dt+0.5*ay*dt*dt

def synthetic_vars(times,xs,ys,prev_lane):
    order=np.argsort(times); times,xs,ys=times[order],xs[order],ys[order]
    if len(times)<2: times=np.array([times[0]-DT_DENSE,times[0]],dtype=float); xs=np.array([xs[0],xs[0]],dtype=float); ys=np.array([ys[0],ys[0]],dtype=float)
    dist=np.hypot(xs,ys); dxdt=np.gradient(xs,times); dydt=np.gradient(ys,times); speed=np.hypot(dxdt,dydt)
    acc=np.gradient(speed,times); rel=np.gradient(dist,times); closing=np.maximum(-rel,0.0)
    ttc=np.where(closing>0.05,dist/np.maximum(closing,0.05),30.0); ttc=np.clip(ttc,0.2,30.0); lat_acc=np.gradient(dxdt,times)
    return pd.DataFrame({"timestamp_s":times,"x_m":xs,"y_m":ys,"distance_m":dist,
        "rel_velocity_mps":np.clip(rel,-22,22),"follower_speed_mps":np.clip(speed,0,40),
        "follower_acc_mps2":np.clip(acc,-8,5),"ttc_s":ttc,"lane_index":prev_lane,
        "lat_speed_mps":np.clip(dxdt,-5,5),"lat_acc_mps2":np.clip(lat_acc,-5,5)})

def with_required_cols(rows,template,model_id,step):
    out=rows.copy(); last=template.sort_values("timestamp_s").iloc[-1]
    for col in MAIN_COLS:
        if col not in out.columns: out[col]=np.nan
    out["object_id"]=last["object_id"]; out["segment_id"]=last["segment_id"]
    if "frame" not in out.columns or out["frame"].isna().all():
        frame0=pd.to_numeric(pd.Series([last.get("frame")]),errors="coerce").iloc[0]
        if np.isfinite(frame0): out["frame"]=frame0+np.arange(1,len(out)+1)
    out["dt_s_fixed"]=out["timestamp_s"].diff().fillna(DT_DENSE); out["source"]="predicted"
    out["model_id"]=model_id; out["step"]=step; return out

def synthesize_from_xy(hist,pred_xy,model_id,step):
    hist=hist.sort_values("timestamp_s").copy(); pred=pred_xy[["timestamp_s","x_m","y_m"]].copy()
    last_lane=pd.to_numeric(hist["lane_index"],errors="coerce").dropna(); lane=float(last_lane.iloc[-1]) if len(last_lane) else 0.0
    context=hist[["timestamp_s","x_m","y_m"]].tail(8)
    tmp=pd.concat([context,pred],ignore_index=True).drop_duplicates("timestamp_s").sort_values("timestamp_s")
    vars_df=synthetic_vars(tmp["timestamp_s"].to_numpy(float),tmp["x_m"].to_numpy(float),tmp["y_m"].to_numpy(float),lane)
    out=vars_df[vars_df["timestamp_s"].isin(pred["timestamp_s"])].copy()
    return with_required_cols(out,hist,model_id,step)

def eval_times(t0,t_end):
    if t_end<=t0+0.25: return np.array([],dtype=float)
    times=list(np.arange(t0+0.5,t_end+1e-6,0.5))
    if not times or abs(times[-1]-t_end)>0.15: times.append(float(t_end))
    return np.asarray(sorted(set(round(float(t),6) for t in times)),dtype=float)

def interpolate_path(path,times):
    p=path.sort_values("timestamp_s").drop_duplicates("timestamp_s")
    tt=pd.to_numeric(p["timestamp_s"],errors="coerce").to_numpy(float)
    xs=pd.to_numeric(p["x_m"],errors="coerce").to_numpy(float); ys=pd.to_numeric(p["y_m"],errors="coerce").to_numpy(float)
    mask=np.isfinite(tt)&np.isfinite(xs)&np.isfinite(ys); tt,xs,ys=tt[mask],xs[mask],ys[mask]
    if len(tt)<2: return np.full((len(times),2),np.nan)
    order=np.argsort(tt); tt,xs,ys=tt[order],xs[order],ys[order]
    return np.column_stack([np.interp(times,tt,xs),np.interp(times,tt,ys)])

def actual_xy(group,times):
    g=group.sort_values("timestamp_s").drop_duplicates("timestamp_s")
    tt=pd.to_numeric(g["timestamp_s"],errors="coerce").to_numpy(float)
    xs=pd.to_numeric(g["x_m"],errors="coerce").to_numpy(float); ys=pd.to_numeric(g["y_m"],errors="coerce").to_numpy(float)
    return np.column_stack([np.interp(times,tt,xs),np.interp(times,tt,ys)])

def trim_segment(seg,t_cur,t_target):
    if seg.empty: return seg
    out=seg.sort_values("timestamp_s").drop_duplicates("timestamp_s",keep="last").copy()
    out["timestamp_s"]=pd.to_numeric(out["timestamp_s"],errors="coerce"); out=out[out["timestamp_s"]>t_cur+1e-6].copy()
    inside=out[out["timestamp_s"]<=t_target+1e-6].copy(); after=out[out["timestamp_s"]>t_target+1e-6].head(1).copy()
    return pd.concat([inside,after],ignore_index=True,sort=False)

def extend_segment_to_target(seg,hist,target,model_id,step):
    if seg.empty: return seg,"empty_prediction"
    seg=seg.sort_values("timestamp_s").drop_duplicates("timestamp_s",keep="last").copy()
    last_t=float(pd.to_numeric(seg["timestamp_s"],errors="coerce").max())
    if last_t>=target-DT_DENSE: return seg,"ok"
    basis=pd.concat([hist,seg],ignore_index=True,sort=False).sort_values("timestamp_s")
    t,x,y=numeric_history(basis)
    if len(t)<2: return seg,"short_prediction_not_extended"
    vx=float(np.clip(line_velocity(t,x,window_s=0.6),-3.0,3.0)); vy=float(np.clip(line_velocity(t,y,window_s=0.6),-22.0,22.0))
    x0=float(x[-1]); y0=float(y[-1])
    times=list(np.arange(last_t+DT_DENSE,target+1e-9,DT_DENSE))
    if not times or abs(times[-1]-target)>0.015: times.append(float(target))
    times_arr=np.asarray(sorted(set(round(float(ti),6) for ti in times)),dtype=float); rel=times_arr-last_t
    ext_xy=pd.DataFrame({"timestamp_s":times_arr,"x_m":x0+vx*rel,"y_m":y0+vy*rel})
    ext=synthesize_from_xy(basis,ext_xy,model_id,step); ext["source"]="predicted_extension"
    out=pd.concat([seg,ext],ignore_index=True,sort=False).sort_values("timestamp_s")
    return out.drop_duplicates("timestamp_s",keep="last"),"extended_to_target"

def rolling_open_loop(predictor,group,t0,sample_key):
    g=group.sort_values("timestamp_s").drop_duplicates("timestamp_s").reset_index(drop=True)
    t_end=float(g["timestamp_s"].iloc[-1]); hist=g[g["timestamp_s"]<=t0+1e-6].copy(); t_cur=float(t0)
    parts=[]; step_rows=[]; max_steps=int(math.ceil(max(t_end-t0,0.0)/3.0))+2
    for step in range(1,max_steps+1):
        if t_cur>=t_end-0.025: break
        target=min(t_cur+3.0,t_end); seg=predictor.fn(hist,t_cur,group,sample_key)
        seg=trim_segment(seg,t_cur,target)
        if seg.empty: step_rows.append({"model_id":predictor.model_id,"sample_key":sample_key,"step":step,"t0":t_cur,"target_t":target,"status":"empty_prediction"}); break
        seg,status=extend_segment_to_target(seg,hist,target,predictor.model_id,step)
        seg=seg.copy(); seg["model_id"]=predictor.model_id; seg["model_label"]=predictor.model_label; seg["sample_key"]=sample_key; seg["step"]=step
        parts.append(seg)
        hist=pd.concat([hist,seg[[c for c in hist.columns if c in seg.columns]]],ignore_index=True,sort=False).drop_duplicates("timestamp_s",keep="last").sort_values("timestamp_s").reset_index(drop=True)
        step_rows.append({"model_id":predictor.model_id,"sample_key":sample_key,"step":step,"t0":t_cur,"target_t":target,"predicted_until_s":float(seg["timestamp_s"].max()),"rows":int(len(seg)),"status":status})
        t_cur=target
    pred=pd.concat(parts,ignore_index=True,sort=False) if parts else pd.DataFrame(columns=list(g.columns)+["model_id","model_label","sample_key","step"])
    anchor=hist[hist["timestamp_s"].le(t0+1e-6)].tail(1).copy()
    anchor["model_id"]=predictor.model_id; anchor["model_label"]=predictor.model_label; anchor["sample_key"]=sample_key; anchor["step"]=0; anchor["source"]="observed_anchor"
    path=pd.concat([anchor,pred],ignore_index=True,sort=False)
    return path.sort_values("timestamp_s").drop_duplicates("timestamp_s",keep="last"), step_rows

def summarize_model_metrics(point_df,object_df):
    rows=[]
    for model_id,g in point_df.groupby("model_id",sort=False):
        obj=object_df[object_df["model_id"].eq(model_id)]; label=obj["model_label"].iloc[0] if not obj.empty else model_id
        errors=pd.to_numeric(g["error_m"],errors="coerce").dropna().to_numpy(float)
        end_errors=pd.to_numeric(obj["trajectory_end_error_m"],errors="coerce").dropna().to_numpy(float)
        if len(errors)==0 or len(end_errors)==0: continue
        rows.append({"model_id":model_id,"model_label":label,"N":int(obj["sample_key"].nunique()),"point_count":int(len(errors)),
            "ADE_m":float(np.nanmean(errors)),"FDE_m":float(np.nanmean(end_errors)),"RMSE_m":float(np.sqrt(np.nanmean(errors**2))),
            "FDE_median_m":float(np.nanmedian(end_errors)),"FDE_q90_m":float(np.nanpercentile(end_errors,90))})
    out=pd.DataFrame(rows)
    if not out.empty: out=out.sort_values(["ADE_m","RMSE_m"]).reset_index(drop=True)
    return out

def set_balanced_xy_limits(ax,xs,ys):
    xvals=np.concatenate([np.asarray(v,dtype=float) for v in xs if len(v)]); yvals=np.concatenate([np.asarray(v,dtype=float) for v in ys if len(v)])
    xvals=xvals[np.isfinite(xvals)]; yvals=yvals[np.isfinite(yvals)]
    if len(xvals)==0 or len(yvals)==0: return
    xr=max(float(np.nanmax(xvals)-np.nanmin(xvals)),1.2); yr=max(float(np.nanmax(yvals)-np.nanmin(yvals)),1.2)
    min_xr=max(3.5,yr*0.12)
    if xr<min_xr: x_mid=float(np.nanmean([np.nanmin(xvals),np.nanmax(xvals)])); xr=min_xr
    else: x_mid=float(np.nanmean([np.nanmin(xvals),np.nanmax(xvals)]))
    ax.set_xlim(x_mid-xr/2-xr*0.12,x_mid+xr/2+xr*0.12); ax.set_ylim(float(np.nanmin(yvals)-max(0.8,yr*0.05)),float(np.nanmax(yvals)+max(0.8,yr*0.05)))
    ax.set_aspect("auto")

# ====== 模型预测器构建函数 ======

def kalman_cv_state(t,x,y):
    if len(t)<2: return 0.0,0.0
    state=np.array([x[0],y[0],line_velocity(t[:2],x[:2]),line_velocity(t[:2],y[:2])],dtype=float)
    p=np.diag([0.5,0.5,10.0,10.0]); h=np.array([[1.,0,0,0],[0,1.,0,0]]); r=np.diag([0.08,0.08]); q=0.6; last_t=t[0]
    for ti,xi,yi in zip(t[1:],x[1:],y[1:]):
        dt=max(float(ti-last_t),1e-4)
        F=np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]); qb=np.array([[dt**4/4,dt**3/2],[dt**3/2,dt**2]])*q
        Q=np.zeros((4,4)); Q[np.ix_([0,2],[0,2])]=qb; Q[np.ix_([1,3],[1,3])]=qb
        state=F@state; p=F@p@F.T+Q; z=np.array([xi,yi]); S=h@p@h.T+r; K=p@h.T@np.linalg.inv(S); state=state+K@(z-h@state); p=(np.eye(4)-K@h)@p; last_t=ti
    return float(state[2]),float(state[3])

def build_motion_predictor():
    def predict(hist,t_cur,group,sample_key):
        t,x,y=numeric_history(hist)
        if len(t)<2: return pd.DataFrame()
        x0,y0=float(x[-1]),float(y[-1]); rel_times=np.arange(DT_DENSE,3.0+1e-9,DT_DENSE)
        vx,vy=kalman_cv_state(t,x,y); px,py=clipped_xy(x0,y0,vx,vy,rel_times)
        return synthesize_from_xy(hist,pd.DataFrame({"timestamp_s":t_cur+rel_times,"x_m":px,"y_m":py}),"Kalman-CV",0)
    return predict

def make_helly_predictor(module,model,centers):
    def predict(hist,t_cur,group,sample_key):
        out=module.predict_helly(model,hist.copy(),"Helly模型","Helly linear car-following model",centers)
        return with_required_cols(out,hist,"Helly",0) if not out.empty else out
    return predict

def make_idm_predictor(module,params,centers,width,context_groups):
    initial_context={}
    def predict(hist,t_cur,group,sample_key):
        hist=hist.copy()
        if sample_key not in initial_context:
            last=hist.sort_values("timestamp_s").iloc[-1]
            initial_context[sample_key]=module.context_row(context_groups,last["object_id"],last.get("frame",np.nan),last["timestamp_s"])
        out,_=module.predict_one("0409累计预测",hist,initial_context[sample_key],params,centers,width)
        return with_required_cols(out,hist,"IDM",0) if not out.empty else out
    return predict

def make_cv_blended_predictor(module,model,centers,model_id,model_name,method_note,blend_decay=0.18,min_model_weight=0.45):
    def predict(hist,t_cur,group,sample_key):
        out=module.predict_learned(model,hist.copy(),model_name,method_note,centers)
        if out.empty: return out
        out=with_required_cols(out,hist,model_id,0); out=out.sort_values("timestamp_s").reset_index(drop=True)
        t,x_arr,y_arr=numeric_history(hist)
        if len(t)<2: return out
        x0,y0=float(x_arr[-1]),float(y_arr[-1]); vx=float(np.clip(line_velocity(t,x_arr,window_s=0.6),-3.0,3.0)); vy=float(np.clip(line_velocity(t,y_arr,window_s=0.6),-22.0,22.0))
        rel_t=pd.to_numeric(out["timestamp_s"],errors="coerce").to_numpy(float)-t_cur; alpha=np.clip(1.0-blend_decay*rel_t,min_model_weight,1.0)
        cv_x=x0+vx*rel_t; cv_y=y0+vy*rel_t
        out["x_m"]=alpha*pd.to_numeric(out["x_m"],errors="coerce").to_numpy(float)+(1.0-alpha)*cv_x
        out["y_m"]=alpha*pd.to_numeric(out["y_m"],errors="coerce").to_numpy(float)+(1.0-alpha)*cv_y
        return out
    return predict

def make_hgb_predictor(train_hgb_mod,hgb_art):
    feature_cols=train_hgb_mod.FEATURE_COLS
    def group_to_arrs(group):
        g=group.sort_values("timestamp_s").drop_duplicates("timestamp_s").reset_index(drop=True)
        return {col:pd.to_numeric(g[col],errors="coerce").to_numpy(float) if col in g.columns else np.full(len(g),np.nan) for col in ["timestamp_s"]+feature_cols}
    def build_feature_and_base(arrs,t0):
        x0=interp_arr(t0,arrs["timestamp_s"],arrs["x_m"]); y0=interp_arr(t0,arrs["timestamp_s"],arrs["y_m"])
        if not(np.isfinite(x0) and np.isfinite(y0)): return None
        feats=[]; rel_times=np.linspace(-3.0,0.0,21)
        for rel in rel_times:
            x=interp_arr(t0+float(rel),arrs["timestamp_s"],arrs["x_m"]); y=interp_arr(t0+float(rel),arrs["timestamp_s"],arrs["y_m"])
            feats.extend([x-x0 if np.isfinite(x) else np.nan,y-y0 if np.isfinite(y) else np.nan])
        for col in feature_cols[2:]:
            vals=[interp_arr(t0+rel,arrs["timestamp_s"],arrs[col]) for rel in [-3,-2.5,-2,-1.5,-1,-0.5,0]]; feats.extend(vals)
            arr=np.asarray(vals,dtype=float); feats.extend([float(arr[np.isfinite(arr)][-1]) if np.isfinite(arr).any() else np.nan,float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan,float(np.nanstd(arr)) if np.isfinite(arr).any() else np.nan])
        for span in [0.25,0.5,1.0,1.5,2.0,3.0]:
            xa=interp_arr(t0-span,arrs["timestamp_s"],arrs["x_m"]); ya=interp_arr(t0-span,arrs["timestamp_s"],arrs["y_m"])
            feats.extend([(x0-xa)/span if np.isfinite(xa) else np.nan,(y0-ya)/span if np.isfinite(ya) else np.nan])
        for col,v0 in [("x_m",x0),("y_m",y0)]:
            tt,vv=[],[]
            for rel in rel_times:
                val=interp_arr(t0+float(rel),arrs["timestamp_s"],arrs[col])
                if np.isfinite(val): tt.append(float(rel)); vv.append(val-v0)
            feats.extend(list(map(float,np.polyfit(np.asarray(tt),np.asarray(vv),3))) if len(tt)>=4 else [np.nan]*4)
        xa1=interp_arr(t0-1.0,arrs["timestamp_s"],arrs["x_m"]); ya1=interp_arr(t0-1.0,arrs["timestamp_s"],arrs["y_m"])
        vx1=(x0-xa1) if np.isfinite(xa1) else 0.0; vy1=(y0-ya1) if np.isfinite(ya1) else 0.0
        xa05=interp_arr(t0-0.5,arrs["timestamp_s"],arrs["x_m"]); ya05=interp_arr(t0-0.5,arrs["timestamp_s"],arrs["y_m"])
        vx05=(x0-xa05)/0.5 if np.isfinite(xa05) else vx1; vy05=(y0-ya05)/0.5 if np.isfinite(ya05) else vy1
        vx=float(np.clip(0.35*vx05+0.65*vx1,-4.0,4.0)); vy=float(np.clip(0.35*vy05+0.65*vy1,-25.0,25.0))
        base_delta=[]; [base_delta.extend([vx*float(h),vy*float(h)]) for h in HGB_HORIZONS]
        return np.asarray(feats,dtype=float),np.asarray(base_delta,dtype=float),float(x0),float(y0)
    def predict(hist,t_cur,group,sample_key):
        arrs=group_to_arrs(hist); built=build_feature_and_base(arrs,t_cur)
        if built is None: return pd.DataFrame()
        x,base,x0,y0=built; x2=x.reshape(1,-1); base2=base.reshape(1,-1)
        base_hgb=train_hgb_mod.cap_displacements(base2+hgb_art["base_model"].predict(x2))
        specialist=hgb_art["endpoint_specialist"].predict(x2); feats=train_hgb_mod.endpoint_corrector_features(x2,base2,base_hgb,specialist)
        correction_sd=hgb_art["endpoint_corrector"].predict(feats)
        pred_base_error=np.clip(hgb_art["base_error_model"].predict(feats),0.0,None); pred_corrected_error=np.clip(hgb_art["corrected_error_model"].predict(feats),0.0,None)
        pred_delta,_,_=train_hgb_mod.apply_endpoint_correction(base_hgb,correction_sd,base2,pred_base_error,pred_corrected_error,hgb_art["policy"])
        rows=[]; delta=pred_delta[0]
        for i,horizon in enumerate(HGB_HORIZONS): rows.append({"timestamp_s":t_cur+float(horizon),"x_m":x0+float(delta[2*i]),"y_m":y0+float(delta[2*i+1])})
        result=synthesize_from_xy(hist,pd.DataFrame(rows),"HGB-ECR",0)
        # 长时域稳定：递归步骤轻度CV锚定
        hist_t=pd.to_numeric(hist["timestamp_s"],errors="coerce").dropna().to_numpy(float)
        if len(hist_t)>0 and t_cur-float(hist_t.min())>3.5 and len(result)>0:
            t_arr,x_arr,y_arr=numeric_history(hist)
            if len(t_arr)>=2:
                cv_vx=float(np.clip(line_velocity(t_arr,x_arr,window_s=0.6),-3.0,3.0)); cv_vy=float(np.clip(line_velocity(t_arr,y_arr,window_s=0.6),-22.0,22.0))
                result=result.sort_values("timestamp_s").reset_index(drop=True)
                rel_t=pd.to_numeric(result["timestamp_s"],errors="coerce").to_numpy(float)-t_cur; cv_weight=np.clip(0.05*rel_t+0.015*rel_t**2,0.0,0.50)
                cv_x=float(x_arr[-1])+cv_vx*rel_t; cv_y=float(y_arr[-1])+cv_vy*rel_t
                rx=pd.to_numeric(result["x_m"],errors="coerce").to_numpy(float); ry=pd.to_numeric(result["y_m"],errors="coerce").to_numpy(float)
                result["x_m"]=(1.0-cv_weight)*rx+cv_weight*cv_x; result["y_m"]=(1.0-cv_weight)*ry+cv_weight*cv_y
        return result
    return predict

# ====== 共用主流程 ======

def load_all(do_train=True):
    """加载数据、验证集和模型。返回 (full_0409, validation, actual_groups, models_dict, train_stats)。"""
    full_0409=read_0409_full(); validation=validation_keys_and_t0(); scene_map=load_scene_map()
    validation=validation.merge(scene_map,on="sample_key",how="left")
    groups={key:g.copy() for key,g in full_0409.groupby("sample_key",sort=False)}
    validation=validation[validation["sample_key"].isin(groups)].reset_index(drop=True)
    actual_groups={row.sample_key:groups[row.sample_key] for row in validation.itertuples(index=False)}
    now_msg(f"验证样本数: {len(validation)}")

    models={}; train_stats={}
    if do_train:
        now_msg("训练GBR/CNN/Helly...")
        gbr_mod=load_patched_model_module("gbr_mod",SCRIPT_DIR/"GBR_Model.py")
        cnn_mod=load_patched_model_module("cnn_mod",SCRIPT_DIR/"CNN_Model.py")
        helly_mod=load_patched_model_module("helly_mod",SCRIPT_DIR/"Helly_Model.py")
        train_frames=[gbr_mod.read_main_xlsx(p) for p in TRAIN_FILES]
        centers,_=gbr_mod.lane_centers(train_frames)
        x_train,y_train=gbr_mod.build_supervised_examples(train_frames)
        now_msg(f"监督训练样本: {len(x_train)}")
        gbr_model=gbr_mod.GradientBoostingWindowRegressor(); gbr_stats=gbr_model.fit(x_train,y_train)
        cnn_model=cnn_mod.NumpyConv1DRegressor(); cnn_stats=cnn_model.fit(x_train,y_train)
        helly_model=helly_mod.HellyModel(); helly_stats=helly_model.fit(train_frames)
        train_stats={"GBR":gbr_stats.val_rmse,"CNN":cnn_stats.val_rmse,"Helly":helly_stats.extra}
        now_msg(f"GBR RMSE: {gbr_stats.val_rmse}")
        now_msg(f"CNN RMSE: {cnn_stats.val_rmse}")
        now_msg(f"Helly: {helly_stats.extra}")

        idm_mod=load_module_from_file("idm_mod",SCRIPT_DIR/"IDM_Avoidance_Model.py")
        idm_mod.FEATURE_PARAM_FILE=FEATURE_PARAM_FILE
        if hasattr(idm_mod,"_FEATURE_PARAMS"): idm_mod._FEATURE_PARAMS=None
        idm_mod.MAX_HORIZON_S=3.0
        train_all_for_geometry=pd.concat(train_frames,ignore_index=True,sort=False)
        idm_params=idm_mod.load_params(); idm_contexts=idm_mod.load_context_groups(FULL_0409)
        idm_centers,idm_width=idm_mod.lane_centers(train_all_for_geometry)
        hgb_mod=load_module_from_file("hgb_mod",SCRIPT_DIR/"train_hgb_ecr.py"); hgb_art=joblib.load(HGB_ARTIFACT)

        models={
            "Kalman-CV":build_motion_predictor(),
            "Helly":make_helly_predictor(helly_mod,helly_model,centers),
            "IDM":make_idm_predictor(idm_mod,idm_params,idm_centers,idm_width,idm_contexts),
            "GBR":make_cv_blended_predictor(gbr_mod,gbr_model,centers,"GBR","GBR","GBR",blend_decay=0.18,min_model_weight=0.45),
            "CNN":make_cv_blended_predictor(cnn_mod,cnn_model,centers,"CNN","CNN","CNN",blend_decay=0.18,min_model_weight=0.45),
            "HGB-ECR":make_hgb_predictor(hgb_mod,hgb_art),
        }
    return full_0409,validation,actual_groups,models,train_stats

def run_one_model(model_id,model_label,predict_fn,validation,actual_groups):
    """运行单个模型的预测，返回 (paths_df, point_rows, object_rows, step_rows)。"""
    predictor=Predictor(model_id,model_label,predict_fn)
    all_paths=[]; point_rows=[]; object_rows=[]; step_rows=[]
    for ridx,row in enumerate(validation.itertuples(index=False),start=1):
        sample_key=row.sample_key; group=actual_groups[sample_key]; t0=float(row.t0)
        t_end=strict_avoidance_end_time(group,t0)
        group_eval=group[group["timestamp_s"].le(t_end+1e-6)].copy()
        if group_eval.empty or float(group_eval["timestamp_s"].max())<t_end-1e-6:
            group_eval=pd.concat([group_eval,group.iloc[(group["timestamp_s"]-t_end).abs().argsort()[:1]]],ignore_index=True,sort=False)
        times=eval_times(t0,t_end)
        if len(times)==0: continue
        path,steps=rolling_open_loop(predictor,group_eval,t0,sample_key)
        all_paths.append(path)
        for st in steps: step_rows.append(st)
        pred_xy=interpolate_path(path,times); true_xy=actual_xy(group_eval,times)
        errors=np.linalg.norm(pred_xy-true_xy,axis=1); valid=np.isfinite(errors)
        if not valid.any(): continue
        errors=errors[valid]; valid_times=times[valid]
        for tt,xy_p,xy_a,err in zip(valid_times,pred_xy[valid],true_xy[valid],errors):
            point_rows.append({"model_id":model_id,"model_label":model_label,"sample_key":sample_key,
                "object_clean":row.object_clean,"segment_clean":row.segment_clean,"scene_type":getattr(row,"scene_type",np.nan),"action":getattr(row,"action",np.nan),
                "timestamp_s":float(tt),"rel_time_s":float(tt-t0),"pred_x_m":float(xy_p[0]),"pred_y_m":float(xy_p[1]),"true_x_m":float(xy_a[0]),"true_y_m":float(xy_a[1]),"error_m":float(err)})
        object_rows.append({"model_id":model_id,"model_label":model_label,"sample_key":sample_key,
            "object_clean":row.object_clean,"segment_clean":row.segment_clean,"scene_type":getattr(row,"scene_type",np.nan),"action":getattr(row,"action",np.nan),
            "t0":t0,"t_end":t_end,"duration_after_input_s":float(t_end-t0),"eval_points":int(len(errors)),
            "ADE_m":float(np.nanmean(errors)),"RMSE_m":float(np.sqrt(np.nanmean(errors**2))),"trajectory_end_error_m":float(errors[-1]),
            "max_error_m":float(np.nanmax(errors)),"predicted_until_s":float(path["timestamp_s"].max()) if not path.empty else np.nan})
    paths_df=pd.concat(all_paths,ignore_index=True,sort=False) if all_paths else pd.DataFrame()
    return paths_df,pd.DataFrame(point_rows),pd.DataFrame(object_rows),pd.DataFrame(step_rows)

def plot_all_figures(point_df,object_df,paths_df,actual_groups,out_dir=None):
    """生成全部指标图。返回 figure_paths 列表。"""
    if out_dir is None: out_dir=FIG_DIR
    summary_df=summarize_model_metrics(point_df,object_df); paths=[]
    data=summary_df.sort_values("ADE_m").copy(); labels=data["model_label"].tolist(); x=np.arange(len(data))
    for col,ylabel,title,fname,color in [("ADE_m","ADE / m","Average Displacement Error","ADE","#4C78A8"),("FDE_m","FDE / m","Final Displacement Error","FDE","#E45756"),("RMSE_m","RMSE / m","Root Mean Square Error","RMSE","#59A14F")]:
        fig,ax=plt.subplots(figsize=(9.0,5.0),dpi=180); sd=data.sort_values(col).copy()
        ax.bar(np.arange(len(sd)),sd[col],color=color,alpha=0.85); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.set_xticks(np.arange(len(sd))); ax.set_xticklabels(sd["model_label"].tolist(),rotation=28,ha="right"); ax.grid(axis="y",linestyle="--",alpha=0.28)
        fig.tight_layout(); p=out_dir/f"Figure_{fname}.png"; fig.savefig(p); plt.close(fig); paths.append(p)
    # FDE 箱线图
    order=list(object_df.groupby("model_label")["trajectory_end_error_m"].median().sort_values().index)
    data_box=[object_df.loc[object_df["model_label"].eq(l),"trajectory_end_error_m"].dropna().clip(0,40).to_numpy() for l in order]
    fig,ax=plt.subplots(figsize=(9.0,5.0),dpi=180); ax.boxplot(data_box,labels=order,showfliers=False,patch_artist=True)
    ax.set_ylabel("FDE / m"); ax.set_title("Final Displacement Error Distribution"); ax.tick_params(axis="x",rotation=28); ax.grid(axis="y",linestyle="--",alpha=0.28)
    fig.tight_layout(); p=out_dir/"Figure_FDE_box.png"; fig.savefig(p); plt.close(fig); paths.append(p)
    # 误差时间曲线
    order2=list(object_df.groupby("model_id")["trajectory_end_error_m"].mean().sort_values().index)
    curves={}; timeline_rows=[]
    for mid in order2:
        g=point_df[point_df["model_id"].eq(mid)].copy()
        if g.empty: continue
        g["rel_time_bin_s"]=(g["rel_time_s"]*2).round()/2
        curve=g.groupby("rel_time_bin_s",as_index=False).agg(ADE_m=("error_m","mean"),
            RMSE_m=("error_m",lambda s:float(np.sqrt(np.nanmean(pd.to_numeric(s,errors="coerce")**2)))),
            FDE_m=("error_m",lambda s:float(pd.to_numeric(s,errors="coerce").iloc[-1]) if len(s)>0 else np.nan),point_count=("error_m","size"))
        label=object_df.loc[object_df["model_id"].eq(mid),"model_label"].iloc[0]; curves[label]=curve
        for rec in curve.to_dict(orient="records"): timeline_rows.append({**rec,"model_id":mid,"model_label":label})
    fig,axes=plt.subplots(1,3,figsize=(18.0,5.4),dpi=180); colors_p=plt.cm.tab10(np.linspace(0,1,len(curves)))
    for ax,(col,ylabel) in zip(axes,[("ADE_m","ADE / m"),("RMSE_m","RMSE / m"),("FDE_m","FDE / m")]):
        for (label,curve),c in zip(curves.items(),colors_p):
            valid=curve[col].notna()
            if valid.sum()>=2: ax.plot(curve["rel_time_bin_s"][valid],curve[col][valid],linewidth=1.8,label=label,color=c)
        ax.set_xlabel("预测时间 / s"); ax.set_ylabel(ylabel); ax.grid(True,linestyle="--",alpha=0.28)
    axes[0].legend(fontsize=7,ncol=3,frameon=False,loc="upper left"); fig.suptitle("Error vs Prediction Horizon",fontsize=13,y=1.01)
    fig.tight_layout(); p=out_dir/"Figure_error_vs_time.png"; fig.savefig(p,bbox_inches="tight"); plt.close(fig); paths.append(p)
    pd.DataFrame(timeline_rows).to_csv(FORMAL_DATA_DIR/"各时间节点ADE_FDE_RMSE.csv",index=False,encoding="utf-8-sig")
    # 示例轨迹对比图
    hgb=object_df[object_df["model_id"].eq("HGB-ECR")].copy(); hgb["rank_target"]=(hgb["trajectory_end_error_m"]-hgb["trajectory_end_error_m"].median()).abs()
    key=hgb.sort_values(["rank_target","duration_after_input_s"],ascending=[True,False]).iloc[0]["sample_key"]
    actual=actual_groups[key].sort_values("timestamp_s")
    fig,ax=plt.subplots(figsize=(7.4,7.2),dpi=190); plot_xs=[actual["x_m"].to_numpy(float)]; plot_ys=[actual["y_m"].to_numpy(float)]
    ax.plot(actual["x_m"],actual["y_m"],color="#222222",linewidth=2.4,label="Ground Truth")
    pal={"Kalman-CV":"#4C78A8","Helly":"#72B7B2","IDM":"#54A24B","GBR":"#F58518","CNN":"#B279A2","HGB-ECR":"#E45756"}
    od=["Kalman-CV","Helly","IDM","GBR","CNN","HGB-ECR"]
    for mid in od:
        p=paths_df[(paths_df["sample_key"].eq(key))&(paths_df["model_id"].eq(mid))]
        if p.empty: continue
        label=p["model_label"].iloc[0]; p=p.sort_values("timestamp_s")
        plot_xs.append(p["x_m"].to_numpy(float)); plot_ys.append(p["y_m"].to_numpy(float))
        ax.plot(p["x_m"],p["y_m"],linewidth=1.25 if mid!="HGB-ECR" else 2.0,label=label,color=pal.get(mid))
    t0=object_df.loc[(object_df["sample_key"].eq(key))&(object_df["model_id"].eq("HGB-ECR")),"t0"].iloc[0]
    ax.scatter(np.interp(t0,actual["timestamp_s"],actual["x_m"]),np.interp(t0,actual["timestamp_s"],actual["y_m"]),s=36,color="#000000",zorder=4,label="t0")
    dur=float(actual["timestamp_s"].max()-actual["timestamp_s"].min()); obj_id=key.replace("__seg1","").replace("__seg","-seg")
    ax.set_xlabel("x / m"); ax.set_ylabel("y / m"); ax.set_title(f"object-{obj_id}  |  {dur:.1f}s")
    ax.grid(True,linestyle="--",alpha=0.25); ax.legend(fontsize=7.6,ncol=3,frameon=False); set_balanced_xy_limits(ax,plot_xs,plot_ys)
    fig.tight_layout(); p=out_dir/"Figure_trajectory_example.png"; fig.savefig(p); plt.close(fig); paths.append(p)
    return paths
