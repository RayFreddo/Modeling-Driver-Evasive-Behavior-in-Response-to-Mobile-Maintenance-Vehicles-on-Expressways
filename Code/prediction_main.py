"""六模型批量预测主入口：预测、汇总指标并生成图表。"""
import json
import pickle
import time

import pandas as pd

from prediction_shared_functions import *

OUT_XLSX = FORMAL_DATA_DIR / "六模型整条轨迹累计预测正式结果.xlsx"
OUT_JSON = FORMAL_DATA_DIR / "六模型整条轨迹累计预测正式结果.json"

MODELS = ["Kalman-CV","Helly","IDM","GBR","CNN","HGB-ECR"]

def run():
    started=time.time()
    full_0409,validation,actual_groups,models,train_stats=load_all(do_train=True)

    # 逐模型预测（带缓存）
    all_point=[]; all_object=[]; all_step=[]; all_paths=[]
    for mid in MODELS:
        cache_path=CACHE_DIR/f"{mid}_prediction.pkl"
        if cache_path.exists():
            now_msg(f"加载缓存: {mid}")
            paths_df,point_df,object_df,step_df=pickle.loads(cache_path.read_bytes())
        else:
            now_msg(f"开始预测: {mid}")
            label=MODEL_LABELS[mid]
            paths_df,point_df,object_df,step_df=run_one_model(mid,label,models[mid],validation,actual_groups)
            cache_path.write_bytes(pickle.dumps((paths_df,point_df,object_df,step_df)))
            now_msg(f"完成预测: {mid} (缓存已保存)")
        all_paths.append(paths_df); all_point.append(point_df); all_object.append(object_df); all_step.append(step_df)

    paths_df=pd.concat(all_paths,ignore_index=True,sort=False) if all_paths else pd.DataFrame()
    point_df=pd.concat(all_point,ignore_index=True,sort=False)
    object_df=pd.concat(all_object,ignore_index=True,sort=False)
    step_df=pd.concat(all_step,ignore_index=True,sort=False)
    summary_df=summarize_model_metrics(point_df,object_df)

    now_msg("生成图表")
    figure_paths=plot_all_figures(point_df,object_df,paths_df,actual_groups)

    now_msg(f"写出结果: {OUT_XLSX}")
    with pd.ExcelWriter(OUT_XLSX,engine="openpyxl") as writer:
        summary_df.to_excel(writer,sheet_name="六模型总体指标",index=False)
        object_df.to_excel(writer,sheet_name="逐车整条轨迹指标",index=False)
        point_df.to_excel(writer,sheet_name="逐时刻误差",index=False)
        step_df.to_excel(writer,sheet_name="递推步骤记录",index=False)
        slim_cols=["model_id","model_label","sample_key","step","source","timestamp_s","x_m","y_m","distance_m","follower_speed_mps","follower_acc_mps2","lane_index"]
        paths_df[[c for c in slim_cols if c in paths_df.columns]].to_excel(writer,sheet_name="预测轨迹点",index=False)

    payload={"created_at":time.strftime("%Y-%m-%d %H:%M:%S"),"runtime_s":round(time.time()-started,3),
        "output_xlsx":str(OUT_XLSX),"figure_dir":str(FIG_DIR),"figures":[str(p) for p in figure_paths],
        "summary":summary_df.to_dict(orient="records"),"training_stats":train_stats}
    OUT_JSON.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    now_msg("完成")
    for r in summary_df.to_dict(orient="records"):
        print(f"  {r['model_label']:12s}  ADE={r['ADE_m']:.2f}  FDE={r['FDE_m']:.2f}  RMSE={r['RMSE_m']:.2f}")

if __name__=="__main__":
    run()
