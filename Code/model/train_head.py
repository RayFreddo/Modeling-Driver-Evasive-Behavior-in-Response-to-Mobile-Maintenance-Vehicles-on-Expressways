"""现行 HGB-ECR 纠错头重训入口；底座和策略沿用登记训练范围。"""
import argparse, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Code/revision_step4_longhead_20261014_run11'))
import longhead as h
from _locations import relocate_loaded_modules
relocate_loaded_modules()
from threadpoolctl import threadpool_limits

def main():
    p = argparse.ArgumentParser(description='现行 HGB-ECR 纠错头重训；拒绝覆盖已有目录。不会自动发布或替换冻结模型。')
    p.add_argument('--held', required=True, choices=h.s.DAYS)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out).resolve()
    if out.exists(): raise FileExistsError(out)
    days = sorted(set(h.s.DAYS) - {args.held})
    original_load, original_refs = h.s.load_day, h.s.references
    def load(day):
        if day not in days: raise AssertionError('禁止读取留出日期训练数据: ' + day)
        return original_load(day)
    def refs(scope):
        if not set(scope) <= set(days): raise AssertionError('禁止读取留出日期标签')
        return original_refs(scope)
    h.s.load_day, h.s.references = load, refs
    with threadpool_limits(limits=2): h.fit(args.held, days, out)
    (out / '模型状态登记.json').write_text(json.dumps(dict(model='HGB-ECR',
        status='重训候选，尚未替换现行冻结权重', component='完整时域纵向纠错头',
        held=args.held, training_days=days, publication_performed=False), ensure_ascii=False, indent=2), encoding='utf-8')
    print('HGB-ECR HEAD REFIT COMPLETE', args.held)

if __name__ == '__main__': main()
