"""现行 HGB-ECR 训练日期内选参入口；复用已核查的内层权重。"""
import argparse, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Code/revision_step4_matched_teacher_20261014_run13'))
import rms_selection as r
from _locations import relocate_loaded_modules
relocate_loaded_modules()
from threadpoolctl import threadpool_limits

def main():
    p = argparse.ArgumentParser(description='仅四个训练日期内部选参；复用登记内层教师和纠错头，拒绝覆盖。')
    p.add_argument('--held', required=True, choices=r.h.s.DAYS)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    r.OUT = Path(args.out).resolve()
    r.OUT.mkdir(parents=True, exist_ok=False)
    allowed = set(r.h.s.DAYS) - {args.held}
    original_load, original_refs = r.h.s.load_day, r.h.s.references
    def load(day):
        if day not in allowed: raise AssertionError('禁止读取留出日期')
        return original_load(day)
    def refs(days):
        if not set(days) <= allowed: raise AssertionError('禁止读取留出日期标签')
        return original_refs(days)
    r.h.s.load_day, r.h.s.references = load, refs
    with threadpool_limits(limits=2): r.select(args.held)
    (r.OUT / '模型状态登记.json').write_text(json.dumps(dict(model='HGB-ECR',
        status='训练内选参复现，尚未替换现行冻结配置', held=args.held), ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__': main()
