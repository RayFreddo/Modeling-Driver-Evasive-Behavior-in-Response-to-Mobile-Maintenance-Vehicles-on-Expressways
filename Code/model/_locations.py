"""Resolve relocated output dependencies without changing historical numerical implementations."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
FOLDS=ROOT/'五折独立实验_20261012'
CURRENT=FOLDS/'HGB-ECR_现行交付_20261014'
ARCHIVE=FOLDS/'历史输出与运行依赖_20261014'
OLD=ROOT/'outputs'

def resolve(path):
    path=Path(path)
    try:parts=path.relative_to(OLD).parts
    except ValueError:return path
    if parts and parts[0]=='HGB-ECR_现行交付_20261014':return CURRENT.joinpath(*parts[1:])
    return ARCHIVE.joinpath(*parts)

def relocate_loaded_modules():
    """Update only Path-valued output/cache constants in loaded project modules."""
    updated=[]
    for module in list(sys.modules.values()):
        if getattr(module,'__name__',None)==__name__:continue
        name=getattr(module,'__file__',None)
        if not name:continue
        try:Path(name).resolve().relative_to(ROOT/'Code')
        except ValueError:continue
        for key,value in list(vars(module).items()):
            if isinstance(value,Path):
                new=resolve(value)
                if new!=value:setattr(module,key,new);updated.append((module.__name__,key,str(new)))
    return updated
