#!/usr/bin/env python3
"""Run the frozen E009 residual-model consensus-abstention experiment."""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rogii_validation.consensus_abstention import run_e009  # noqa: E402

def resolve(root:Path,path:Path)->Path: return path if path.is_absolute() else root/path

def parser()->argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--config',type=Path,default=Path('experiments/E009/config.json')); p.add_argument('--train-dir',type=Path,default=Path('data/train'))
    p.add_argument('--output-dir',type=Path,default=Path('experiments/E009/results')); p.add_argument('--artifact-dir',type=Path,default=Path('artifacts/E009')); p.add_argument('--code-sha',default='')
    return p

def main(argv:list[str]|None=None)->int:
    a=parser().parse_args(argv); root=a.root.resolve(); config=json.loads(resolve(root,a.config).read_text()); sha=a.code_sha.strip() or subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(); started=time.perf_counter()
    summary=run_e009(root=root,train_dir=resolve(root,a.train_dir),output_dir=resolve(root,a.output_dir),artifact_dir=resolve(root,a.artifact_dir),config=config,code_sha=sha)
    reported=summary['reported_candidate']; print(json.dumps({'status':summary['status'],'selected_candidate':summary['selected_candidate'],'reported_candidate':reported,'baseline_rmse':summary['baseline_metrics']['rmse'],'e006_rmse':summary['e006_metrics']['rmse'],'reported_rmse':summary['candidate_metrics'][reported]['rmse'],'eligible_candidates':summary['eligible_candidates'],'wall_seconds':time.perf_counter()-started},indent=2,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
