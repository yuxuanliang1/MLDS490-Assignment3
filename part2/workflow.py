import argparse
from dataclasses import replace
from pathlib import Path
import time
import os
import numpy as np
import torch
from part1_1.data import load_development,validate_arrays
from part1_1.federation import evaluate_tensors
from part1_1.model import initial_model
from part1_1.utils import setup,read_json,write_json,write_csv,sha256,environment,load_checkpoint
from .runner import Config,run

ROOT=Path('part2_outputs')
SCALES=[0.,.025,.05,.1,.2,.4,.8]
SEEDS=[2026,2027,2028]
SPLIT=Path('part1_1_deepdish5_outputs/splits.json')
PROTECTED=['part1_1','part1_1_deepdish5_outputs','part1_2','part1_2_outputs','Assignment3-data']


def protected_hashes():
    return {str(p):sha256(p) for folder in PROTECTED for p in sorted(Path(folder).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.lock'}


def name(b,seed):return f'b{b:g}_s{seed}'


def development(device):
    if not SPLIT.exists():raise FileNotFoundError('Required original split manifest is missing; do not regenerate')
    return load_development('Assignment3-data',SPLIT,device)[0]


def initialize(device):
    ROOT.mkdir(exist_ok=True)
    if not (ROOT/'protected_snapshot.json').exists():write_json(ROOT/'protected_snapshot.json',protected_hashes())
    meta=environment(device);meta['visible_gpu_uuid']=os.environ.get('CUDA_VISIBLE_DEVICES');meta['physical_gpu']=os.environ.get('PART2_PHYSICAL_GPU')
    write_json(ROOT/'environment.json',meta)
    protocol=dict(scales=SCALES,seeds=SEEDS,config=Config(device=device).__dict__,
       primary='fixed per-client training noise; clean train/validation evaluation; no clipping',
       test_protocol='Evaluate round-80 checkpoints for ALL seven b and ALL three seeds once, after freeze. Also score the validation-best b=0.1 seed2026 checkpoint once for detailed Question1, labeled separately.',
       recommendation='Largest positive b with mean final clean validation accuracy no more than 2 percentage points below mean b=0; no test-based changes',
       normalized_units=True,split_sha256=sha256(SPLIT),created_at=time.time())
    if (ROOT/'predefined_protocol.json').exists():
        old=read_json(ROOT/'predefined_protocol.json');assert old['scales']==SCALES and old['seeds']==SEEDS
    else:write_json(ROOT/'predefined_protocol.json',protocol)


def pilot(clean,device):
    rows=[]
    for b in [0.,.1,.8]:rows.append(run(clean,Config(b=b,rounds=5,device=device),ROOT/'pilot'/name(b,2026)))
    write_json(ROOT/'pilot_results.json',rows)


def grid(clean,device):
    rows=[]
    for seed in SEEDS:
        for b in SCALES:
            rows.append(dict(name=name(b,seed),**run(clean,Config(b=b,seed=seed,device=device),ROOT/'experiments'/name(b,seed))))
            write_json(ROOT/'comparison.json',rows)
    stats=[]
    for b in SCALES:
        group=[r for r in rows if r['b']==b]
        vals=[r['final_metrics']['validation_accuracy'] for r in group]
        stats.append(dict(b=b,seeds=len(group),mean_validation=float(np.mean(vals)),sd_validation=float(np.std(vals,ddof=1))))
    write_csv(ROOT/'validation_seed_summary.csv',stats);write_json(ROOT/'validation_seed_summary.json',stats)


def freeze():
    if (ROOT/'frozen_protocol.json').exists():return
    validations=read_json(ROOT/'validation/results.json')
    assert all(r['passed'] for r in validations if r.get('gating'))
    rows=read_json(ROOT/'comparison.json');assert len(rows)==21
    stats=read_json(ROOT/'validation_seed_summary.json');baseline=stats[0]['mean_validation']
    eligible=[r for r in stats if r['b']>0 and r['mean_validation']>=baseline-.02]
    recommended=max(eligible,key=lambda r:r['b'])['b'] if eligible else 0.
    checkpoints=[]
    for r in rows:
        p=ROOT/'experiments'/r['name']/'last.pt'
        payload=load_checkpoint(p);assert payload['round']==80
        checkpoints.append(dict(name=r['name'],b=r['b'],seed=r['seed'],round=80,path=str(p),sha256=sha256(p)))
    best=ROOT/'experiments'/name(.1,2026)/'best.pt'
    write_json(ROOT/'frozen_protocol.json',dict(predefined_protocol_sha256=sha256(ROOT/'predefined_protocol.json'),
        frozen_at=time.time(),recommended_b=recommended,validation_tolerance_pp=2,
        rationale='Largest perturbation satisfying predefined 2pp mean validation tolerance; test not yet read in Part2',
        final_checkpoints=checkpoints,q1_best=dict(path=str(best),sha256=sha256(best),round=load_checkpoint(best)['round'])))


def test(device):
    protocol=read_json(ROOT/'frozen_protocol.json');protocol_hash=sha256(ROOT/'frozen_protocol.json')
    if (ROOT/'test_results.json').exists():
        result=read_json(ROOT/'test_results.json');assert result['protocol_sha256']==protocol_hash;return
    # This is the only Part2 function that opens held-out examples.
    raw=np.load('Assignment3-data/test_data.npy',allow_pickle=True)
    assert raw.shape==(1,)
    x,y=validate_arrays(raw[0]['images'],raw[0]['labels']);x=torch.tensor(1-x,device=device);y=torch.tensor(y,device=device)
    partial_path=ROOT/'test_progress.json'
    partial=read_json(partial_path) if partial_path.exists() else dict(protocol_sha256=protocol_hash,results=[])
    assert partial['protocol_sha256']==protocol_hash
    completed={r['name'] for r in partial['results']}
    for c in protocol['final_checkpoints']:
        if c['name'] in completed:continue
        assert sha256(c['path'])==c['sha256']
        p=load_checkpoint(c['path']);model=initial_model(c['seed'],device);model.load_state_dict(p['state'])
        scores=evaluate_tensors(model,x,y)
        partial['results'].append(dict(name=c['name'],b=c['b'],seed=c['seed'],round=80,**scores,accuracy=scores['correct']/scores['n']))
        write_json(partial_path,partial)
    c=protocol['q1_best'];assert sha256(c['path'])==c['sha256']
    if 'q1_best' not in partial:
        model=initial_model(2026,device);model.load_state_dict(load_checkpoint(c['path'])['state'])
        scores=evaluate_tensors(model,x,y)
        partial['q1_best']=dict(b=.1,seed=2026,round=c['round'],**scores,accuracy=scores['correct']/scores['n'])
        write_json(partial_path,partial)
    write_json(ROOT/'test_results.json',partial)
    rows=[]
    for train in read_json(ROOT/'comparison.json'):
        score=next(s for s in partial['results'] if s['name']==train['name'])
        rows.append(dict(b=train['b'],seed=train['seed'],final_clean_training=train['final_metrics']['train_accuracy'],
          final_validation=train['final_metrics']['validation_accuracy'],best_validation=train['best_validation'],best_round=train['best_round'],
          final_round_test=score['accuracy'],correct=score['correct'],n=score['n'],training_seconds=train['elapsed_seconds']))
    write_csv(ROOT/'final_results.csv',rows);write_json(ROOT/'final_results.json',rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['init','math','pilot','grid','science','freeze','test','artifacts','audit','train'])
    p.add_argument('--device',default='cuda:0');p.add_argument('--b',type=float,default=.1);p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--rounds',type=int,default=80);p.add_argument('--epochs',type=int,default=3);p.add_argument('--lr',type=float,default=.3)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--convention',choices=['fixed','resampled'],default='fixed');p.add_argument('--name',default='custom')
    args=p.parse_args();setup(args.seed,2)
    if args.phase=='init':return initialize(args.device)
    if args.phase=='freeze':return freeze()
    if args.phase=='test':return test(args.device)
    if args.phase=='artifacts':
        from .artifacts import generate
        return generate()
    if args.phase=='audit':
        from .audit import audit
        return audit()
    clean=development(args.device)
    if args.phase=='math':
        from .validation import mathematical
        return mathematical(clean,args.device)
    if args.phase=='science':
        from .validation import scientific
        return scientific(clean,args.device)
    if args.phase=='pilot':return pilot(clean,args.device)
    if args.phase=='grid':return grid(clean,args.device)
    if args.phase=='train':return run(clean,Config(b=args.b,seed=args.seed,rounds=args.rounds,epochs=args.epochs,lr=args.lr,batch_size=args.batch_size,device=args.device,convention=args.convention),ROOT/'custom'/args.name)

if __name__=='__main__':main()
