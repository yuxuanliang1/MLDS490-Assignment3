import argparse
from pathlib import Path
import time
import os
import subprocess
import numpy as np
import torch
import ray
from part1_1.data import load_development,validate_arrays
from part1_1.federation import Client,evaluate_global,evaluate_tensors,aggregate
from part1_1.model import initial_model,clone_state
from part1_1.utils import setup,write_json,read_json,write_csv,save_checkpoint,load_checkpoint,sha256,environment
from part1_1.tests.test_correctness import aggregation_check
from .core import Engine,sequential

ROOT=Path('part1_2_outputs')

def prepare():
    ROOT.mkdir(exist_ok=True)
    data,manifest=load_development('Assignment3-data','part1_1_deepdish5_outputs/splits.json','cpu')
    local=ROOT/'clients';local.mkdir(exist_ok=True)
    for cid,c in data.items():
        if not (local/f'{cid}.npz').exists(): np.savez(local/f'{cid}.npz',x=c.train_x.numpy(),y=c.train_y.numpy())
    return data

def registry(data):return {cid:Client(c,lambda:initial_model(2026)) for cid,c in data.items()}

def maxdiff(a,b):return max(float((a[k]-b[k]).abs().max()) for k in a)

def schedule(seed=2026,rounds=80):
    rng=np.random.default_rng(seed)
    return [rng.choice(100,4,replace=False).tolist() for _ in range(rounds)]

def train(data,mode,rounds=80,tag=None):
    out=ROOT/(tag or mode);out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():return read_json(out/'summary.json')
    engine=Engine(ROOT/'clients',mode) if mode!='sequential' else None
    started=time.perf_counter();model=initial_model(2026);clients=registry(data)
    metrics,_=evaluate_global(model,clients)
    history=[dict(round=0,**metrics)];events=[];timings=[];selections=[]
    best=(metrics['validation_accuracy'],-metrics['validation_loss']);best_round=0
    previous=out/'last.pt'
    if not previous.exists():
        save_checkpoint(out/'best.pt',dict(state=clone_state(model),round=0,metrics=metrics))
    if previous.exists():
        last=load_checkpoint(previous);model.load_state_dict(last['state']);history=last['history'];events=last['events'];timings=last['timings'];selections=last['selections'];best=last['best'];best_round=last['best_round']
        metrics={k:v for k,v in history[-1].items() if k!='round'}
    try:
        for r,ids in enumerate(schedule(rounds=rounds),1):
            if r<=history[-1]['round']:continue
            broadcast=clone_state(model);start=time.perf_counter()
            if engine:
                state,updates,timing=engine.round(ids,broadcast,r)
                events.extend(timing.pop('events'));timings.append(timing)
            else:
                state,updates,timing=sequential(data,ids,broadcast,r)
                timings.append(dict(round=r,**timing,total_seconds=time.perf_counter()-start))
            model.load_state_dict(state);metrics,_=evaluate_global(model,clients)
            history.append(dict(round=r,**metrics));selections.extend(dict(round=r,client_id=u.client_id,n_train=u.n_train) for u in updates)
            key=(metrics['validation_accuracy'],-metrics['validation_loss'])
            if key>best:
                best=key;best_round=r;save_checkpoint(out/'best.pt',dict(state=clone_state(model),round=r,metrics=metrics))
            save_checkpoint(out/'last.pt',dict(state=clone_state(model),history=history,events=events,timings=timings,selections=selections,best=best,best_round=best_round))
            write_csv(out/'history.csv',history);write_csv(out/'selections.csv',selections);write_csv(out/'timings.csv',timings)
            if events:write_csv(out/'events.csv',events)
            print(tag or mode,'round',r,'validation',metrics['validation_accuracy'],flush=True)
    finally:
        if engine:engine.close()
    summary=dict(mode=mode,rounds=rounds,seed=2026,epochs=3,lr=.3,batch_size=32,clients_per_round=4,
                 best_validation=best[0],best_round=best_round,final_metrics=metrics,elapsed_seconds=time.perf_counter()-started,
                 ray_startup_seconds=engine.startup if engine else 0,split_sha256=sha256('part1_1_deepdish5_outputs/splits.json'))
    write_json(out/'summary.json',summary)
    return summary

def validate_and_benchmark(data):
    if all((ROOT/f).exists() for f in ['validation.json','benchmarks.json','execution_choice.json']):
        print('Reusing completed validation and benchmarks',flush=True);return
    results=[]
    def record(name,hypothesis,design,expected,observed,passed,limitations):
        results.append(dict(test=name,hypothesis=hypothesis,design=design,expected=expected,observed=observed,passed=bool(passed),limitations=limitations))
        write_json(ROOT/'validation.json',results);print(name,passed,observed,flush=True)
    state=clone_state(initial_model(2026));ids=[25,67,16,36]
    seq,us,st=sequential(data,ids,state,1)
    clients=registry(data);m=initial_model(2026);m.load_state_dict(seq);sm,_=evaluate_global(m,clients)
    benchmarks=[]
    for mode in ['cpu','cuda:0']:
        engine=None
        try:
            engine=Engine(ROOT/'clients',mode)
            for repetition in range(3):
                start=time.perf_counter();actual,updates,timing=engine.round(ids,state,1)
                wall=time.perf_counter()-start
                benchmarks.append(dict(mode=mode,repetition=repetition,wall_seconds=wall,ray_startup_seconds=engine.startup,**{k:v for k,v in timing.items() if k!='events'}))
                if repetition==0:
                    m.load_state_dict(actual);am,_=evaluate_global(m,clients)
                    diffs=[maxdiff(u.state,v.state) for u,v in zip(us,updates)]
                    manual={k:sum(u.state[k].double()*u.n_train for u in updates)/sum(u.n_train for u in updates) for k in actual}
                    evidence=dict(client_max_differences=diffs,global_max_difference=maxdiff(seq,actual),metric_differences={k:abs(sm[k]-am[k]) for k in sm},manual_aggregation_error=maxdiff(actual,manual),events=timing['events'])
                    record('A_D_equivalence_'+mode,'Parallelism preserves local SGD and weighted synchronous FedAvg','Unequal real clients 25,67,16,36; identical CPU initialization and keyed batch order; E3 LR.3','CPU parameter difference <=1e-6; GPU <=1e-4',evidence,max(diffs)<= (1e-6 if mode=='cpu' else 1e-4),'GPU kernels may differ from CPU; one-round test complemented by full trajectory comparison')
            if mode=='cpu':
                for failure in ['exception','oom','nan','interrupt']:
                    caught=None
                    try:engine.round(ids,state,1,fail=failure,timeout=30)
                    except Exception as exc:caught=type(exc).__name__+': '+str(exc)[:500]
                    record('G_'+failure,'An incomplete or invalid round is never committed','Inject failure in one of four real-data actors; other actors terminated in finally','Explicit exception, resources returned',dict(error=caught,available_resources=ray.available_resources()),caught is not None,'OOM and interruption are controlled injections, not real resource exhaustion')
                engine.device='cuda:0'
                caught=None
                try:engine.round(ids,state,1,timeout=.5)
                except Exception as exc:caught=type(exc).__name__
                finally:engine.device='cpu'
                record('G_unavailable_GPU','Unavailable GPU requests fail with bounded waiting','CPU-only Ray runtime; request four fractional-GPU Actors, 0.5s deadline','Timeout and cleanup, no aggregation',dict(error=caught,resources=ray.available_resources()),caught is not None,'Tests a logical resource shortage without occupying another GPU')
                tiny={}
                from part1_1.data import ClientData
                for i,n in enumerate([1,2,7,13]):
                    x=torch.zeros(n,28,28);y=torch.zeros(n,dtype=torch.long)
                    tiny[1000+i]=ClientData(1000+i,x,y,x[:0],y[:0])
                    np.savez(ROOT/'clients'/f'{1000+i}.npz',x=x.numpy(),y=y.numpy())
                ts,_,_=sequential(tiny,list(tiny),state,1)
                tr,_,_=engine.round(list(tiny),state,1)
                repeat,_,_=engine.round(ids,state,1)
                record('G_tiny_and_recovery','Tiny, missing-class clients and replay after failure preserve FedAvg','Synthetic n=1,2,7,13, all class zero; then replay failed real round','Both parameter differences <=1e-6',dict(tiny_error=maxdiff(ts,tr),replay_error=maxdiff(seq,repeat)),maxdiff(ts,tr)<1e-6 and maxdiff(seq,repeat)<1e-6,'Synthetic edge clients are excluded from the 100-user experiment')
        except Exception as exc:
            if mode=='cpu':raise
            record('GPU_feasibility','GPU sharing must be feasible before adoption','Four actors, .25 GPU each, selected physical GPU only','Successful resource and numerical checks or explicit rejection',dict(error=str(exc)),False,'CPU execution remains valid; never fall back onto another GPU')
        finally:
            if engine:engine.close()
    for i in range(3):
        start=time.perf_counter();_,_,t=sequential(data,ids,state,1)
        benchmarks.append(dict(mode='sequential',repetition=i,wall_seconds=time.perf_counter()-start,ray_startup_seconds=0,round=1,startup_seconds=0,cleanup_seconds=0,aggregation_seconds=t['aggregation_seconds'],total_seconds=time.perf_counter()-start))
    write_json(ROOT/'benchmarks.json',benchmarks)
    for mode in ['cpu','cuda:0']:
        assert any(r['mode']==mode for r in benchmarks) or mode=='cuda:0'
    choice=min(['cpu']+(['cuda:0'] if any(r['mode']=='cuda:0' for r in benchmarks) and all(r['passed'] for r in results if r['test']=='A_D_equivalence_cuda:0') else []),key=lambda mode:np.median([r['wall_seconds'] for r in benchmarks if r['mode']==mode]))
    write_json(ROOT/'execution_choice.json',dict(mode=choice,rule='Lower median complete actor-round time over three repetitions; correctness and GPU feasibility required'))
    counts=np.zeros(100,int);samples=schedule(91,10000)
    for row in samples:counts[row]+=1
    maxz=float(np.abs(counts-400).max()/np.sqrt(10000*.04*.96))
    record('F_sampling','Uniform fixed-size sampling is reproducible','10000 rounds, 100 eligible clients, k=4','No duplicates, exact reproducibility; different seed differs; max z<5',dict(min=int(counts.min()),max=int(counts.max()),expected=400,max_abs_z=maxz),all(len(set(s))==4 for s in samples) and samples==schedule(91,10000) and samples!=schedule(92,10000) and maxz<5,'Marginal count diagnostic does not prove independence')
    record('D_synthetic_aggregation','Unequal counts give manual weighted average','Reuse Part1.1 deterministic synthetic vectors','Error <=1e-6',aggregation_check(),aggregation_check()['max_abs_error']<=1e-6,'Synthetic arithmetic test')
    if not all(r['passed'] for r in results if not r['test'].startswith('GPU_feasibility')):raise RuntimeError('Validation gate failed')

def final(data):
    choice=read_json(ROOT/'execution_choice.json')['mode']
    pilot=train(data,choice,5,'pilot')
    assert pilot['final_metrics']['validation_accuracy']>.1,'Pilot failed learning check'
    parallel=train(data,choice,80,'parallel')
    seq=train(data,'sequential',80,'sequential')
    p=load_checkpoint(ROOT/'parallel/last.pt');s=load_checkpoint(ROOT/'sequential/last.pt')
    difference=maxdiff(p['state'],s['state'])
    assert difference < (1e-6 if choice=='cpu' else 1e-3)
    write_json(ROOT/'trajectory_equivalence.json',dict(final_max_parameter_difference=difference,
              max_metric_difference=max(abs(a[k]-b[k]) for a,b in zip(p['history'],s['history']) for k in ['train_loss','validation_loss','train_accuracy','validation_accuracy']),
              identical_selections=p['selections']==s['selections']))
    selected=ROOT/'parallel/best.pt'
    if not (ROOT/'selection.json').exists():write_json(ROOT/'selection.json',dict(checkpoint=str(selected),sha256=sha256(selected),round=parallel['best_round'],criterion='Maximum validation accuracy; lower validation loss breaks ties',config=parallel))
    if not (ROOT/'test.json').exists():
        selection=read_json(ROOT/'selection.json');assert sha256(selected)==selection['sha256']
        raw=np.load('Assignment3-data/test_data.npy',allow_pickle=True)
        x,y=validate_arrays(raw[0]['images'],raw[0]['labels']);model=initial_model(2026);model.load_state_dict(load_checkpoint(selected)['state'])
        result=evaluate_tensors(model,torch.tensor(1-x),torch.tensor(y))
        write_json(ROOT/'test.json',dict(**result,accuracy=result['correct']/result['n'],selection_sha256=sha256(ROOT/'selection.json')))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','validate','train','artifacts']);args=parser.parse_args()
    setup(2026,1)
    if args.phase=='artifacts':
        from .artifacts import generate
        generate();return
    data=prepare()
    if args.phase=='prepare':write_json(ROOT/'environment.json',dict(**environment('cpu'),ray=ray.__version__))
    if args.phase=='validate':validate_and_benchmark(data)
    if args.phase=='train':final(data)

if __name__=='__main__':main()
