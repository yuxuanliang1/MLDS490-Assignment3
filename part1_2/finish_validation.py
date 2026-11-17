"""Recover independently verifiable validation records without retraining main runs."""
import numpy as np
from part1_1.model import clone_state,initial_model
from part1_1.utils import setup,read_json,write_json
from part1_1.federation import evaluate_global
from part1_1.tests.test_correctness import aggregation_check
from .core import Engine,sequential
from .workflow import ROOT,prepare,registry,maxdiff,schedule

def main():
    setup(2026,1);data=prepare();records=read_json(ROOT/'validation.json');state=clone_state(initial_model(2026));ids=[25,67,16,36]
    def add(name,hypothesis,design,expected,observed,passed,limitations):
        nonlocal records
        records=[r for r in records if r['test']!=name]
        records.append(dict(test=name,hypothesis=hypothesis,design=design,expected=expected,observed=observed,passed=bool(passed),limitations=limitations))
        assert passed,name
    seq,us,_=sequential(data,ids,state,1);engine=Engine(ROOT/'clients','cuda:0')
    try:actual,updates,timing=engine.round(ids,state,1)
    finally:engine.close()
    model=initial_model(2026);model.load_state_dict(seq);sm,_=evaluate_global(model,registry(data));model.load_state_dict(actual);pm,_=evaluate_global(model,registry(data))
    errors=[maxdiff(a.state,b.state) for a,b in zip(us,updates)]
    manual={k:sum(u.state[k].double()*u.n_train for u in updates)/sum(u.n_train for u in updates) for k in actual}
    add('A_D_equivalence_cuda:0','Fractional-GPU Actor updates match CPU SGD','Four unequal real clients, same initial weights, batch order, LR .3 E3','Max parameter error <=1e-4',dict(client_max_differences=errors,global_max_difference=maxdiff(seq,actual),metric_differences={k:abs(sm[k]-pm[k]) for k in sm},manual_aggregation_error=maxdiff(actual,manual),events=timing['events'],placements=timing['placements']),max(errors)<1e-4,'One-round GPU test; full matched trajectory uses selected CPU strategy')
    samples=schedule(91,10000);counts=np.zeros(100,int)
    for s in samples:counts[s]+=1
    z=float(np.abs(counts-400).max()/np.sqrt(10000*.04*.96))
    add('F_sampling','Uniform four-client sampling is reproducible','10000 rounds; two identical seeds and one different seed','Exactly four unique IDs, same-seed equality, different-seed difference, max z<5',dict(min=int(counts.min()),max=int(counts.max()),expected=400,max_abs_z=z,counts=counts.tolist()),z<5 and all(len(set(s))==4 for s in samples) and samples==schedule(91,10000) and samples!=schedule(92,10000),'Marginal counts are a diagnostic, not proof of uniformity')
    a=aggregation_check();add('D_synthetic_aggregation','Unequal counts produce manual weighted average','Reuse Part 1.1 deterministic vectors','Error <=1e-6',a,a['max_abs_error']<=1e-6,'Small synthetic arithmetic test')
    write_json(ROOT/'validation.json',records);print('Completed validation evidence',[(r['test'],r['passed']) for r in records],flush=True)

if __name__=='__main__':main()
