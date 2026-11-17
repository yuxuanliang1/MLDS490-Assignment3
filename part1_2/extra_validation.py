"""Additional bounded failure tests; runs on DeepDish5 after main training."""
import numpy as np
import torch
import ray
from part1_1.data import ClientData
from part1_1.model import initial_model,clone_state
from part1_1.utils import setup,write_json
from .core import Engine,sequential
from .workflow import ROOT,prepare,maxdiff

def main():
    setup(2026,1);data=prepare();state=clone_state(initial_model(2026));e=Engine(ROOT/'clients','cpu');out={}
    try:
        e.device='cuda:0'
        try:e.round([25,67,16,36],state,1,timeout=.5)
        except Exception as exc:out['unavailable_gpu']=dict(reported=type(exc).__name__,cpus_returned=ray.available_resources().get('CPU',0))
        else:raise AssertionError('Unschedulable GPU request unexpectedly succeeded')
        finally:e.device='cpu'
        tiny={}
        for i,n in enumerate([1,2,7,13]):
            x=torch.zeros(n,28,28);y=torch.zeros(n,dtype=torch.long)
            tiny[1000+i]=ClientData(1000+i,x,y,x[:0],y[:0]);np.savez(ROOT/'clients'/f'{1000+i}.npz',x=x.numpy(),y=y.numpy())
        s,_,_=sequential(tiny,list(tiny),state,1);r,_,_=e.round(list(tiny),state,1)
        out['tiny_missing_classes']=dict(sizes=[1,2,7,13],parameter_difference=maxdiff(s,r))
        ids=[25,67,16,36]
        try:e.round(ids,state,1,fail='interrupt')
        except InterruptedError:pass
        else:raise AssertionError('Interruption not reported')
        a,_,_=e.round(ids,state,1);b,_,_=e.round(ids,state,1)
        out['replay_and_repeat']=dict(parameter_difference=maxdiff(a,b),input_unchanged=maxdiff(state,clone_state(initial_model(2026)))==0)
        assert maxdiff(a,b)==0 and maxdiff(s,r)<1e-6
        out['passed']=True
    finally:e.close()
    write_json(ROOT/'extra_validation.json',out);print(out,flush=True)

if __name__=='__main__':main()
