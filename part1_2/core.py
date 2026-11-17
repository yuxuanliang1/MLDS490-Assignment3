import os
import time
from pathlib import Path
import numpy as np
import psutil
import ray
import torch
from part1_1.data import ClientData
from part1_1.federation import Client, aggregate
from part1_1.model import initial_model
from part1_1.utils import setup, keyed_seed


@ray.remote(num_cpus=1, max_restarts=0, max_task_retries=0)
class LocalActor:
    def __init__(self, cid, directory, device, seed):
        self.created=time.time()
        setup(seed,1)
        self.device=device
        p=np.load(Path(directory)/f'{cid}.npz')
        empty=torch.empty(0,28,28,device=device)
        self.data=ClientData(cid,torch.tensor(p['x'],device=device),torch.tensor(p['y'],device=device),empty,torch.empty(0,dtype=torch.long,device=device))
        self.client=Client(self.data,lambda:initial_model(seed,device))

    def ready(self):
        return dict(pid=os.getpid(),created=self.created,cid=self.data.client_id,
                    visible=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu_ids=ray.get_gpu_ids(),
                    gpu=torch.cuda.get_device_name() if self.device=='cuda:0' else None,
                    rss=psutil.Process().memory_info().rss,torch_threads=torch.get_num_threads())

    def train(self,state,round_id,seed,epochs,lr,batch,start_at=0,fail=None):
        while time.time()<start_at:
            time.sleep(min(.005,start_at-time.time()))
        start=time.time()
        if fail=='exception': raise RuntimeError('Injected client exception')
        if fail=='oom': raise MemoryError('Injected allocation failure; no physical memory exhaustion')
        if fail=='interrupt': time.sleep(10)
        u=self.client.train(state,epochs,lr,batch,keyed_seed(seed,'local',round_id,self.data.client_id))
        u.state={k:v.cpu() for k,v in u.state.items()}
        if fail=='nan': next(iter(u.state.values())).fill_(float('nan'))
        return u,dict(client_id=u.client_id,round=round_id,pid=os.getpid(),created=self.created,
                      start=start,finish=time.time(),n_train=u.n_train,
                      rss=psutil.Process().memory_info().rss,
                      cuda_peak=torch.cuda.max_memory_allocated() if self.device=='cuda:0' else 0)


class Engine:
    def __init__(self, directory, device='cpu'):
        self.directory=str(Path(directory).resolve()); self.device=device
        self.startup=time.perf_counter()
        ray.init(num_cpus=4,num_gpus=1 if device=='cuda:0' else 0,include_dashboard=False,
                 object_store_memory=256*1024**2,log_to_driver=False)
        self.startup=time.perf_counter()-self.startup

    def close(self): ray.shutdown()

    def round(self,ids,state,r,seed=2026,epochs=3,lr=.3,batch=32,fail=None,timeout=180):
        if len(ids)!=4 or len(set(ids))!=4: raise ValueError('Exactly four distinct clients required')
        actors=[]; t0=time.perf_counter(); timings=[]; ready=[]
        try:
            for cid in ids:
                actors.append(LocalActor.options(num_gpus=.25 if self.device=='cuda:0' else 0).remote(cid,self.directory,self.device,seed))
            ready=ray.get([a.ready.remote() for a in actors],timeout=timeout)
            startup=time.perf_counter()-t0
            shared=ray.put(state)
            # A common near-future release time prevents startup skew obscuring overlap.
            release=time.time()+.15
            refs=[a.train.remote(shared,r,seed,epochs,lr,batch,release,fail if i==0 else None) for i,a in enumerate(actors)]
            if fail=='interrupt':
                time.sleep(.3); raise InterruptedError('Injected interruption: round not committed')
            done,pending=ray.wait(refs,num_returns=4,timeout=timeout)
            if pending: raise TimeoutError('Incomplete round: no aggregation permitted')
            results=ray.get(refs) # Preserve sampled order, not asynchronous completion order.
            updates,timings=map(list,zip(*results))
            aggregation_start=time.time()
            t=time.perf_counter(); merged=aggregate(updates); aggregation=time.perf_counter()-t
            end=time.time()
        finally:
            clean=time.perf_counter()
            for a in actors: ray.kill(a,no_restart=True)
            deadline=time.perf_counter()+30
            while ray.available_resources().get('CPU',0)<4 and time.perf_counter()<deadline:
                time.sleep(.02)
            # Resource return AND OS process exit are checked, not merely loss of handles.
            pids=[x['pid'] for x in ready]
            while any(psutil.pid_exists(p) for p in pids) and time.perf_counter()<deadline:
                time.sleep(.02)
            cleanup=time.perf_counter()-clean
            if ray.available_resources().get('CPU',0)<4 or any(psutil.pid_exists(p) for p in pids):
                raise RuntimeError('Actor cleanup failed')
        return merged, updates, dict(round=r,startup_seconds=startup,cleanup_seconds=cleanup,
                 aggregation_seconds=aggregation,total_seconds=time.perf_counter()-t0,
                 events=timings,placements=ready,aggregation_start=aggregation_start,aggregation_finish=end,
                 available_cpus=ray.available_resources().get('CPU',0),driver_rss=psutil.Process().memory_info().rss,
                 live_actor_pids=sum(psutil.pid_exists(p) for p in pids))


def sequential(data,ids,state,r,seed=2026,epochs=3,lr=.3,batch=32):
    t=time.perf_counter()
    updates=[Client(data[c],lambda:initial_model(seed)).train(state,epochs,lr,batch,keyed_seed(seed,'local',r,c)) for c in ids]
    train=time.perf_counter()-t
    t=time.perf_counter(); merged=aggregate(updates)
    return merged,updates,dict(training_seconds=train,aggregation_seconds=time.perf_counter()-t)
