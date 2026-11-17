from dataclasses import asdict, dataclass
import time
from pathlib import Path
import numpy as np
import torch
from part1_1.federation import Client, ClientUpdate, aggregate, select_clients, evaluate_global
from part1_1.model import initial_model, clone_state
from part1_1.runner import state_digest
from part1_1.utils import keyed_seed, read_json, write_json, write_csv, save_checkpoint, load_checkpoint, sha256
from .noise import perturbed_clients, perturb, noise_key


@dataclass(frozen=True)
class Config:
    b: float = .1
    seed: int = 2026
    rounds: int = 80
    epochs: int = 3
    lr: float = .3
    batch_size: int = 32
    fraction: float = .1
    device: str = 'cuda:0'
    convention: str = 'fixed'
    tag: str = 'original_users'


class ResampledClient(Client):
    """Additional ablation: new full local noise tensor at each local epoch."""
    def train_resampled(self, state, config, round_id):
        c=self.data
        model=self.model_factory();model.load_state_dict(state);model.train()
        opt=torch.optim.SGD(model.parameters(),lr=config.lr,momentum=0)
        g=torch.Generator().manual_seed(keyed_seed(config.seed,'local',round_id,c.client_id))
        steps=0
        for ep in range(config.epochs):
            x=perturb(c.train_x,config.b,noise_key(config.seed,c.client_id,config.b,'round',round_id,'epoch',ep))
            indices=torch.randperm(c.n_train,generator=g).to(c.train_y.device)
            for idx in indices.split(config.batch_size):
                opt.zero_grad(set_to_none=True)
                loss=torch.nn.functional.cross_entropy(model(x[idx]),c.train_y[idx])
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite resampled local loss')
                loss.backward();opt.step();steps+=1
        updated=clone_state(model)
        if any(not torch.isfinite(v).all() for v in updated.values()):raise FloatingPointError('Non-finite resampled update')
        return ClientUpdate(c.client_id,c.n_train,updated,steps)


def run(clean, config, out, stop_after=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if not 0<config.fraction<=.1 or not np.isfinite(config.b) or config.b<0:
        raise ValueError('Invalid C or b')
    if config.rounds<1 or config.epochs<1 or config.batch_size<1 or not np.isfinite(config.lr) or config.lr<=0 or config.convention not in ['fixed','resampled']:
        raise ValueError('Invalid optimization settings or noise convention')
    expected=asdict(config)
    if (out/'config.json').exists() and read_json(out/'config.json')!=expected:raise ValueError('Cache config mismatch')
    write_json(out/'config.json',expected)
    if (out/'summary.json').exists():
        print('CACHE',out.name,flush=True);return read_json(out/'summary.json')
    started=time.perf_counter()
    training=perturbed_clients(clean,config.b,config.seed) if config.convention=='fixed' else clean
    factory=lambda:initial_model(config.seed,config.device)
    klass=Client if config.convention=='fixed' else ResampledClient
    clients={cid:klass(c,factory) for cid,c in training.items()}
    evaluators={cid:Client(c,factory) for cid,c in clean.items()}
    eligible=[cid for cid,c in clients.items() if c.n_train]
    model=factory()
    rng=np.random.default_rng(keyed_seed(config.seed,'sampling',config.fraction))
    noise_metadata=[]
    for cid,c in training.items():
        difference=c.train_x-clean[cid].train_x
        noise_metadata.append(dict(client_id=cid,n_train=c.n_train,noise_sha256=state_digest({'noise':difference}),
                                   mean_squared_noise=float(difference.square().mean()) if c.n_train else 0,
                                   outside_unit_fraction=float(((c.train_x<0)|(c.train_x>1)).float().mean()) if c.n_train else 0))
    write_csv(out/'noise_clients.csv',noise_metadata)
    initial=clone_state(model);digest=state_digest(initial)
    elapsed_before=0.
    if (out/'last.pt').exists():
        last=load_checkpoint(out/'last.pt');model.load_state_dict(last['state'])
        history=last['history'];selections=last['selections'];best=tuple(last['best']);best_round=last['best_round']
        rng.bit_generator.state=last['rng'];elapsed_before=last['elapsed_seconds']
    else:
        metrics,per=evaluate_global(model,evaluators)
        history=[dict(round=0,**metrics,round_seconds=0.)];selections=[]
        best=(metrics['validation_accuracy'],-metrics['validation_loss']);best_round=0
        save_checkpoint(out/'initial.pt',dict(state=initial,digest=digest))
        save_checkpoint(out/'best.pt',dict(state=initial,round=0,metrics=metrics,config=expected))
    loop_start=time.perf_counter()
    if config.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    for r in range(history[-1]['round']+1,config.rounds+1):
        tick=time.perf_counter();ids=select_clients(eligible,config.fraction,rng);broadcast=clone_state(model)
        updates=[]
        for cid in ids:
            if config.convention=='fixed':
                u=clients[cid].train(broadcast,config.epochs,config.lr,config.batch_size,keyed_seed(config.seed,'local',r,cid))
            else:u=clients[cid].train_resampled(broadcast,config,r)
            updates.append(u)
        model.load_state_dict(aggregate(updates))
        metrics,per=evaluate_global(model,evaluators) # ALWAYS clean train and validation.
        history.append(dict(round=r,**metrics,round_seconds=time.perf_counter()-tick))
        selections.extend(dict(round=r,selection_order=i,client_id=u.client_id,n_train=u.n_train,epochs=config.epochs,sgd_steps=u.steps) for i,u in enumerate(updates))
        key=(metrics['validation_accuracy'],-metrics['validation_loss'])
        if key>best:
            best=key;best_round=r;save_checkpoint(out/'best.pt',dict(state=clone_state(model),round=r,metrics=metrics,config=expected))
        elapsed=elapsed_before+time.perf_counter()-loop_start
        save_checkpoint(out/'last.pt',dict(state=clone_state(model),round=r,config=expected,history=history,selections=selections,best=best,best_round=best_round,rng=rng.bit_generator.state,elapsed_seconds=elapsed))
        write_csv(out/'history.csv',history);write_csv(out/'selections.csv',selections)
        if r==1 or r%10==0 or r==config.rounds:print(out.name,'round',r,'clean_val',metrics['validation_accuracy'],'seconds',round(elapsed,2),flush=True)
        if stop_after is not None and r>=stop_after and r<config.rounds:return dict(interrupted_after=r)
    metrics,per=evaluate_global(model,evaluators);write_csv(out/'client_final.csv',per)
    write_json(out/'history.json',history)
    summary=dict(**expected,initial_digest=digest,eligible_clients=len(eligible),final_metrics=metrics,
                 best_validation=best[0],best_round=best_round,elapsed_seconds=elapsed_before+time.perf_counter()-loop_start,
                 preparation_seconds=loop_start-started,peak_cuda_bytes=torch.cuda.max_memory_allocated() if config.device.startswith('cuda') else 0,
                 training_examples=sum(c.n_train for c in clean.values()),split_sha256=sha256('part1_1_deepdish5_outputs/splits.json'))
    write_json(out/'summary.json',summary);return summary
