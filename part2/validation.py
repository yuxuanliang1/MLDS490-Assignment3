from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from part1_1.data import ClientData
from part1_1.federation import Client,aggregate,evaluate_tensors
from part1_1.model import initial_model,clone_state
from part1_1.runner import Config as OriginalConfig,run_experiment,state_digest
from part1_1.utils import read_json,write_json,write_csv,sha256,load_checkpoint,keyed_seed
from .noise import laplace_noise,perturb,noise_key,perturbed_clients
from .runner import Config,run
from .workflow import ROOT,name

def record(test,hypothesis,setup,expected,observed,passed,interpretation,limitations,gating=False):
    path=ROOT/'validation/results.json'
    rows=read_json(path) if path.exists() else []
    rows=[r for r in rows if r['test']!=test]
    rows.append(dict(test=test,hypothesis=hypothesis,setup=setup,expected=expected,observed=observed,
              passed=bool(passed),gating=gating,interpretation=interpretation,limitations=limitations))
    write_json(path,rows);print('VALIDATION',test,'PASS' if passed else 'OBSERVED FAILURE',observed,flush=True)
    if gating and not passed:raise AssertionError(test)

def diff(a,b):return max(float((a[k]-b[k]).abs().max()) for k in a)

def mathematical(clean,device):
    out=ROOT/'validation';out.mkdir(exist_ok=True)
    stats=[];hist=[];n=1000000
    for i,b in enumerate([.025,.1,.8]):
        z=laplace_noise((n,),b,700+i).numpy().astype(np.float64)
        mean=float(z.mean());variance=float(z.var());mae=float(np.abs(z).mean())
        probs=np.array([.01,.1,.25,.5,.75,.9,.99]);theory=np.where(probs<.5,b*np.log(2*probs),-b*np.log(2*(1-probs)))
        quant=np.quantile(z,probs);density=np.exp(-np.abs(theory)/b)/(2*b)
        quant_z=np.abs(quant-theory)/(np.sqrt(probs*(1-probs)/n)/density)
        tail=float(np.mean(np.abs(z)>3*b));tail_se=np.sqrt(np.exp(-3)*(1-np.exp(-3))/n)
        standard=dict(mean_z=abs(mean)/(np.sqrt(2)*b/np.sqrt(n)),variance_z=abs(variance-2*b*b)/(np.sqrt(20)*b*b/np.sqrt(n)),
                     absolute_mean_z=abs(mae-b)/(b/np.sqrt(n)),max_quantile_z=float(quant_z.max()),tail_z=abs(tail-np.exp(-3))/tail_se)
        stats.append(dict(b=b,n=n,mean=mean,variance=variance,theoretical_variance=2*b*b,mean_absolute=mae,
                  quantile_probabilities=probs.tolist(),empirical_quantiles=quant.tolist(),theoretical_quantiles=theory.tolist(),tail_probability=tail,standard_errors=standard))
        counts,edges=np.histogram(z/b,bins=np.linspace(-7,7,101),density=True)
        hist.append(dict(b=b,centers=((edges[1:]+edges[:-1])/2).tolist(),density=counts.tolist()))
    write_json(out/'noise_histograms.json',hist)
    record('A_Laplace_distribution','IID Laplace draws have the specified moments, tails and quantiles','1,000,000 draws per scale .025,.1,.8; float32 samples, float64 statistics',
       'All moment, quantile and tail deviations within 6 approximate sampling SE',stats,
       all(max(s['standard_errors'].values())<6 for s in stats),'Scale-invariant normalized moments agree with Laplace theory','Normal/delta-method SE are approximate; multiple tests use conservative 6-SE bounds',True)
    z=laplace_noise((20000,28,28),.1,711).numpy().reshape(20000,784)
    positions=np.linspace(0,783,32,dtype=int);corr=np.corrcoef(z[:,positions],rowvar=False);pair=corr[np.triu_indices(32,1)]
    write_json(out/'pixel_correlations.json',dict(positions=positions.tolist(),correlation=corr.tolist()))
    record('B_pixel_independence','Distinct pixel streams have near-zero correlations','20,000 independently perturbed images; 496 pairs across 32 spread pixel positions',
       'Maximum |r| < 6/sqrt(20000), matching shape 20000x28x28',dict(max_abs_correlation=float(np.abs(pair).max()),mean_correlation=float(pair.mean()),threshold=6/np.sqrt(20000),shape=[20000,28,28]),
       np.abs(pair).max()<6/np.sqrt(20000),'No material linear dependence detected across tested positions','Correlation is not a proof of independence; generator algorithm supplies independence',True)
    del z
    hashes={cid:dict(x=state_digest({'x':c.train_x}),y=state_digest({'y':c.train_y}),v=state_digest({'v':c.val_x})) for cid,c in clean.items()}
    test_hash=sha256('Assignment3-data/test_data.npy') # bytes only; never read labels/images.
    noisy=perturbed_clients(clean,.1,2026)
    checks=[]
    for cid,c in clean.items():
        checks.append(state_digest({'x':c.train_x})==hashes[cid]['x'] and state_digest({'y':noisy[cid].train_y})==hashes[cid]['y'] and state_digest({'v':noisy[cid].val_x})==hashes[cid]['v'])
    record('D_data_isolation','Perturbation changes training pixels only','Hash every client training input/label and clean validation tensor; hash held-out FILE bytes without loading examples',
        'All 100 clients retain original tensors, labels, validation and boundaries; test file hash unchanged',dict(clients_checked=len(checks),all_checks=all(checks),test_file_sha256=test_hash),
        all(checks) and test_hash==sha256('Assignment3-data/test_data.npy'),'Original client-specific tensors are immutable inputs; aggregation imports unchanged parameter-only interface','Shared-machine logical boundary is not a security sandbox',True)
    del noisy
    a=run(clean,Config(b=0,rounds=3,device=device),out/'zero_modified')
    ref=run_experiment(clean,OriginalConfig(rounds=3,fraction=.1,epochs=3,lr=.3,seed=2026,device=device),out/'zero_original')
    p=load_checkpoint(out/'zero_modified/last.pt');q=load_checkpoint(out/'zero_original/last.pt')
    errors={k:max(abs(x[k]-y[k]) for x,y in zip(p['history'],q['history'])) for k in ['train_loss','validation_loss','train_accuracy','validation_accuracy']}
    error=diff(p['state'],q['state'])
    record('C_zero_noise_equivalence','b=0 preserves Part1.1 FedAvg exactly','Three real rounds; same architecture, weights, sampling, shuffle seeds, E3 LR.3 batch32, same GPU',
       'Every recorded metric and final parameter equal bitwise',dict(max_parameter_difference=error,metric_differences=errors,selections_equal=p['selections']==q['selections']),
       error==0 and max(errors.values())==0 and p['selections']==q['selections'],'New evaluation/noise plumbing does not change clean FedAvg','Short check; all 80-round zero-noise runs are also compared with saved Part1.1 checkpoints later',True)
    x=clean[0].train_x[:4];n1=laplace_noise(x.shape,.1,noise_key(2026,0,.1),device)
    cpu=laplace_noise(x.shape,.1,noise_key(2026,0,.1),'cpu')
    repeat=laplace_noise(x.shape,.1,noise_key(2026,0,.1),device)
    other=laplace_noise(x.shape,.1,noise_key(2027,0,.1),device)
    peer=laplace_noise(x.shape,.1,noise_key(2026,1,.1),device)
    invalid=[]
    for b in [-1.,float('nan'),float('inf')]:
        try:perturb(x,b,0)
        except ValueError:invalid.append(str(b))
    overflow=False
    try:perturb(x,1e40,0)
    except FloatingPointError:overflow=True
    zero=perturb(x,0,0)
    bad=False
    try:perturb(torch.full_like(x,float('nan')),.1,0)
    except ValueError:bad=True
    record('I_noise_reproducibility_and_edges','Keyed CPU-generated noise transfers identically across devices','Repeat keys, vary experiment seed/client ID, copy b0, reject negative/nonfinite scales and overflowing float32 samples',
       'Identical same keys and CPU/GPU; different seeds/clients differ; invalid inputs explicitly fail',
       dict(cpu_gpu_max_difference=float((n1.cpu()-cpu).abs().max()),repeat_equal=torch.equal(n1,repeat),different_seed=not torch.equal(n1,other),different_client=not torch.equal(n1,peer),zero_copy_equal=torch.equal(x,zero),zero_copy_distinct=x.data_ptr()!=zero.data_ptr(),invalid_rejected=invalid,overflow_rejected=overflow,nonfinite_input_rejected=bad),
       torch.equal(n1.cpu(),cpu) and torch.equal(n1,repeat) and not torch.equal(n1,other) and not torch.equal(n1,peer) and torch.equal(x,zero) and len(invalid)==3 and overflow and bad,
       'Noise RNG is independent of model, sampling and shuffle RNGs; b=0 takes a clone-only path','CPU generation is deliberate; does not claim identical native CUDA PRNG algorithms',True)


def scientific(clean,device):
    out=ROOT/'validation'
    rows=read_json(ROOT/'comparison.json')
    zero=[]
    for seed in [2026,2027,2028]:
        current=load_checkpoint(ROOT/'experiments'/name(0,seed)/'last.pt')
        old=load_checkpoint(Path('part1_1_deepdish5_outputs/experiments')/f'c10_e3_s{seed}'/'last.pt')
        m=max(abs(a[k]-b[k]) for a,b in zip(current['history'],old['history']) for k in ['train_loss','validation_loss','train_accuracy','validation_accuracy'])
        zero.append(dict(seed=seed,max_parameter_difference=diff(current['state'],old['state']),max_metric_difference=m,selections_equal=current['selections']==old['selections']))
    record('C_full_zero_control','Full no-noise trajectories reproduce saved Part1.1 results','Compare all 81 metric rows, 800 selections and round80 parameters for three seeds','Bitwise zero differences',zero,
           all(z['max_parameter_difference']==0 and z['max_metric_difference']==0 and z['selections_equal'] for z in zero),
           'Validates full-run reuse across noise plumbing','Identical software/GPU architecture required for bitwise agreement',True)
    signal={};g=np.random.default_rng(810)
    for cid in range(20):
        y=np.tile(np.arange(2),16);g.shuffle(y);x=g.normal(0,.01,(32,28,28)).astype(np.float32)
        x[y==0,10:12,10:12]+=.6;x[y==1,16:18,16:18]+=.6
        x=torch.tensor(x,device=device);y=torch.tensor(y,device=device)
        signal[cid]=ClientData(cid,x[:24],y[:24],x[24:],y[24:])
    controls=[]
    for seed in [371,372,373]:
        for b in [0.,.2,2.]:
            cfg=Config(b=b,seed=seed,rounds=30,epochs=1,lr=.1,batch_size=16,device=device,tag='synthetic_spatial_signal')
            try:
                r=run(signal,cfg,out/'signal'/name(b,seed));controls.append(dict(seed=seed,b=b,validation=r['final_metrics']['validation_accuracy'],finite=True))
            except FloatingPointError as e:controls.append(dict(seed=seed,b=b,validation=None,finite=False,error=str(e)))
    write_json(out/'controlled_noise_results.json',controls)
    record('E_controlled_noise_degradation','Increasing training noise weakens learning a small spatial signal','20 synthetic clients, 480 train/160 validation, three seeds, b0/.2/2, 30 rounds E1 LR.1','Compare seed-averaged validation, without requiring each run to decrease monotonically',controls,True,
       'Measured controlled responses are reported rather than assuming monotonic accuracy','Synthetic signal and small sample size differ from real handwriting')
    # Repeat a real positive-noise trajectory and compare its prefix, including state.
    cfg=Config(b=.1,rounds=5,device=device)
    r=run(clean,cfg,out/'repeat_real_b01')
    a=load_checkpoint(ROOT/'pilot'/name(.1,2026)/'last.pt');b=load_checkpoint(out/'repeat_real_b01/last.pt')
    record('I_training_reproducibility','Same seed reproduces positive-noise training','Two independent fixed b=.1 datasets and five-round runs','Exactly equal parameters and recorded metrics',dict(max_parameter_difference=diff(a['state'],b['state']),history_metrics_equal=all(x[k]==y[k] for x,y in zip(a['history'],b['history']) for k in ['train_loss','validation_loss','train_accuracy','validation_accuracy'])),
       diff(a['state'],b['state'])==0,'Fixed noise and keyed local/sampling streams permit deterministic replay','Only one positive noise scale is independently repeated',True)
    ablation=[]
    for b in [.1,.4]:
        for convention in ['fixed','resampled']:
            r=run(clean,Config(b=b,rounds=40,device=device,convention=convention),out/'ablation'/f'{convention}_b{b:g}')
            ablation.append(dict(b=b,convention=convention,final_validation=r['final_metrics']['validation_accuracy'],best_validation=r['best_validation']))
    record('J_fixed_vs_resampled','Fresh per-epoch noise changes the optimization distribution','Matched real data, same 40 rounds E3 LR.3 batch32 and sampling for b=.1,.4','Quantify difference; no universal ordering assumed',ablation,True,
       'This is a separate stochastic-training ablation and excluded from the primary fixed-noise/test grid','One seed and 40 rounds; no formal privacy accounting')
    # High-noise real-data local updates with explicit gradient instrumentation.
    stability=[]
    for b in [1e-12,1.6,3.2,10.,100.,1000.,1e6]:
        gradients=[]
        def factory():
            m=initial_model(2026,device)
            def observe(grad):
                if not torch.isfinite(grad).all():raise FloatingPointError('Non-finite gradient detected')
                gradients.append(float(grad.abs().max()));return grad
            for p in m.parameters():p.register_hook(observe)
            return m
        c=clean[25]
        x=perturb(c.train_x,b,noise_key(2026,25,b))
        noisy=ClientData(25,x,c.train_y,c.val_x,c.val_y)
        try:
            state=clone_state(initial_model(2026,device));u=Client(noisy,factory).train(state,3,.3,32,keyed_seed(2026,'local',1,25))
            m=initial_model(2026,device);m.load_state_dict(u.state)
            score=evaluate_tensors(m,c.val_x,c.val_y)
            with torch.no_grad():confidence=float(m(c.val_x).softmax(1).max(1).values.mean())
            stability.append(dict(b=b,status='finite',max_abs_pixel=float(x.abs().max()),max_abs_gradient=max(gradients),validation_loss=score['loss_sum']/score['n'],validation_accuracy=score['correct']/score['n'],mean_prediction_confidence=confidence))
        except FloatingPointError as e:stability.append(dict(b=b,status='explicit_failure',error=str(e),max_abs_pixel=float(x.abs().max())))
    record('H_high_noise_stability','Large perturbations may degrade numerical stability and useful confidence','One real small client, 3 local epochs, scales up to1e6 (stress only); parameter hooks inspect gradients; no clipping','Record finite deterioration or explicit rejection, never replace a failed update',stability,True,
       'High-noise failures are retained; finite softmax confidence alone does not establish predictive quality','Local stress test is not a full large-b FedAvg trajectory')
    # Tiny, missing-class, empty clients and cross-round resumption on synthetic data.
    tiny={}
    for cid,n in enumerate([0,1,2,7,101,23,3,9,11,13]):
        x=torch.zeros(n,28,28,device=device);y=torch.zeros(n,dtype=torch.long,device=device)
        vx=torch.zeros(2,28,28,device=device);vy=torch.zeros(2,dtype=torch.long,device=device)
        tiny[cid]=ClientData(cid,x,y,vx,vy)
    a=run(tiny,Config(b=.2,rounds=3,device=device,tag='synthetic_tiny'),out/'tiny_complete')
    cfg=Config(b=.2,rounds=3,device=device,tag='synthetic_tiny')
    run(tiny,cfg,out/'tiny_resumed',stop_after=1);run(tiny,cfg,out/'tiny_resumed')
    error=diff(load_checkpoint(out/'tiny_complete/last.pt')['state'],load_checkpoint(out/'tiny_resumed/last.pt')['state'])
    record('edge_tiny_and_resume','Missing classes, size imbalance and interrupted runs remain well-defined','Synthetic clients n=0,1,2,7,101,23,3,9,11,13, class0 only; one-round stop/resume','9 eligible clients; resumed/uninterrupted parameters equal',dict(eligible_clients=a['eligible_clients'],max_resumption_difference=error),a['eligible_clients']==9 and error==0,
       'Empty clients are ineligible; tiny nonempty clients keep their original samples and missing classes','These are synthetic stress fixtures, not observed empty real clients',True)
    diagnostics(clean)


def diagnostics(clean):
    import csv
    from .workflow import SCALES,SEEDS
    def table(path):
        with Path(path).open() as f:return list(csv.DictReader(f))
    stats={int(r['client_id']):r for r in table('part1_1_deepdish5_outputs/analysis/client_statistics.csv')}
    client_rows=[]
    for seed in SEEDS:
        base={int(r['client_id']):float(r['accuracy']) for r in table(ROOT/'experiments'/name(0,seed)/'client_final.csv')}
        for b in SCALES:
            scores=table(ROOT/'experiments'/name(b,seed)/'client_final.csv')
            for r in scores:
                cid=int(r['client_id']);s=stats[cid]
                client_rows.append(dict(client_id=cid,b=b,seed=seed,train_n=int(s['train_n']),entropy_bits=float(s['entropy_bits']),
                       observed_classes=int(s['observed_classes']),validation_n=int(r['n']),accuracy=float(r['accuracy']),change=float(r['accuracy'])-base[cid]))
    write_csv(ROOT/'client_robustness.csv',client_rows)
    correlations=[];extremes={}
    for b in [.1,.8]:
        clients=[]
        for cid in clean:
            rs=[r for r in client_rows if r['client_id']==cid and r['b']==b]
            clients.append(dict(client_id=cid,mean_change=float(np.mean([r['change'] for r in rs])),train_n=rs[0]['train_n'],entropy_bits=rs[0]['entropy_bits'],observed_classes=rs[0]['observed_classes']))
        cor={f:float(np.corrcoef([r[f] for r in clients],[r['mean_change'] for r in clients])[0,1]) for f in ['train_n','entropy_bits','observed_classes']}
        groups={}
        for f in ['train_n','entropy_bits']:
            ordered=sorted(clients,key=lambda r:r[f])
            groups[f]=dict(bottom_quartile_change=float(np.mean([r['mean_change'] for r in ordered[:25]])),top_quartile_change=float(np.mean([r['mean_change'] for r in ordered[-25:]])))
        correlations.append(dict(b=b,pearson_correlations=cor,quartile_changes=groups))
        extremes[str(b)]=sorted(clients,key=lambda r:r['mean_change'])[:3]
    record('G_client_noise_sensitivity','Federated clients may lose unequal utility under perturbation','100 original clients; per-seed final validation differences relative to matching b0, averaged over three seeds','Quantify changes and associations; no causal/pass threshold for correlations',dict(correlations=correlations,largest_degradations=extremes),True,
       'The heatmap orders clients by fixed training-label entropy; size/diversity associations are observational','Small local validation samples and writer/class differences confound associations; no causal or formal privacy interpretation')
    signal_power=sum(float(c.train_x.double().square().sum()) for c in clean.values())/(sum(c.n_train for c in clean.values())*784)
    distortion=[]
    for b in SCALES:
        mses=[];fractions=[]
        for seed in SEEDS:
            rs=table(ROOT/'experiments'/name(b,seed)/'noise_clients.csv');total=sum(int(r['n_train']) for r in rs)
            mses.append(sum(float(r['mean_squared_noise'])*int(r['n_train']) for r in rs)/total)
            fractions.append(sum(float(r['outside_unit_fraction'])*int(r['n_train']) for r in rs)/total)
        mse=float(np.mean(mses));distortion.append(dict(b=b,mean_squared_distortion=mse,rms_distortion=float(np.sqrt(mse)),
                    mean_outside_unit_fraction=float(np.mean(fractions)),ink_signal_power=signal_power,
                    ink_snr_db=float(10*np.log10(signal_power/mse)) if mse else None))
    write_json(ROOT/'distortion.json',distortion)
    record('F_image_distortion','Independent pixel noise obscures ink at increasing scale','Actual per-pixel perturbation MSE and range violations from all 21 fixed datasets; display four fixed training images with shared [0,1] display scale',
       'MSE near 2b^2; noise and out-of-range values remain unmodified in training',distortion,
       all(abs(r['mean_squared_distortion']/(2*r['b']**2)-1)<.01 for r in distortion if r['b']),
       'Ink-energy SNR uses signal 1-X to avoid white-background energy dominating the ratio; display saturation is not training clipping',
       'Images and distortion demonstrate information loss but do not measure a privacy parameter')
