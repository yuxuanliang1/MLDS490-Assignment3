import csv
import json
import ast
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from part1_1.utils import read_json,write_json,write_csv,sha256
from .workflow import ROOT

def rows(p):
    with open(p,newline='') as f:return list(csv.DictReader(f))

def overlaps(events):
    points=sorted([(e['start'],1) for e in events]+[(e['finish'],-1) for e in events])
    count=peak=0
    for _,change in points:count+=change;peak=max(peak,count)
    return peak

def generate():
    p=read_json(ROOT/'parallel/summary.json');s=read_json(ROOT/'sequential/summary.json')
    test=read_json(ROOT/'test.json');v=read_json(ROOT/'validation.json');bench=read_json(ROOT/'benchmarks.json')
    ph=rows(ROOT/'parallel/history.csv');sh=rows(ROOT/'sequential/history.csv')
    events=[{k:float(x) for k,x in r.items()} for r in rows(ROOT/'parallel/events.csv')]
    timings=rows(ROOT/'parallel/timings.csv')
    assert len(ph)==len(sh)==81
    assert all(int(r['train_n'])==25418 and int(r['validation_n'])==6407 for r in ph)
    peaks=[];spreads=[]
    for r in range(1,81):
        ev=[e for e in events if e['round']==r]
        assert len(ev)==4 and len({e['pid'] for e in ev})==4 and len({e['client_id'] for e in ev})==4
        peaks.append(overlaps(ev));spreads.append(max(e['finish']-e['start'] for e in ev)-min(e['finish']-e['start'] for e in ev))
    assert all(int(r['live_actor_pids'])==0 and float(r['available_cpus'])==4 for r in timings)
    rss=np.array([float(r['driver_rss']) for r in timings])/2**20
    audit=dict(rounds=80,client_actors_per_round=4,distinct_processes_per_round=True,
               maximum_training_concurrency=max(peaks),rounds_with_four_overlapping=sum(x==4 for x in peaks),
               all_actor_processes_exited_between_rounds=True,all_cpu_resources_returned=True,
               driver_rss_first_MiB=rss[0],driver_rss_last_MiB=rss[-1],driver_rss_max_MiB=rss.max(),
               mean_fastest_slowest_gap_seconds=float(np.mean(spreads)),maximum_fastest_slowest_gap_seconds=max(spreads))
    audit.update(mean_client_training_seconds=float(np.mean([e['finish']-e['start'] for e in events])),
                 mean_actor_startup_seconds=float(np.mean([float(r['startup_seconds']) for r in timings])),
                 mean_cleanup_seconds=float(np.mean([float(r['cleanup_seconds']) for r in timings])),
                 mean_aggregation_seconds=float(np.mean([float(r['aggregation_seconds']) for r in timings])),
                 mean_completion_wait_seconds=float(np.mean([max(x['finish'] for x in events if x['round']==e['round'])-e['finish'] for e in events])))
    write_json(ROOT/'lifecycle_concurrency_audit.json',audit)
    perf=[]
    for mode in ['sequential','cpu','cuda:0']:
        b=[r for r in bench if r['mode']==mode]
        if not b:continue
        times=np.array([r['wall_seconds'] for r in b]);startup=b[0]['ray_startup_seconds']
        perf.append(dict(mode=mode,repetitions=len(b),mean_round_seconds=float(times.mean()),sd_round_seconds=float(times.std(ddof=1)),
                         warm_mean_seconds=float(times[1:].mean()),runtime_startup_seconds=startup,
                         end_to_end_three_round_seconds=float(times.sum()+startup),
                         mean_actor_startup_seconds=float(np.mean([r['startup_seconds'] for r in b])),
                         mean_cleanup_seconds=float(np.mean([r['cleanup_seconds'] for r in b])),
                         mean_aggregation_seconds=float(np.mean([r['aggregation_seconds'] for r in b]))))
    reference=next(r['mean_round_seconds'] for r in perf if r['mode']=='sequential')
    for r in perf:r['speedup_vs_sequential_CPU']=reference/r['mean_round_seconds']
    write_csv(ROOT/'performance_summary.csv',perf);write_json(ROOT/'performance_summary.json',perf)
    out=ROOT/'figures';out.mkdir(exist_ok=True);manifest=[]
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'axes.grid':True,'grid.alpha':.2})
    def save(fig,name):
        fig.savefig(out/f'{name}.png',dpi=300,bbox_inches='tight');fig.savefig(out/f'{name}.pdf',bbox_inches='tight');plt.close(fig);manifest.append(name)
    rounds=np.arange(81)
    for metric,name,title,y in [('loss','ray_loss','Ray FedAvg convergence','Cross-entropy per example'),('accuracy','ray_accuracy','Ray FedAvg accuracy','Accuracy (%)')]:
        fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
        for prefix in ['train','validation']:ax.plot(rounds,[float(r[f'{prefix}_{metric}'])*(100 if metric=='accuracy' else 1) for r in ph],label=prefix.capitalize())
        ax.set(xlabel='Communication round',ylabel=y,title=title+' | 4 clients, E=3, SGD LR=0.3');ax.legend();save(fig,name)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,prefix in zip(axes,['train','validation']):
        for name,h,style in [('Sequential',sh,'-'),('Ray parallel',ph,'--')]:ax.plot(rounds,[100*float(r[f'{prefix}_accuracy']) for r in h],style,label=name)
        ax.set(xlabel='Communication round',ylabel='Accuracy (%)',title=prefix.capitalize()+' | matched initialization, clients and seeds');ax.legend()
    save(fig,'sequential_parallel_accuracy')
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    axes[0].bar([r['mode'] for r in perf],[r['mean_round_seconds'] for r in perf],yerr=[r['sd_round_seconds'] for r in perf],color=['#487f96','#e3a34c','#74a77f'][:len(perf)])
    axes[0].set(ylabel='Seconds per round (mean +/- sample SD)',title='Three repeats; fresh Actors each round')
    axes[1].bar(['Sequential CPU','Ray '+p['mode']],[s['elapsed_seconds'],p['elapsed_seconds']+p['ray_startup_seconds']],color=['#487f96','#e3a34c'])
    axes[1].set(ylabel='Full 80-round wall time (seconds)',title='Includes metrics, I/O and Ray startup')
    save(fig,'parallel_performance')
    fig,axes=plt.subplots(1,3,figsize=(15,5.5),layout='constrained')
    for round_id,ax in enumerate(axes,1):
        ev=sorted([e for e in events if e['round']==round_id],key=lambda e:e['client_id'])
        t0=min(e['created'] for e in ev)
        for y,e in enumerate(ev):
            ax.barh(y,e['finish']-e['start'],left=e['start']-t0,height=.6,color='#3b8b98')
            ax.scatter(e['created']-t0,y,marker='|',color='black')
        timing=timings[round_id-1]
        ax.axvspan(float(timing['aggregation_start'])-t0,float(timing['aggregation_finish'])-t0,color='orange',alpha=.9)
        ax.axvline(float(timing['aggregation_start'])-t0,color='orange',linewidth=1)
        ax.set(yticks=range(4),yticklabels=[f"Client {int(e['client_id'])}\nPID {int(e['pid'])}" for e in ev],
               xlabel='Measured seconds since first Actor initialization',title=f'Communication round {round_id}',ylim=(-.7,3.7))
        ax.tick_params(axis='y',labelsize=8)
    fig.suptitle('Actual concurrent local training\nBlack ticks: Actor initialization | Teal intervals: training | Orange: aggregation',fontsize=13)
    save(fig,'actor_execution_timeline');write_json(out/'manifest.json',manifest)
    extra=read_json(ROOT/'extra_validation.json') if (ROOT/'extra_validation.json').exists() else {'pending':True}
    eq=read_json(ROOT/'trajectory_equivalence.json');env=read_json(ROOT/'environment.json')
    report=['# Assignment 3 Part 1.2: Ray Core parallel clients','',
      '## Design and preserved components','',
      'Part 1.1 source/results and supplied data/references are preserved. This implementation imports its MLP (784-128-ReLU-62), keyed local SGD, weighted aggregation, sample-weighted evaluation, and checkpoint helpers. The saved split manifest is reused without modification: 100 clients, 25,418 train and 6,407 validation examples. Images use 1-x. Every required round uniformly samples exactly four distinct clients, creates exactly four fresh Ray Core Actors, broadcasts identical CPU model parameters, runs E=3 plain SGD LR=.3 batch32 concurrently, gathers all updates, aggregates in sampled order, evaluates the full pool, and terminates every Actor before the next round. Round zero is included.','',
      'Each Actor reads only its assigned client NPZ file. It returns parameters/counts/metadata, never raw examples. The shared filesystem is a logical simulation boundary, not an adversarial access-control guarantee. The server uses client-side metric summaries from the preserved evaluator. A common 150ms release barrier follows Actor readiness; this is measured orchestration overhead, not artificial training duration.','',
      'CPU Actors reserve one CPU and zero GPU; fractional GPU Actors reserve one CPU and .25 GPU each. Ray starts a private four-CPU runtime and exposes zero GPUs for CPU mode or only the verified physical GPU UUID for GPU mode. No existing Ray cluster is joined or stopped. Finally blocks kill all Actors, wait for CPU resource return AND OS PID exit; incomplete rounds never aggregate or commit. Retries are explicit: restart replays the uncommitted round with identical keyed seeds.','',
      '## Environment and execution choice','',f"Host {env['hostname']}; Python {env['python'].split()[0]}; PyTorch {env['torch']}; Ray {env['ray']}; CUDA build {env['cuda_build']}. Physical GPU 0 was an idle RTX 5060 Ti (16 GB), verified by UUID; busy GPU 1 was excluded. See nvidia_before/after and per-Actor placement records. Chosen mode: **{p['mode']}**, by lower median complete Actor-round time among numerically valid strategies. CPU driver uses one thread; four CPU Actors each use one thread (four concurrent cores). GPU allocation is logical and does not isolate memory.",'',
      '## Training, numerical equivalence and final test','',f"80 rounds: training accuracy {100*p['final_metrics']['train_accuracy']:.4f}%, validation {100*p['final_metrics']['validation_accuracy']:.4f}%; best validation {100*p['best_validation']:.4f}% at round {p['best_round']}. The five-round real pilot used the same configuration; no test-based adjustment occurred.",'',
      'The full matched sequential run uses the same 80 selections, initial weights, split, E, LR, batches and seed. Asynchronous finish order cannot change this synchronous algorithm: aggregation waits for all four and preserves order. Floating-point summation is not reordered by completion time. Full trajectory evidence:', '```json',json.dumps(eq,indent=2),'```','',
      f"Frozen validation-selected checkpoint: parallel/best.pt, round {p['best_round']}. Held-out result: **{100*test['accuracy']:.4f}% ({test['correct']}/{test['n']})**, CE {test['loss_sum']/test['n']:.6f}. Selection/checkpoint hashes were saved before this Part 1.2 test access. No settings were revised using this result. Part 1.1's 10-client score is not a target or a fair algorithm comparison against this four-client setting.",'',
      '## Performance and interpretation','',
      '| Strategy | Mean round s | Sample SD s | Warm mean s | Ray startup s | 3-round end-to-end s | CPU-reference speedup |','|---|---:|---:|---:|---:|---:|---:|']
    for r in perf:report.append(f"| {r['mode']} | {r['mean_round_seconds']:.5f} | {r['sd_round_seconds']:.5f} | {r['warm_mean_seconds']:.5f} | {r['runtime_startup_seconds']:.4f} | {r['end_to_end_three_round_seconds']:.4f} | {r['speedup_vs_sequential_CPU']:.5f} |")
    report+=['',f"Full-workload speedup (sequential wall / Ray wall including runtime startup) = **{s['elapsed_seconds']/(p['elapsed_seconds']+p['ray_startup_seconds']):.5f}x**. All measured values include fresh Actor lifecycle costs; the three-round end-to-end comparison adds Ray runtime initialization once, not once per round. Benchmark repetition 0 includes first-worker effects; warm columns use repetitions 1-2. The full workload additionally includes every global evaluation and checkpoint I/O. No speedup is inferred merely from concurrency.",'',
      'Sequential CPU uses one thread; Ray CPU uses four one-thread processes on the same host. GPU mode uses a different accelerator resource and is a strategy benchmark, not an equal-hardware CPU speedup claim. Driver-side evaluation stays CPU for both full trajectories. Shared server load and filesystem performance are uncontrolled; only three timing repeats were used. Startup imports, optimizer construction, serialization, process termination, and the measured release barrier overwhelm this small model. GPU Actors additionally pay context initialization and share one device without memory isolation.','',
      '## Scientific validation A-G','']
    for r in v:report += [f"### {r['test']}",'',f"Hypothesis: {r['hypothesis']}. Design: {r['design']}. Criterion: {r['expected']}. Outcome: {'PASS' if r['passed'] else 'REJECTED'}. Limitations: {r['limitations']}.",'','```json',json.dumps(r['observed'],indent=2),'```','']
    report+=['### B/E: actual concurrency and lifecycle','', 'Hypothesis: selected clients run in separate overlapping processes and release all resources. Design: inspect measured start/finish timestamps, four PIDs per round, CPU resources after cleanup, and driver RSS across all 80 rounds. Criterion: observed overlap, exactly four Actors per round, zero surviving Actor PIDs and four returned CPUs. Results:','```json',json.dumps(audit,indent=2),'```','',
      'The timeline displays actual timestamps, including Actor initialization and aggregation. Timing differences in local dataset sizes and computation make synchronous FedAvg wait for the slowest client; the reported fastest/slowest gap quantifies one aspect of this straggler effect. RSS is observational evidence and is not a formal leak detector; process exit and returned resources provide stronger lifecycle evidence.','',
      '### Additional robustness and repeatability','', 'Controlled tests distinguish injected MemoryError from actual hardware exhaustion. Exceptions, nonfinite updates and interrupted rounds are reported, all Actors cleaned up, and no partial aggregate committed. Extra tests use tiny synthetic clients, a CPU-only cluster requesting unavailable GPU resources with bounded wait, and deterministic replay after interruption. Results:','```json',json.dumps(extra,indent=2),'```','',
      '## Limitations and three-month extension plan','',
      'Only one full training seed was used; fixed-seed repeatability is separate from across-seed statistical stability. GPU numerical evidence covers one round; the chosen strategy has the full matched 80-round comparison. OOM is injected safely, not produced by exhausting physical RAM. A deliberately killed driver and actual machine/network failure were not tested; interruption is injected before commit and restart logic uses atomic checkpoints. Empty clients and invalid pixel/label cases are covered by preserved Part 1.1 tests, not observed in this real dataset.','',
      'Month 1: repeat timing over 10+ independent Ray sessions and training over 5-10 seeds, bootstrap at client level, vary client sizes/model size and quantify crossover workloads. Month 2: inject real worker termination and driver restart in isolated jobs, test bounded retries, heterogeneous delays, memory pressure under safe quotas and numerical agreement across architectures. Month 3: prespecify CPU/GPU budgets, collect CPU-seconds/energy/communication and client-fairness metrics, compare fixed-compute budgets, run long lifecycle stress tests and validate on newly untouched writer-held-out data. Do not repeatedly tune against the current test set.','',
      '## Figures and reproduction','', 'Five 300-DPI PNG/vector PDF pairs: loss, accuracy, matched sequential/Ray curves, performance, and actual Actor execution timeline. See README.md for exact commands. No Part 2 or combined assignment report is created.','']
    (ROOT/'REPORT_NOTES.md').write_text('\n'.join(report),encoding='utf-8')
    write_json(ROOT/'completion.json',dict(parallel=p,sequential=s,test=test,equivalence=eq,performance=perf,lifecycle=audit,extra_validation=extra,figures=manifest))

