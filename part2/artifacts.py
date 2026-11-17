"""Only saved metrics and fixed development examples feed the final figures."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from part1_1.utils import read_json,write_json,write_csv
from .workflow import ROOT,SCALES,SEEDS,name,development
from .noise import perturb,noise_key


def table(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def generate():
    rows=read_json(ROOT/'final_results.json');test=read_json(ROOT/'test_results.json')
    frozen=read_json(ROOT/'frozen_protocol.json');validation=read_json(ROOT/'validation/results.json')
    assert len(rows)==21 and len(test['results'])==21
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.18,'pdf.fonttype':42,'savefig.dpi':300})
    manifest=[]
    def save(fig,filename,description):
        fig.savefig(out/f'{filename}.png',dpi=300,bbox_inches='tight',facecolor='white')
        fig.savefig(out/f'{filename}.pdf',bbox_inches='tight',facecolor='white');plt.close(fig)
        manifest.append(dict(name=filename,description=description))
    for metric,title,ylabel,filename in [('accuracy','Accuracy','Accuracy (%)','dp_accuracy_convergence'),('loss','Loss','Cross-entropy per clean example','dp_loss_convergence')]:
        fig,ax=plt.subplots(figsize=(10,5.2),layout='constrained')
        for b,color in [(0,'#287c8e'),(.1,'#cb773a')]:
            h=table(ROOT/'experiments'/name(b,2026)/'history.csv')
            for partition,style in [('train','-'),('validation','--')]:
                ax.plot([int(r['round']) for r in h],[float(r[f'{partition}_{metric}'])*(100 if metric=='accuracy' else 1) for r in h],color=color,linestyle=style,label=f'b={b:g}, clean {partition}')
        ax.set(xlabel='Communication round',ylabel=ylabel,title=f'FedAvg with fixed Laplace training perturbations: {title.lower()}\nSeed 2026 | 10 clients/round, E=3, SGD LR=0.3');ax.legend(ncol=2,fontsize=9)
        save(fig,filename,'Clean train/validation curves for b=0 and b=0.1; includes round zero')
    primary=sorted([r for r in rows if r['seed']==2026],key=lambda r:r['b'])
    fig,ax=plt.subplots(figsize=(10,5.2),layout='constrained')
    ax.plot(SCALES,[100*r['final_clean_training'] for r in primary],'-o',label='Clean training accuracy, round 80',color='#287c8e')
    ax.plot(SCALES,[100*r['final_round_test'] for r in primary],'-s',label='Clean held-out test accuracy, round 80',color='#cb773a')
    ax.set(xlabel='Laplace scale b (normalized pixel units)',ylabel='Accuracy (%)',title='Final-round utility versus Laplace scale | seed 2026',xticks=SCALES)
    ax.set_xscale('symlog',linthresh=.025,base=2)
    ax.set_xticks(SCALES,labels=[f'{b:g}' for b in SCALES])
    ax.set_xlabel('Laplace scale b (normalized pixel units; symlog spacing)')
    ax.set_xlim(-.002,.88)
    ax.tick_params(axis='x',labelsize=9);ax.legend()
    save(fig,'dp_accuracy_vs_noise_scale','Both curves use each configuration\'s round-80 checkpoint; no best/final checkpoint mixing')
    seed_summary=[]
    for b in SCALES:
        rs=[r for r in rows if r['b']==b]
        seed_summary.append(dict(b=b,mean_validation=float(np.mean([r['final_validation'] for r in rs])),sd_validation=float(np.std([r['final_validation'] for r in rs],ddof=1)),
             mean_test=float(np.mean([r['final_round_test'] for r in rs])),sd_test=float(np.std([r['final_round_test'] for r in rs],ddof=1)),mean_training=float(np.mean([r['final_clean_training'] for r in rs]))))
    write_json(ROOT/'seed_summary.json',seed_summary);write_csv(ROOT/'seed_summary.csv',seed_summary)
    fig,ax=plt.subplots(figsize=(10,5.2),layout='constrained')
    for metric,color,marker in [('validation','#287c8e','o'),('test','#cb773a','s')]:
        ax.errorbar(SCALES,[100*r[f'mean_{metric}'] for r in seed_summary],yerr=[100*r[f'sd_{metric}'] for r in seed_summary],marker=marker,color=color,capsize=4,label=f'Clean {metric}: mean +/- 1 sample SD')
    ax.set(xlabel='Laplace scale b (normalized pixel units)',ylabel='Final-round accuracy (%)',title='Noise sensitivity across seeds 2026, 2027 and 2028',xticks=SCALES);ax.tick_params(axis='x',labelsize=9);ax.legend()
    ax.set_xscale('symlog',linthresh=.025,base=2)
    ax.set_xticks(SCALES,labels=[f'{b:g}' for b in SCALES])
    ax.set_xlabel('Laplace scale b (normalized pixel units; symlog spacing)')
    ax.set_xlim(-.002,.88)
    save(fig,'dp_noise_seed_stability','All three prespecified seeds are reported; sample SD is not a confidence interval')
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    h=read_json(ROOT/'validation/noise_histograms.json');x=np.linspace(-7,7,501)
    for r in h:axes[0].plot(r['centers'],r['density'],label=f"Empirical b={r['b']}",alpha=.8)
    axes[0].plot(x,.5*np.exp(-np.abs(x)),'k--',label='Theoretical Laplace(0,1)',linewidth=2)
    axes[0].set(xlabel='Noise / b',ylabel='Normalized density',title='One million independent samples per scale');axes[0].legend(fontsize=8)
    a=next(r['observed'] for r in validation if r['test']=='A_Laplace_distribution')
    for r in a:axes[1].plot(np.array(r['theoretical_quantiles'])/r['b'],np.array(r['empirical_quantiles'])/r['b'],'o-',label=f"b={r['b']}")
    axes[1].plot([-4.5,4.5],[-4.5,4.5],'k--',label='Perfect agreement');axes[1].set(xlabel='Theoretical quantile / b',ylabel='Empirical quantile / b',title='Quantile agreement');axes[1].legend(fontsize=8)
    save(fig,'dp_noise_distribution_validation','Moment, quantile and tail criteria use sampling uncertainty; histograms show the normalized central range')
    clean=development('cpu');c=clean[0]
    fig,axes=plt.subplots(4,4,figsize=(9,9),layout='constrained')
    for j,b in enumerate([0.,.025,.1,.8]):
        noisy=perturb(c.train_x[:4],b,noise_key(2026,0,b))
        for i in range(4):
            axes[i,j].imshow(1-noisy[i].numpy(),cmap='gray',vmin=0,vmax=1,interpolation='nearest')
            axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
            if i==0:axes[i,j].set_title(f'b={b:g}')
            if j==0:axes[i,j].set_ylabel(f'Client 0, train index {i}\nLabel ID {int(c.train_y[i])}',fontsize=9)
    fig.suptitle('Fixed training examples under Laplace perturbation\nShared display range [0,1]; training values are never clipped',fontsize=12)
    save(fig,'dp_image_perturbations','First four saved training indices of client 0; positive raw-image noise is applied before 1-X transformation')
    robustness=table(ROOT/'client_robustness.csv');matrix=np.zeros((100,7))
    for cid in range(100):
        for j,b in enumerate(SCALES):matrix[cid,j]=100*np.mean([float(r['change']) for r in robustness if int(r['client_id'])==cid and float(r['b'])==b])
    entropy={int(r['client_id']):float(r['entropy_bits']) for r in robustness}
    order=sorted(range(100),key=lambda cid:(entropy[cid],cid));worst=np.argsort(matrix[:,-1])[:3]
    fig,ax=plt.subplots(figsize=(10,10),layout='constrained');bound=max(1,float(np.max(np.abs(matrix))))
    image=ax.imshow(matrix[order],aspect='auto',interpolation='nearest',cmap='RdBu',vmin=-bound,vmax=bound)
    ticks=[order.index(int(cid)) for cid in worst]
    for i in range(0,100,10):
        if all(abs(i-j)>=5 for j in ticks):ticks.append(i)
    ticks=sorted(ticks)
    ax.set(xticks=range(7),xticklabels=[f'{b:g}' for b in SCALES],yticks=ticks,
           yticklabels=[f'{order[i]}'+(' *' if order[i] in worst else '') for i in ticks],xlabel='Laplace scale b',ylabel='Client ID, ordered by training-label entropy (ascending)',
           title='Client-Level Robustness to Laplace Perturbation\nRound 80; paired differences averaged across three seeds; * largest losses at b=0.8')
    ax.grid(False);fig.colorbar(image,ax=ax,label='Clean validation accuracy change from matching b=0 (percentage points)',shrink=.8)
    save(fig,'dp_client_robustness_heatmap','Non-standard client-level utility diagnostic; ordering uses training-label entropy, not test data')
    distortion=read_json(ROOT/'distortion.json')
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    axes[0].plot(SCALES,[r['rms_distortion'] for r in distortion],'-o',label='Measured fixed-noise RMS')
    axes[0].plot(SCALES,np.sqrt(2)*np.array(SCALES),'k--',label='Theoretical sqrt(2) b')
    axes[0].set(xlabel='Laplace scale b',ylabel='Pixel distortion RMS (normalized units)',title='Perturbation magnitude');axes[0].legend(fontsize=8)
    axes[1].errorbar([r['rms_distortion'] for r in distortion],[100*r['mean_validation'] for r in seed_summary],yerr=[100*r['sd_validation'] for r in seed_summary],fmt='-o',capsize=3)
    label_positions=[(.02,77),(.18,76),(.34,74.5),(.45,71.5),(.39,65.5),(.62,59),(1.,38)]
    for d,r,position in zip(distortion,seed_summary,label_positions):axes[1].annotate(f"b={r['b']:g}",(d['rms_distortion'],100*r['mean_validation']),xytext=position,textcoords='data',fontsize=8,arrowprops=dict(arrowstyle='-',color='#777777',linewidth=.5))
    axes[1].set_ylim(30,80)
    axes[1].set(xlabel='Measured pixel distortion RMS',ylabel='Mean clean validation accuracy (%)',title='Utility versus perturbation (three seeds)')
    save(fig,'dp_privacy_utility_tradeoff','Noise magnitude is not epsilon; recommendation uses validation utility only')
    write_json(out/'manifest.json',manifest)
    from .documentation import generate_notes
    generate_notes(seed_summary,manifest)
