"""Render report-sized PNGs from existing results only; no training or PDF output.

Run from any directory with NumPy/Matplotlib installed and the repository's
completed output directories present. Overleaf does not need this script.
"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPORT = Path(__file__).resolve().parents[1]
REPO = REPORT.parent
P11 = REPO / 'part1_1_deepdish5_outputs'
P12 = REPO / 'part1_2_outputs'
P2 = REPO / 'part2_outputs'
records = []

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))

def values(table, key):
    return np.array([float(r[key]) for r in table])

def save(fig, part, name, sources):
    path = REPORT / 'figures' / part / (name + '.png')
    fig.savefig(path, dpi=300, facecolor='white')
    plt.close(fig)
    records.append(dict(report_path=path.relative_to(REPORT).as_posix(),
                        source_data=[p.relative_to(REPO).as_posix() for p in sources],
                        source_hashes={p.relative_to(REPO).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        operation='Presentation-only regeneration from saved numerical results'))

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.titlesize': 10, 'axes.labelsize': 9, 'xtick.labelsize': 8.5,
                     'ytick.labelsize': 8.5, 'legend.fontsize': 8,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': .18})

counts_path = P11 / 'analysis/class_counts.json'
summary_path = P11 / 'analysis/dataset_summary.json'
counts = read(counts_path)
summary = read(summary_path)
labels = np.arange(62)
ticks = [*range(0, 60, 5), 61]
fig, ax = plt.subplots(figsize=(6.7, 2.65), layout='constrained')
ax.bar(labels, counts['development_pool'], color='#256d85', width=.85)
ax.set(xlabel='Numeric class ID', ylabel='Development examples',
       title='Overall class distribution (31,825 examples)', xlim=(-1, 62), xticks=ticks)
ax.set_xticks(labels, minor=True)
ax.grid(axis='x', visible=False)
assert sum(counts['development_pool']) == 31825
save(fig, 'part1_1', 'overall_class_distribution', [counts_path])

fig, axes = plt.subplots(5, 1, figsize=(6.7, 5.25), sharex=True, layout='constrained')
for ax, r in zip(axes, summary['representative_clients']):
    cid = r['client_id']
    ys = counts['client_training'][cid]
    assert sum(ys) == r['train_n']
    ax.bar(labels, ys, color='#328e89', width=.85)
    ax.set_title(f"Client {cid}: n={r['train_n']}, {r['observed_classes']} classes, H={r['entropy_bits']:.2f} bits", loc='left', fontsize=9)
    ax.set_ylabel('Count')
    ax.set_xlim(-1, 62)
    ax.set_xticks(ticks)
    ax.set_xticks(labels, minor=True)
    ax.grid(axis='x', visible=False)
axes[-1].set_xlabel('Numeric class ID')
save(fig, 'part1_1', 'five_client_distributions', [counts_path, summary_path])

path = P11 / 'experiments/c10_e1_s2026/history.csv'
table = rows(path)
fig, axes = plt.subplots(2, 2, figsize=(6.7, 4.3), layout='constrained')
for ax, metric, title in zip(axes.flat,
        ['train_loss', 'validation_loss', 'train_accuracy', 'validation_accuracy'],
        ['Training loss', 'Validation loss', 'Training accuracy', 'Validation accuracy']):
    accuracy = metric.endswith('accuracy')
    ax.plot(values(table, 'round'), values(table, metric) * (100 if accuracy else 1),
            color='#287c8e', linewidth=1.3)
    ax.set(xlabel='Communication round', ylabel='Accuracy (%)' if accuracy else 'Cross-entropy', title=title)
save(fig, 'part1_1', 'baseline_convergence', [path])

path = P11 / 'validation/client_benefits.csv'
table = rows(path)
fig, ax = plt.subplots(figsize=(6.7, 3.8), layout='constrained')
sizes = values(table, 'train_n')
sc = ax.scatter(values(table, 'participation'), 100*values(table, 'benefit'),
                c=values(table, 'entropy_bits'), s=sizes*.28, cmap='viridis',
                alpha=.8, edgecolors='white', linewidths=.4)
ax.axhline(0, color='#555555', linestyle='--', linewidth=.8)
ax.axvline(8, color='#b74f37', linestyle=':', linewidth=1.3)
for cid in [29, 5, 6, 62]:
    row = next(r for r in table if int(r['client_id']) == cid)
    ax.annotate(str(cid), (float(row['participation']), 100*float(row['benefit'])),
                xytext=(4, 4), textcoords='offset points', fontsize=8)
ax.set(xlabel='Selections in 80 rounds', ylabel='Validation accuracy improvement (pp)',
        title='Client participation, heterogeneity, and learning benefit')
fig.colorbar(sc, ax=ax, label='Training-label entropy (bits)')
handles = [ax.scatter([], [], s=n*.28, color='#698080', label=str(n)) for n in [100, 200, 300]]
ax.legend(handles=handles, title='Training count', loc='lower right', fontsize=8, title_fontsize=8, ncol=3)
save(fig, 'part1_1', 'participation_heterogeneity_benefit', [path])

fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.15), layout='constrained')
sources = []
colors = {.02: '#3366a5', .05: '#c57912', .1: '#26867e'}
for fraction in [.02, .05, .1]:
    for epochs in [1, 3]:
        path = P11 / f'experiments/c{round(fraction*100):02d}_e{epochs}_s2026/history.csv'
        table = rows(path)
        sources.append(path)
        for ax, partition in zip(axes, ['train', 'validation']):
            ax.plot(values(table, 'round'), 100 * values(table, partition + '_accuracy'),
                    color=colors[fraction], linestyle='-' if epochs == 1 else '--',
                    linewidth=1.1, label=f'C={fraction:.2f}, E={epochs}')
for ax, title in zip(axes, ['Training', 'Validation']):
    ax.set(xlabel='Communication round', ylabel='Accuracy (%)', title=title)
axes[1].legend(loc='lower right', fontsize=7.5, framealpha=.9)
save(fig, 'part1_1', 'hyperparameter_accuracy', sources)

fig, ax = plt.subplots(figsize=(6.7, 2.8), layout='constrained')
sources = []
for name, label, color in [('randomized_homogeneous', 'Randomized homogeneous', '#287c8e'),
                            ('label_sorted_heterogeneous', 'Label-segregated', '#cb773a')]:
    path = P11 / 'validation' / name / 'history.csv'
    sources.append(path)
    table = rows(path)
    ax.plot(values(table, 'round'), values(table, 'validation_accuracy') * 100,
            label=label, color=color, linewidth=1.5)
ax.set(xlabel='Communication round', ylabel='Validation accuracy (%)',
       title='Controlled partition heterogeneity', ylim=(0, 100))
ax.legend(loc='upper left', ncol=2)
save(fig, 'part1_1', 'scientific_controls', sources)

path = P12 / 'parallel/history.csv'
table = rows(path)
fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.0), layout='constrained')
for ax, metric, ylabel in zip(axes, ['accuracy', 'loss'], ['Accuracy (%)', 'Cross-entropy']):
    for partition, color in [('train', '#287c8e'), ('validation', '#cb773a')]:
        ax.plot(values(table, 'round'), values(table, partition+'_'+metric) * (100 if metric == 'accuracy' else 1),
                label=partition.capitalize(), color=color, linewidth=1.3)
    ax.set(xlabel='Communication round', ylabel=ylabel)
    ax.legend()
save(fig, 'part1_2', 'ray_convergence', [path])

event_path = P12 / 'parallel/events.csv'
timing_path = P12 / 'parallel/timings.csv'
events = [r for r in rows(event_path) if int(r['round']) == 1]
timing = rows(timing_path)[0]
start = min(float(e['created']) for e in events)
fig, ax = plt.subplots(figsize=(6.7, 2.8), layout='constrained')
for i, e in enumerate(events):
    ax.barh(i, float(e['finish']) - float(e['start']), left=float(e['start']) - start,
            height=.6, color='#3b8b98')
    ax.scatter(float(e['created']) - start, i, color='black', marker='|', s=60)
ax.axvspan(float(timing['aggregation_start']) - start,
           float(timing['aggregation_finish']) - start, color='#e58b13')
ax.axvline(float(timing['aggregation_start']) - start, color='#e58b13', linewidth=1)
ax.set(yticks=range(4), yticklabels=[f"Client {e['client_id']} | PID {e['pid']}" for e in events],
       xlabel='Seconds since first Actor initialization', title='Round 1: concurrent local training')
save(fig, 'part1_2', 'actor_execution_timeline', [event_path, timing_path])

path = P2 / 'client_robustness.csv'
table = rows(path)
scales = [0., .025, .05, .1, .2, .4, .8]
entropy = {int(r['client_id']): float(r['entropy_bits']) for r in table}
order = sorted(range(100), key=lambda cid: (entropy[cid], cid))
matrix = np.array([[100*np.mean([float(r['change']) for r in table if int(r['client_id']) == cid and float(r['b']) == b])
                    for b in scales] for cid in order])
assert np.all(matrix[:, 0] == 0)
worst = np.argsort(matrix[:, -1])[:3]
ticks_y = sorted(set(range(0, 100, 10)) | set(int(i) for i in worst))
# Keep the three largest-loss labels clear of neighboring regular ticks.
ticks_y = [i for i in ticks_y if i in worst or all(abs(i-int(j)) >= 5 for j in worst)]
fig, ax = plt.subplots(figsize=(6.7, 4.6), layout='constrained')
bound = max(1, float(np.max(np.abs(matrix))))
im = ax.imshow(matrix, aspect='auto', interpolation='nearest', cmap='RdBu', vmin=-bound, vmax=bound)
ax.set(xticks=range(7), xticklabels=[f'{b:g}' for b in scales], yticks=ticks_y,
       yticklabels=[str(order[i])+(' *' if i in worst else '') for i in ticks_y],
       xlabel='Laplace scale b', ylabel='Client ID (ascending label entropy)',
       title='Client-level robustness: round-80 mean change')
ax.grid(False)
fig.colorbar(im, ax=ax, label='Validation accuracy change (percentage points)')
save(fig, 'part2', 'dp_client_robustness_heatmap', [path])

manifest_path = REPORT / 'FIGURE_MANIFEST.json'
manifest = read(manifest_path)
replacements = {r['report_path'].removesuffix('.png')+'.pdf': r['report_path'] for r in records}
replacements['figures/part1_2/ray_accuracy.pdf'] = 'figures/part1_2/ray_convergence.png'
replacements['figures/part1_2/ray_loss.pdf'] = 'figures/part1_2/ray_convergence.png'
manifest = [r for r in manifest if r['report_path'] not in replacements and r['report_path'] not in {n['report_path'] for n in records}]
manifest.extend(records)
manifest_path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
for old in replacements:
    path = (REPORT / old).resolve()
    assert path.is_relative_to((REPORT / 'figures').resolve())
    if path.exists():
        path.unlink()  # Original experiment PDFs remain unchanged outside report/.
print(f'Regenerated {len(records)} PNG figures from saved data only.')
