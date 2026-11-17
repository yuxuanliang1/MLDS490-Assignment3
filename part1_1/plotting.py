"""High-resolution PNG and vector PDF figures from saved results only."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .utils import read_json, write_json


def table(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def values(rows, field):
    return np.array([r[field] for r in rows])


def generate(root):
    root = Path(root)
    out = root/"figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "savefig.dpi": 300,
                         "pdf.fonttype": 42, "axes.grid": True, "grid.alpha": .18})
    figures = []

    def save(fig, name, description):
        fig.savefig(out/f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(out/f"{name}.pdf", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        figures.append(dict(name=name, description=description))

    summary = read_json(root/"analysis/dataset_summary.json")
    counts = read_json(root/"analysis/class_counts.json")
    labels = np.arange(62)
    fig, ax = plt.subplots(figsize=(14, 4.5), layout="constrained")
    ax.bar(labels, counts["development_pool"], color="#256d85", width=.85)
    ax.set(xticks=labels, xlabel="Supplied numeric class ID", ylabel="Number of development examples",
           title=f"Class distribution of the development pool ({summary['pool_examples']:,} examples)")
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    ax.grid(axis="x", visible=False)
    save(fig, "overall_class_distribution", "Pre-split development pool only; no held-out test labels")

    fig, axes = plt.subplots(5, 1, figsize=(14, 11), sharex=True, layout="constrained")
    for ax, row in zip(axes, summary["representative_clients"]):
        cid = row["client_id"]
        ax.bar(labels, counts["client_training"][cid], color="#328e89", width=.85)
        ax.set_title(f"Client {cid}: {row['reason']} | n={row['train_n']}, classes={row['observed_classes']}, H={row['entropy_bits']:.2f} bits", loc="left", fontsize=10)
        ax.set_ylabel("Train count")
        ax.grid(axis="x", visible=False)
    axes[-1].set(xticks=labels, xlabel="Supplied numeric class ID")
    axes[-1].tick_params(axis="x", rotation=90, labelsize=7)
    save(fig, "five_client_distributions", "Systematic selection by size extremes, diversity extremes, and median profile")

    matrix = np.array(counts["client_training"], dtype=float)
    proportions = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(14, 8), layout="constrained")
    im = ax.imshow(proportions, aspect="auto", interpolation="nearest", cmap="magma", vmin=0)
    ax.grid(False)
    ax.set(xticks=labels, yticks=np.arange(0, 100, 5), xlabel="Supplied numeric class ID", ylabel="Original client ID",
           title="Within-client training-label proportions")
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    fig.colorbar(im, ax=ax, label="Fraction of client's training examples", shrink=.8)
    save(fig, "client_class_heatmap", "Row-normalized class frequencies retain original user boundaries")

    baseline = table(root/"experiments/c10_e1_s2026/history.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, metric in zip(axes.flat, ["train_loss", "validation_loss", "train_accuracy", "validation_accuracy"]):
        acc = "accuracy" in metric
        ax.plot(values(baseline, "round"), values(baseline, metric)*(100 if acc else 1), color="#216b92", linewidth=1.8)
        ax.set(xlabel="Communication round", ylabel="Accuracy (%)" if acc else "Cross-entropy per example",
               title=metric.replace("_", " ").capitalize())
    config = read_json(root/"experiments/c10_e1_s2026/config.json")
    fig.suptitle(f"FedAvg baseline: C=0.10, E=1, SGD LR={config['lr']}, batch=32, seed=2026", fontsize=14)
    save(fig, "baseline_convergence", "Entire aggregate train/validation partitions after every updated global model; includes round zero")

    primary = [r for r in read_json(root/"comparison.json") if r["seed"] == 2026]
    colors = {.02: "#3366a5", .05: "#db8c27", .1: "#26867e"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), layout="constrained")
    cost_fig, cost_ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    for r in primary:
        h = table(root/"experiments"/r["name"]/"history.csv")
        style = dict(color=colors[r["fraction"]], linestyle="-" if r["epochs"] == 1 else "--", linewidth=1.7)
        label = f"C={r['fraction']:.2f}, E={r['epochs']}"
        for ax, metric in zip(axes, ["train_accuracy", "validation_accuracy"]):
            ax.plot(values(h,"round"), 100*values(h,metric), label=label, **style)
            ax.set(xlabel="Communication round", ylabel="Accuracy (%)", title=metric.replace("_", " ").capitalize())
        cost_ax.plot(values(h,"cumulative_client_epochs"), 100*values(h,"validation_accuracy"), label=label, **style)
    axes[1].legend(fontsize=9, loc="lower right")
    cost_ax.set(xlabel="Cumulative client-epochs", ylabel="Validation accuracy (%)", title="Accuracy versus local computation exposure")
    cost_ax.legend(fontsize=9)
    save(fig, "hyperparameter_accuracy", "Six controlled configurations, common split/initialization/LR/batch/80-round budget")
    save(cost_fig, "accuracy_vs_client_epochs", "Equal rounds have different computation: 160 to 2400 client-epochs")

    repeats = read_json(root/"repeat_candidates.json")
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    for r in repeats:
        curves = [table(root/"experiments"/f"c{int(r['fraction']*100):02d}_e{r['epochs']}_s{s}"/"history.csv") for s in [2026,2027,2028]]
        ys = np.stack([values(h,"validation_accuracy") for h in curves])*100
        x = values(curves[0],"round")
        mean, sd = ys.mean(0), ys.std(0, ddof=1)
        color = colors[r["fraction"]]
        ax.plot(x, mean, color=color, label=f"C={r['fraction']:.2f}, E={r['epochs']}: mean +/- 1 SD")
        ax.fill_between(x, mean-sd, mean+sd, color=color, alpha=.17)
    ax.set(xlabel="Communication round", ylabel="Validation accuracy (%)", title="Leading settings across three seeds (2026-2028)")
    ax.legend()
    save(fig,"seed_stability", "Pointwise sample SD across seeds; not a confidence interval")

    benefits = table(root/"validation/client_benefits.csv")
    fig, ax = plt.subplots(figsize=(11, 6.8), layout="constrained")
    sizes = values(benefits,"train_n")
    sc = ax.scatter(values(benefits,"participation"), 100*values(benefits,"benefit"),
                    c=values(benefits,"entropy_bits"), s=sizes*.6, cmap="viridis", alpha=.8,
                    edgecolors="white", linewidths=.5)
    ax.axhline(0,color="#4d5559",linestyle="--",linewidth=1,label="Zero accuracy improvement")
    ax.axvline(8,color="#b74f37",linestyle=":",linewidth=1.5,label="Expected participation = 80 x 0.10 = 8")
    fig.colorbar(sc,ax=ax,label="Training-label entropy (bits)",shrink=.8)
    for r in sorted(benefits,key=lambda r:r["benefit"])[:2] + sorted(benefits,key=lambda r:-r["benefit"])[:2]:
        ax.annotate(f"ID {int(r['client_id'])}",(r["participation"],100*r["benefit"]),xytext=(5,5),textcoords="offset points",fontsize=8)
    reference_legend = ax.legend(loc="lower left",fontsize=8)
    ax.add_artist(reference_legend)
    handles = [Line2D([],[],marker="o",linestyle="",markersize=np.sqrt(n*.6),markerfacecolor="#698080",markeredgecolor="white",label=f"n_train={n}") for n in [100,200,300]]
    ax.legend(handles=handles,title="Point area: training size",loc="lower right",fontsize=8,title_fontsize=8)
    ax.set(xlabel="Number of rounds client was selected", ylabel="Client validation accuracy change (percentage points)",
           title="Client Participation, Heterogeneity, and Learning Benefit")
    save(fig,"participation_heterogeneity_benefit", "Baseline original clients: final round minus round zero; observational associations only")

    fig,ax=plt.subplots(figsize=(9,4.5),layout="constrained")
    p=values(benefits,"participation")
    bins=np.arange(p.min()-.5,p.max()+1.5)
    ax.hist(p,bins=bins,color="#367c91",edgecolor="white")
    ax.axvline(8,color="#b74f37",linestyle="--",label="Expected count = 8")
    ax.set(xlabel="Selections in 80 rounds",ylabel="Number of clients",title="Baseline client participation frequency")
    ax.legend()
    save(fig,"participation_histogram", "Actual baseline selections, 800 total client participations")

    fig,axes=plt.subplots(1,3,figsize=(16,4.8),layout="constrained")
    groups=[[("signal_lr0.0001","Synthetic LR=0.0001"),("signal_lr0.1","Synthetic LR=0.1")],
            [("correct_labels","Correct labels"),("shuffled_labels","Shuffled train labels")],
            [("randomized_homogeneous","Randomized homogeneous"),("label_sorted_heterogeneous","Label-sorted heterogeneous")]]
    for ax,group,title in zip(axes,groups,["Controlled signal learning","Label-randomization negative control","Controlled partition heterogeneity"]):
        for name,label in group:
            h=table(root/"validation"/name/"history.csv")
            ax.plot(values(h,"round"),100*values(h,"validation_accuracy"),label=label,linewidth=1.8)
        ax.set(xlabel="Communication round",ylabel="Validation accuracy (%)",title=title,ylim=(0,103))
        ax.legend(fontsize=8)
    save(fig,"scientific_controls", "Matched controlled interventions; development/synthetic data only")

    fig,ax=plt.subplots(figsize=(10,5),layout="constrained")
    pilots=read_json(root/"pilot_decision.json")["pilots"]
    for r in pilots:
        h=table(root/"experiments"/r["name"]/"history.csv")
        ax.plot(values(h,"round"),100*values(h,"validation_accuracy"),linestyle="-" if r["epochs"]==1 else "--",label=f"LR={r['lr']}, E={r['epochs']}")
    ax.set(xlabel="Communication round",ylabel="Validation accuracy (%)",title="DeepDish5 full-pool pilot: shared learning-rate selection")
    ax.legend(fontsize=9,ncol=2)
    save(fig,"pilot_learning_rates", "Shared LR selected by mean round-12 validation across E=1 and E=3")
    write_json(out/"manifest.json",figures)
