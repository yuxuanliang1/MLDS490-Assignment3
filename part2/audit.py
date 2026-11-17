import csv
from collections import Counter
from pathlib import Path
import numpy as np
from part1_1.utils import read_json,write_json,sha256,load_checkpoint
from .workflow import ROOT,SCALES,SEEDS,protected_hashes,name


def table(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def audit(pretest=False):
    protected=read_json(ROOT/'protected_snapshot.json');current=protected_hashes()
    assert protected==current,'An earlier source/result/input file changed'
    validations=read_json(ROOT/'validation/results.json')
    required={'A_Laplace_distribution','B_pixel_independence','C_zero_noise_equivalence','C_full_zero_control','D_data_isolation','E_controlled_noise_degradation',
              'F_image_distortion','G_client_noise_sensitivity','H_high_noise_stability','I_noise_reproducibility_and_edges','I_training_reproducibility','J_fixed_vs_resampled','edge_tiny_and_resume'}
    assert required.issubset({r['test'] for r in validations})
    assert all(r['passed'] for r in validations if r['gating'])
    assert all(r['passed'] for r in validations),'An empirical criterion failed; inspect, do not hide it'
    manifest=read_json('part1_1_deepdish5_outputs/splits.json');ntrain={int(k):len(v['train']) for k,v in manifest['clients'].items()}
    rows=read_json(ROOT/'comparison.json');assert len(rows)==21
    for seed in SEEDS:
        group=[r for r in rows if r['seed']==seed]
        assert len({r['initial_digest'] for r in group})==1
        selections_hash=set()
        for b in SCALES:
            out=ROOT/'experiments'/name(b,seed);h=read_json(out/'history.json')
            assert [r['round'] for r in h]==list(range(81))
            assert all(r['train_n']==25418 and r['validation_n']==6407 for r in h)
            assert all(np.isfinite(r[k]) for r in h for k in ['train_loss','validation_loss','train_accuracy','validation_accuracy'])
            selections=table(out/'selections.csv');assert len(selections)==800
            for r in range(1,81):
                s=[v for v in selections if int(v['round'])==r]
                assert len(s)==10 and len({v['client_id'] for v in s})==10
                assert all(int(v['n_train'])==ntrain[int(v['client_id'])] for v in s)
            selections_hash.add(sha256(out/'selections.csv'))
            best=max(h,key=lambda r:(r['validation_accuracy'],-r['validation_loss'],-r['round']))
            assert load_checkpoint(out/'best.pt')['round']==best['round']
            assert load_checkpoint(out/'last.pt')['round']==80
        assert len(selections_hash)==1
    result=dict(passed=True,protected_files_unchanged=len(protected),main_runs=21,
          all_81_round_histories_and_full_clean_pool_metrics_verified=True,all_selection_counts_and_training_weights_verified=True,
          identical_weights_and_sampling_within_each_seed=True,validation_investigations=len(validations),pretest=pretest)
    if not pretest:
        frozen=read_json(ROOT/'frozen_protocol.json');scores=read_json(ROOT/'test_results.json')
        assert sha256(ROOT/'frozen_protocol.json')==scores['protocol_sha256']
        assert len(scores['results'])==21
        for c in frozen['final_checkpoints']:assert sha256(c['path'])==c['sha256']
        assert all(s['round']==80 and s['accuracy']==s['correct']/s['n'] and s['n']==3621 for s in scores['results'])
        figures=read_json(ROOT/'figures/manifest.json');assert len(figures)==8
        for f in figures:
            for ext in ['png','pdf']:assert (ROOT/'figures'/f"{f['name']}.{ext}").stat().st_size>1000
        assert (ROOT/'REPORT_NOTES.md').stat().st_size>15000
        result.update(test_protocol_and_checkpoint_hashes_verified=True,all_21_final_test_scores_verified=True,figure_pairs=8)
    result['source_sha256']={str(p):sha256(p) for p in sorted(Path('part2').rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    write_json(ROOT/('pretest_audit.json' if pretest else 'final_audit.json'),result)
    if not pretest:
        text=['# Assignment 3 Part 2 artifacts','',
              '[Report notes](REPORT_NOTES.md) | [README](../part2/README.md) | [Final audit](final_audit.json)','',
              '- [All 21 final-round results](final_results.csv)',
              '- [Across-seed means and sample SD](seed_summary.csv)',
              '- [Frozen validation-based recommendation and scoring protocol](frozen_protocol.json)',
              '- [Scientific validation A-J and edge cases](validation/results.json)',
              '- [All test scores and separately labeled Question1 checkpoint](test_results.json)',
              '- [Environment and GPU UUID](environment.json)',
              '- [Preservation checksums for prior work](protected_snapshot.json)','',
              '| Figure | PNG | PDF |','|---|---|---|']
        for f in figures:text.append(f"| {f['name']} | [PNG](figures/{f['name']}.png) | [PDF](figures/{f['name']}.pdf) |")
        text+=['','Individual configurations, per-round CSV/JSON histories, noise hashes/diagnostics, selections, initial/best/resumable final checkpoints and per-client final validation metrics are in experiments/.','',
               'Baseline/recommendation use validation only. Test comparisons always use fixed final-round checkpoints. No formal epsilon-DP guarantee is claimed.','']
        (ROOT/'INDEX.md').write_text('\n'.join(text),encoding='utf-8')
    print('PART2 AUDIT PASSED',result['main_runs'],'runs;',len(protected),'protected files unchanged',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--pretest',action='store_true');args=p.parse_args();audit(args.pretest)
