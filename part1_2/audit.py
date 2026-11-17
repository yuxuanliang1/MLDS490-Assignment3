"""Audit completed artifacts without training or accessing test examples."""
from pathlib import Path
from collections import Counter
import csv
from part1_1.utils import read_json,write_json,sha256,load_checkpoint

def main():
    root=Path('part1_2_outputs')
    preserved=read_json('part1_1_deepdish5_outputs/source_manifest.json')
    for path,digest in {**preserved['source'],**preserved['supplied_inputs']}.items():assert sha256(path)==digest,path
    split=read_json('part1_1_deepdish5_outputs/splits.json')
    with (root/'parallel/selections.csv').open() as f:selections=list(csv.DictReader(f))
    assert Counter(int(r['round']) for r in selections)==Counter({r:4 for r in range(1,81)})
    for r in range(1,81):assert len({x['client_id'] for x in selections if int(x['round'])==r})==4
    for x in selections:assert int(x['n_train'])==len(split['clients'][x['client_id']]['train'])
    with (root/'parallel/history.csv').open() as f:history=list(csv.DictReader(f))
    best=max(history,key=lambda r:(float(r['validation_accuracy']),-float(r['validation_loss']),-int(r['round'])))
    selection=read_json(root/'selection.json');test=read_json(root/'test.json')
    assert sha256(selection['checkpoint'])==selection['sha256']
    assert int(best['round'])==selection['round']==load_checkpoint(selection['checkpoint'])['round']
    assert sha256(root/'selection.json')==test['selection_sha256']
    assert all(r['passed'] for r in read_json(root/'validation.json'))
    names={r['test'] for r in read_json(root/'validation.json')}
    assert {'A_D_equivalence_cpu','A_D_equivalence_cuda:0','F_sampling','D_synthetic_aggregation','G_exception','G_nan','G_oom','G_interrupt'}.issubset(names)
    assert read_json(root/'extra_validation.json')['passed']
    placements=[p for b in read_json(root/'benchmarks.json') if b['mode']=='cuda:0' for p in b['placements']]
    assert placements and all(p['gpu']=='NVIDIA GeForce RTX 5060 Ti' and p['torch_threads']==1 for p in placements)
    result=dict(passed=True,part1_1_source_and_supplied_data_preserved=True,
                exactly_four_unique_selected_clients_in_all_80_rounds=True,training_weights_match_fixed_split=True,
                checkpoint_matches_validation_rule=True,test_selection_hash_matches=True,
                GPU_benchmark_placement_verified=True,extra_validation_passed=True,
                source_sha256={str(p):sha256(p) for p in Path('part1_2').rglob('*') if p.is_file() and '__pycache__' not in p.parts})
    write_json(root/'final_audit.json',result)
    (root/'INDEX.md').write_text('# Part 1.2 results\n\n[Report notes](REPORT_NOTES.md) | [README](../part1_2/README.md) | [Final audit](final_audit.json)\n\n[Completion summary](completion.json) | [Scientific validation](validation.json) | [Extra edge cases](extra_validation.json) | [Benchmark](performance_summary.csv) | [Concurrency and lifecycle](lifecycle_concurrency_audit.json) | [Final test](test.json) | [Best checkpoint](parallel/best.pt)\n\n[Figure manifest](figures/manifest.json). Five PNG/PDF figure pairs are in figures/.\n',encoding='utf-8')
    print('PART 1.2 FINAL AUDIT PASSED',flush=True)

if __name__=='__main__':main()
