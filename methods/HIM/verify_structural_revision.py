"""Reselect archived sets with the corrected ASW-J; never relabel old run hashes."""
from pathlib import Path
import json
import random
import torch
from dataload.dataset import DataLoad
from methods.HIM.HIM import HIM
from methods.HIM.structural_selection import select_structural_asw
from utils.runtime import Runner
from utils.tools import MyTool as MT
from selection_provenance import ROOT, HOME, SELECTOR, FROZEN_SELECTOR, REVISION_AUDIT, sha


def main():
    torch.set_num_threads(1)
    historical = sha(FROZEN_SELECTOR)
    records = {}
    for dataset, seeds in [('cora_ml', [142,143,144]), ('power_grid', [142,143,144]),
                           ('facebook', [142,143,144]), ('ciao', [42,43,44])]:
        graph = DataLoad(dataset, 'WIC').social_G
        tc, mc = MT.load_model_params('HIM_WIC', dataset)
        graph_hash = sha(HOME/f'data/{dataset}/{dataset}.G')
        probability_hash = sha(HOME/f'configs/diffusion/{dataset}_WIC_random_probs.pkl')
        for seed in seeds:
            name = (f'structural_asw_ciao_seed{seed}_2026-10-29.json' if dataset == 'ciao'
                    else f'expanded_{dataset}_s{seed}_2026-10-30.json')
            path = ROOT/'results/HIM'/name
            run = json.loads(path.read_text())
            stored = run.get('selector_sha256', run.get('input_hashes', {}).get('selector'))
            assert stored == historical
            if dataset == 'ciao':
                assert run['graph_sha256'] == graph_hash and run['probability_file_sha256'] == probability_hash
                legacy = ROOT/f'results/HIM/paired_selection_ciao_wic_train30_seed{seed}_2026-10-11.json'
                assert run['legacy_result_sha256'] == sha(legacy)
            else:
                assert run['input_hashes']['graph'] == graph_hash
                assert run['input_hashes']['probabilities'] == probability_hash
            checkpoint = ROOT/run['checkpoint']; assert sha(checkpoint) == run['checkpoint_sha256']
            model = HIM(len(graph), list(dict(graph.degree()).values()), mc.c, mc.num_dim,
                        mc.init_size, mc.num_negs, torch.device('cpu'))
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)); model.eval()
            runner = Runner(model, None, 0, torch.device('cpu')); k = run['seed_budget']
            selected = {
                'ASW': sorted(map(int, runner.seed_set_selection(k, 'ASW', tc.window_weight, graph))),
                'MD': sorted(map(int, runner.seed_set_selection(k, 'MD', tc.window_weight, graph))),
                'ASW-J': sorted(select_structural_asw(graph, model.embs.weight, k, int(tc.window_weight*k), .5)),
                'Degree': sorted(sorted(graph, key=lambda u: (-graph.degree(u), u))[:k]),
                'Random': sorted(random.Random(seed).sample(list(range(len(graph))), k))}
            for method, values in selected.items():
                assert values == run['methods'][method]['seeds'], (dataset, seed, method)
            records[str(path.relative_to(ROOT))] = {
                'artifact_sha256': sha(path), 'checkpoint_sha256': sha(checkpoint),
                'graph_sha256': graph_hash, 'probability_sha256': probability_hash,
                'dataset': dataset, 'training_seed': seed, 'primary_selection_unchanged': True,
                'methods_reselected': list(selected)}
    report = {'status': 'pass', 'current_selector_sha256': sha(SELECTOR),
              'historical_selector_sha256': historical, 'verifier_sha256': sha(Path(__file__)),
              'records': records, 'checkpoints': len(records), 'new_diffusion_simulations': 0,
              'interpretation': 'Historical run files and hashes are retained unchanged. Corrected graph-degree normalization and open-neighborhood handling select exactly the same primary seed sets on these 12 checkpoints. Existing outcomes remain applicable to those identical sets, not to arbitrary future revisions or diagnostic alpha values.'}
    REVISION_AUDIT.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status':'pass','checkpoints':len(records),'primary_sets_unchanged':True,
                      'current_selector_sha256':sha(SELECTOR),'new_diffusion_simulations':0}, indent=2))


if __name__ == '__main__':
    main()
