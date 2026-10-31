"""Frozen three-checkpoint Ciao study for the ASW-J course extension.

Run from repository root with methods/HIM/.venv/bin/python. Original author
selection and diffusion code is unchanged. Resume reuses complete per-run files.
"""
from pathlib import Path
import hashlib
import json
import random
import time

import numpy as np
from scipy.stats import t
import torch

from dataload.dataset import DataLoad
from evaluate_checkpoint import sha256_file
from selection_provenance import validate_cached_structural
from methods.HIM.HIM import HIM
from methods.HIM.structural_selection import select_structural_asw
from modules.diffusion import DiffusionTools
from utils.runtime import Runner
from utils.tools import MyTool as MT

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/HIM'
PROTOCOL = OUT / 'structural_asw_protocol_2026-10-29.json'


def main():
    protocol = json.loads(PROTOCOL.read_text())
    protocol_hash = sha256_file(PROTOCOL)
    graph_path = ROOT / 'methods/HIM/data/ciao/ciao.G'
    probability_path = ROOT / 'methods/HIM/configs/diffusion/ciao_WIC_random_probs.pkl'
    graph = DataLoad('ciao', 'WIC').social_G
    evaluator = DiffusionTools('ciao', graph, diffusion='WIC', num_bunch=100, random_type='random')
    train_cfg, model_cfg = MT.load_model_params('HIM_WIC', 'ciao')
    n, k, rounds = len(graph), protocol['seed_budget'], protocol['rounds']
    graph_hash, probability_hash = sha256_file(graph_path), sha256_file(probability_path)
    cache, runs = {}, []
    for training_seed in protocol['training_seeds']:
        archived_path = OUT / f'paired_selection_ciao_wic_train30_seed{training_seed}_2026-10-11.json'
        archived = json.loads(archived_path.read_text())
        assert archived['graph_sha256'] == graph_hash
        assert archived['probability_file_sha256'] == probability_hash
        assert archived['evaluation_seed_base'] == protocol['evaluation_seed_base']
        assert archived['evaluation_rounds'] == rounds and archived['seed_budget'] == k
        checkpoint = ROOT / archived['checkpoint']
        assert sha256_file(checkpoint) == archived['checkpoint_sha256']
        for method in ['ASW', 'MD']:
            key = tuple(sorted(archived['methods'][method]['seeds']))
            values = archived['methods'][method]['activated_per_replicate']
            assert len(values) == rounds
            if key in cache:
                assert values == cache[key][0]
            cache[key] = (values, str(archived_path.relative_to(ROOT)) + ':' + method)
        output = OUT / f'structural_asw_ciao_seed{training_seed}_2026-10-29.json'
        model = HIM(n, list(dict(graph.degree()).values()), model_cfg.c, model_cfg.num_dim,
                    model_cfg.init_size, model_cfg.num_negs, torch.device('cpu'))
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        model.eval()
        runner = Runner(model, None, 0, torch.device('cpu'))
        selections, timings = {}, {}
        for method in ['ASW', 'MD', 'ASW-J', 'Degree', 'Random']:
            started = time.perf_counter()
            if method in ['ASW', 'MD']:
                selected = runner.seed_set_selection(k, method, train_cfg.window_weight, graph)
                assert sorted(selected) == sorted(archived['methods'][method]['seeds']), 'Legacy selector differs from archive'
            elif method == 'ASW-J':
                selected = select_structural_asw(graph, model.embs.weight, k,
                            int(train_cfg.window_weight * k), protocol['primary_alpha'])
            elif method == 'Degree':
                selected = sorted(graph, key=lambda u: (-graph.degree(u), u))[:k]
            else:
                selected = random.Random(training_seed).sample(list(range(n)), k)
            selections[method] = sorted(map(int, selected))
            timings[method] = time.perf_counter() - started
            assert len(selections[method]) == len(set(selections[method])) == k
        if output.exists():
            saved = json.loads(output.read_text())
            expected = {'protocol_sha256': protocol_hash, 'graph_sha256': graph_hash,
                'probability_file_sha256': probability_hash, 'checkpoint_sha256': sha256_file(checkpoint),
                'legacy_result_sha256': sha256_file(archived_path),
                'selector_sha256': sha256_file(Path(__file__).parent/'methods/HIM/structural_selection.py'),
                'evaluation_seed_base': protocol['evaluation_seed_base'], 'evaluation_rounds': rounds,
                'seed_budget': k, 'training_seed': training_seed, 'nodes': n}
            validate_cached_structural(saved, expected, selections, output)
            for name, detail in saved['methods'].items():
                cache[tuple(detail['seeds'])] = (detail['activated_per_replicate'], str(output.relative_to(ROOT)) + ':' + name)
            runs.append(saved)
            continue
        print(json.dumps({'training_seed': training_seed, 'phase':'selected',
              'ASW-J_changed_nodes': len(set(selections['ASW-J']) - set(selections['ASW']))}), flush=True)
        methods = {}
        for method, selected in selections.items():
            key = tuple(selected)
            started = time.perf_counter()
            reused = None
            if key in cache:
                outcomes, reused = cache[key]
            else:
                outcomes = []
                for replicate in range(rounds):
                    seed = protocol['evaluation_seed_base'] + replicate
                    random.seed(seed)
                    np.random.seed(seed)
                    mean, _ = evaluator.spread_evaluation(selected, num_rounds=1)
                    outcomes.append(int(mean))
                    if (replicate + 1) % 250 == 0:
                        print(json.dumps({'training_seed':training_seed,'method':method,'replicates':replicate+1}),flush=True)
                cache[key] = (outcomes, str(output.relative_to(ROOT)) + ':' + method)
            methods[method] = {
                'seeds':selected, 'selection_seconds':timings[method],
                'evaluation_seconds_this_run':time.perf_counter()-started,
                'reused_outcomes_from':reused, 'activated_per_replicate':outcomes,
                'mean_activated':float(np.mean(outcomes)),
                'coverage_percent':float(100*np.mean(outcomes)/n),
            }
            print(json.dumps({'training_seed':training_seed,'method':method,
                   'coverage':methods[method]['coverage_percent'],'reused':bool(reused)}),flush=True)
        difference = np.array(methods['ASW-J']['activated_per_replicate']) - np.array(methods['ASW']['activated_per_replicate'])
        se = float(difference.std(ddof=1)/np.sqrt(rounds))
        result = {
            'training_seed':training_seed,'dataset':'ciao','nodes':n,'seed_budget':k,
            'alpha':protocol['primary_alpha'],'window_size':int(train_cfg.window_weight*k),
            'checkpoint':archived['checkpoint'],'checkpoint_sha256':archived['checkpoint_sha256'],
            'graph_sha256':graph_hash,'probability_file_sha256':probability_hash,'probability_id':evaluator.prob_id,
            'evaluation_seed_base':protocol['evaluation_seed_base'],'evaluation_rounds':rounds,
            'protocol_sha256':protocol_hash,'protocol_file':str(PROTOCOL.relative_to(ROOT)),
            'selector_sha256':sha256_file(Path(__file__).parent/'methods/HIM/structural_selection.py'),
            'legacy_result_sha256':sha256_file(archived_path),'methods':methods,
            'ASW-J_changed_nodes':len(set(selections['ASW-J'])-set(selections['ASW'])),
            'ASW-J_minus_ASW_mean_activated':float(difference.mean()),
            'ASW-J_minus_ASW_ci95_conditional':(difference.mean()+np.array([-1,1])*t.ppf(.975,rounds-1)*se).tolist(),
        }
        output.write_text(json.dumps(result,indent=2)+'\n')
        runs.append(result)
    differences = np.array([r['ASW-J_minus_ASW_mean_activated'] for r in runs])
    mean, se = float(differences.mean()), float(differences.std(ddof=1)/np.sqrt(len(runs)))
    summary = {
        'protocol':protocol,'protocol_sha256':protocol_hash,'training_seeds':protocol['training_seeds'],
        'coverage_percent_mean':{m:float(np.mean([r['methods'][m]['coverage_percent'] for r in runs])) for m in protocol['methods']},
        'mean_selection_seconds':{m:float(np.mean([r['methods'][m]['selection_seconds'] for r in runs])) for m in protocol['methods']},
        'ASW-J_changed_nodes':[r['ASW-J_changed_nodes'] for r in runs],
        'ASW-J_minus_ASW_per_training_seed_activated':differences.tolist(),
        'ASW-J_minus_ASW_mean_activated':mean,
        'ASW-J_minus_ASW_ci95_t_df2_activated':(mean+np.array([-1,1])*t.ppf(.975,len(runs)-1)*se).tolist(),
        'run_artifacts':[f'structural_asw_ciao_seed{s}_2026-10-29.json' for s in protocol['training_seeds']],
        'interpretation':'Exploratory fixed-checkpoint selector extension, not a new representation model. Three embeddings, one graph/budget/probability bank. Degree is one deterministic seed set repeated, Random uses three independently sampled sets. Timing is a single wall-clock measurement, not a repeated performance benchmark. No test-stream tuning was performed.'
    }
    (OUT/'structural_asw_ciao_summary_2026-10-29.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
