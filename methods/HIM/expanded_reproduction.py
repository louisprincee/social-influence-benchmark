"""Three original networks, three fresh training seeds and frozen selector controls.

Run from repository root. Uses the unchanged author training and simulator.
Protocol is frozen before training; existing outcomes are hash-checked on resume.
"""
from pathlib import Path
import hashlib
import json
import os
import random
import subprocess
import sys
import time

from selection_provenance import validate_input_hashes

ROOT = Path(__file__).resolve().parents[2]
HOME = Path(__file__).resolve().parent
OUT = ROOT / 'results/HIM'
PROTOCOL = OUT / 'expanded_protocol_2026-10-30.json'


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_training(protocol):
    """Spawn before initializing OpenMP/PyTorch, and repair a known config gap."""
    # Author configs are local ignored downloads. Make this documented repair
    # repeatable on another checkout, rather than depending on our local YAML.
    import yaml
    config_path = HOME / 'configs/HIM_WIC.yaml'
    configs = yaml.safe_load(config_path.read_text())
    if 'sample_ratio' not in configs['power_grid']['train']:
        configs['power_grid']['train']['sample_ratio'] = .2
        config_path.write_text(yaml.safe_dump(configs, sort_keys=False))
    counts = {'cora_ml':2810, 'power_grid':4941, 'facebook':5908}
    ph = sha256_file(PROTOCOL)
    for dataset in protocol['datasets']:
        inputs = {
            'graph':sha256_file(HOME/f'data/{dataset}/{dataset}.G'),
            'probabilities':sha256_file(HOME/f'configs/diffusion/{dataset}_WIC_random_probs.pkl'),
            'config':sha256_file(HOME/'configs/HIM_WIC.yaml'),
            'selector':sha256_file(HOME/'methods/HIM/structural_selection.py'),
            'training_script':sha256_file(HOME/'run.py'),
        }
        for seed in protocol['training_seeds']:
            stem=f'expanded_{dataset}_s{seed}_2026-10-30'
            manifest=HOME/'local_checks'/(stem+'_training.json')
            manifest.parent.mkdir(exist_ok=True)
            if manifest.exists():
                saved=json.loads(manifest.read_text())
                assert saved['protocol_sha256']==ph
                validate_input_hashes(saved['input_hashes'], inputs, OUT/(stem+'.json'))
                assert sha256_file(ROOT/saved['checkpoint'])==saved['checkpoint_sha256']
                continue
            folder=HOME/'save'/dataset
            before=set(folder.glob('*')) if folder.exists() else set()
            command=[sys.executable,'run.py','--data_name',dataset,'--diffusion_model','WIC',
                '--seed_ratio',str(protocol['budget_percent']),'--seed_num',str(int(counts[dataset]*protocol['budget_percent']/100)),
                '--num_instances',str(protocol['training_instances']),'--epochs',str(protocol['max_epochs']),
                '--R',str(protocol['validation_rounds']),'--patience',str(protocol['patience']),
                '--seed',str(seed),'--method','ASW']
            print('Training',dataset,seed,flush=True)
            started=time.perf_counter()
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            with (HOME/'local_checks'/(stem+'.log')).open('w') as log:
                subprocess.run(command,cwd=HOME,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            created=set(folder.glob('*'))-before
            matching=[]
            for directory in created:
                meta_path=directory/'seeds_best_epoch.json'
                if meta_path.exists() and json.loads(meta_path.read_text()).get('training_seed')==seed:
                    matching.append(directory)
            assert len(matching)==1, 'Training output must match the requested training seed'
            checkpoint=matching[0]/'model_best_epoch.pt'
            assert checkpoint.exists()
            saved={'protocol_sha256':ph,'input_hashes':inputs,'command':command,
                'checkpoint':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':sha256_file(checkpoint),
                'training_seconds':time.perf_counter()-started}
            manifest.write_text(json.dumps(saved,indent=2)+'\n')


def interval(values):
    import numpy as np
    from scipy.stats import t
    a = np.asarray(values, dtype=float)
    if len(a) < 2:
        raise ValueError('At least two independent units are needed')
    return (a.mean() + np.array([-1, 1]) * t.ppf(.975, len(a)-1)
            * a.std(ddof=1) / np.sqrt(len(a))).tolist()


def main():
    protocol = json.loads(PROTOCOL.read_text())
    requested = sys.argv[1] if len(sys.argv) > 1 else None
    if requested is not None and requested not in protocol['datasets']:
        raise ValueError('Optional argument must name one protocol dataset')
    datasets = [requested] if requested else protocol['datasets']
    requested_seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if requested_seed is not None and requested_seed not in protocol['training_seeds']:
        raise ValueError('Optional seed must belong to the frozen protocol')
    training_seeds = [requested_seed] if requested_seed is not None else protocol['training_seeds']
    prepare_training(dict(protocol, datasets=datasets, training_seeds=training_seeds))
    global np, torch, t, DataLoad, HIM, select_structural_asw, DiffusionTools, Runner, MT
    import numpy as np
    import torch
    from scipy.stats import t
    from dataload.dataset import DataLoad
    from methods.HIM.HIM import HIM
    from methods.HIM.structural_selection import select_structural_asw
    from modules.diffusion import DiffusionTools
    from utils.runtime import Runner
    from utils.tools import MyTool as MT
    torch.set_num_threads(1)
    protocol_hash = sha256_file(PROTOCOL)
    runs = []
    for dataset in datasets:
        graph = DataLoad(dataset, 'WIC').social_G
        n = len(graph)
        k = int(n * protocol['budget_percent'] / 100)
        tc, mc = MT.load_model_params('HIM_WIC', dataset)
        evaluator = DiffusionTools(dataset, graph, diffusion='WIC', random_type='random')
        inputs = {
            'graph': sha256_file(HOME / f'data/{dataset}/{dataset}.G'),
            'probabilities': sha256_file(HOME / f'configs/diffusion/{dataset}_WIC_random_probs.pkl'),
            'config': sha256_file(HOME / 'configs/HIM_WIC.yaml'),
            'selector': sha256_file(HOME / 'methods/HIM/structural_selection.py'),
            'training_script': sha256_file(HOME / 'run.py'),
        }
        outcome_cache = {}
        for training_seed in training_seeds:
            stem = f'expanded_{dataset}_s{training_seed}_2026-10-30'
            result_path = OUT / (stem + '.json')
            if result_path.exists():
                saved = json.loads(result_path.read_text())
                assert saved['protocol_sha256'] == protocol_hash
                validate_input_hashes(saved['input_hashes'], inputs, result_path)
                assert sha256_file(ROOT / saved['checkpoint']) == saved['checkpoint_sha256']
                for method, detail in saved['methods'].items():
                    outcome_cache[tuple(detail['seeds'])] = (detail['activated_per_replicate'], result_path.name+':'+method)
                runs.append(saved)
                print('Reused', stem, flush=True)
                continue
            training_manifest = HOME / 'local_checks' / (stem + '_training.json')
            training_manifest.parent.mkdir(exist_ok=True)
            training = json.loads(training_manifest.read_text())
            assert training['protocol_sha256'] == protocol_hash
            validate_input_hashes(training['input_hashes'], inputs, result_path)
            checkpoint = ROOT / training['checkpoint']
            assert sha256_file(checkpoint) == training['checkpoint_sha256']
            metadata = json.loads((checkpoint.parent/'seeds_best_epoch.json').read_text())
            assert metadata['training_seed'] == training_seed and metadata['training_instances'] == 30
            model = HIM(n, list(dict(graph.degree()).values()), mc.c, mc.num_dim,
                        mc.init_size, mc.num_negs, torch.device('cpu'))
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
            model.eval()
            runner = Runner(model, None, 0, torch.device('cpu'))
            selected, timings = {}, {}
            for method in protocol['methods']:
                started = time.perf_counter()
                if method in ['ASW', 'MD']:
                    seeds = runner.seed_set_selection(k, method, tc.window_weight, graph)
                elif method == 'ASW-J':
                    seeds = select_structural_asw(graph, model.embs.weight, k,
                                                 int(tc.window_weight*k), protocol['alpha'])
                elif method == 'Degree':
                    seeds = sorted(graph, key=lambda u: (-graph.degree(u), u))[:k]
                else:
                    seeds = random.Random(training_seed).sample(list(range(n)), k)
                selected[method] = sorted(map(int, seeds))
                timings[method] = time.perf_counter()-started
                assert len(seeds) == len(set(seeds)) == k
            assert selected['ASW'] == sorted(metadata['seeds'])
            # Alpha is diagnostic only. Never pick a setting using these outcomes.
            sensitivity = {}
            for alpha in protocol['diagnostic_alphas']:
                seeds = select_structural_asw(graph, model.embs.weight, k, int(tc.window_weight*k), alpha)
                sensitivity[str(alpha)] = {'changed_from_ASW': len(set(seeds)-set(selected['ASW'])), 'seeds': sorted(map(int, seeds))}
            methods = {}
            for method, seeds in selected.items():
                key = tuple(seeds)
                recovery_path = HOME / 'local_checks' / (stem + '_eval_' + method + '.json')
                if recovery_path.exists():
                    recovery = json.loads(recovery_path.read_text())
                    assert recovery['protocol_sha256'] == protocol_hash
                    assert recovery['checkpoint_sha256'] == sha256_file(checkpoint)
                    assert recovery['input_hashes'] == inputs and recovery['seeds'] == seeds
                    assert len(recovery['outcomes']) == protocol['evaluation_rounds']
                    outcome_cache[key] = (recovery['outcomes'], recovery['reused_outcomes_from'])
                reused = None
                if key in outcome_cache:
                    outcomes, reused = outcome_cache[key]
                else:
                    outcomes = []
                    for rep in range(protocol['evaluation_rounds']):
                        seed = protocol['evaluation_seed_base']+rep
                        random.seed(seed)
                        np.random.seed(seed)
                        value, _ = evaluator.spread_evaluation(seeds, 1)
                        outcomes.append(int(value))
                    outcome_cache[key] = (outcomes, result_path.name+':'+method)
                recovery = {'protocol_sha256':protocol_hash, 'checkpoint_sha256':sha256_file(checkpoint),
                    'input_hashes':inputs,'seeds':seeds,'outcomes':outcomes,'reused_outcomes_from':reused}
                recovery_tmp = recovery_path.with_suffix(f'.{os.getpid()}.tmp')
                recovery_tmp.write_text(json.dumps(recovery)+'\n')
                recovery_tmp.replace(recovery_path)
                methods[method] = {
                    'seeds': seeds, 'selection_seconds': timings[method],
                    'activated_per_replicate': outcomes, 'reused_outcomes_from': reused,
                    'mean_activated': float(np.mean(outcomes)),
                    'coverage_percent': float(np.mean(outcomes)*100/n),
                }
                print(dataset, training_seed, method, round(methods[method]['coverage_percent'],4), flush=True)
            differences = {}
            for method in ['MD', 'ASW-J']:
                d = np.array(methods[method]['activated_per_replicate'])-np.array(methods['ASW']['activated_per_replicate'])
                differences[method+'-ASW'] = {'mean_activated':float(d.mean()), 'ci95_conditional':interval(d)}
            result = {
                'dataset':dataset, 'nodes':n, 'edges':graph.number_of_edges(), 'seed_budget':k,
                'training_seed':training_seed, 'training':training, 'best_epoch':metadata['epoch'],
                'checkpoint':str(checkpoint.relative_to(ROOT)), 'checkpoint_sha256':sha256_file(checkpoint),
                'training_metadata':metadata, 'input_hashes':inputs, 'protocol_sha256':protocol_hash,
                'probability_id':evaluator.prob_id,
                'probability_range':[min(evaluator.true_probs.values()), max(evaluator.true_probs.values())],
                'methods':methods, 'differences':differences, 'alpha_selection_diagnostic':sensitivity,
                'ASW-J_changed_nodes':len(set(selected['ASW-J'])-set(selected['ASW'])),
            }
            temporary = result_path.with_suffix(f'.{os.getpid()}.tmp')
            temporary.write_text(json.dumps(result,indent=2)+'\n')
            temporary.replace(result_path)
            runs.append(result)
    if requested_seed is not None:
        print('Single-run artifact completed', requested, requested_seed, flush=True)
        return
    aggregate = {}
    for dataset in datasets:
        rows = [r for r in runs if r['dataset']==dataset]
        aggregate[dataset] = {
            'nodes':rows[0]['nodes'], 'edges':rows[0]['edges'], 'seed_budget':rows[0]['seed_budget'],
            'best_epochs':[r['best_epoch'] for r in rows],
            'methods':{m:{
                'coverage_percent_mean':float(np.mean([r['methods'][m]['coverage_percent'] for r in rows])),
                'coverage_percent_per_training_seed':[r['methods'][m]['coverage_percent'] for r in rows],
            } for m in protocol['methods']},
            'ASW-J_changed_nodes':[r['ASW-J_changed_nodes'] for r in rows],
            'contrasts':{},
            'alpha_changed_counts':{str(a):[r['alpha_selection_diagnostic'][str(a)]['changed_from_ASW'] for r in rows] for a in protocol['diagnostic_alphas']},
        }
        for m in ['MD','ASW-J']:
            d=[r['differences'][m+'-ASW']['mean_activated'] for r in rows]
            aggregate[dataset]['contrasts'][m+'-ASW']={'mean_activated':float(np.mean(d)), 'per_training_seed':d,'ci95_t_df2':interval(d)}
    summary={'protocol':protocol,'protocol_sha256':protocol_hash,'datasets':aggregate,
        'run_artifacts':[f"expanded_{r['dataset']}_s{r['training_seed']}_2026-10-30.json" for r in runs],
        'interpretation':'Author-code WIC reproduction on three original networks at 10% budget. Not a full paper replication. Frozen author probability banks differ by dataset and from the paper description. Alpha fixed at .5; other alphas only seed-set diagnostics. Three independent training runs per network, 500 paired simulation replicates per set. No cross-network averaging. Degree is one deterministic set per network, Random has three sampled sets. Paired initial RNG streams do not ensure edge-level coupling.'}
    summary_name = f'expanded_summary_{requested}_2026-10-30.json' if requested else 'expanded_summary_2026-10-30.json'
    (OUT/summary_name).write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(aggregate,indent=2),flush=True)


if __name__=='__main__':
    main()
