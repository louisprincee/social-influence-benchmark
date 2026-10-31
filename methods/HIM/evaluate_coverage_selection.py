"""Evaluate one dataset from the frozen HIM-C study; no training or tuning."""
from pathlib import Path
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import networkx as nx
import torch
from scipy.stats import t
from dataload.dataset import DataLoad
from methods.HIM.HIM import HIM
from methods.HIM.coverage_selection import greedy_coverage,him_edge_prior,coverage_value
from modules.hyperbolic.lorentz_function import Lorentzian_vector,squared_lorentzian_distance
from modules.diffusion import DiffusionTools
from utils.runtime import Runner
from utils.tools import MyTool as MT

ROOT=Path(__file__).resolve().parents[2]
HOME=Path(__file__).resolve().parent
OUT=ROOT/'results/HIM'
PROTOCOL=OUT/'coverage_protocol_2026-10-30.json'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def ci(values):
    a=np.asarray(values,dtype=float)
    return (a.mean()+np.array([-1,1])*t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a))).tolist()


def write(path,value):
    tmp=path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def main():
    protocol=json.loads(PROTOCOL.read_text());ph=sha(PROTOCOL)
    dataset=sys.argv[1]
    assert dataset in protocol['datasets']
    torch.set_num_threads(1)
    graph=DataLoad(dataset,'WIC').social_G
    # q(v,v)=1 already accounts for choosing v. Original self-loops must not
    # create a second self-coverage attempt. Keep training/evaluation unchanged.
    loop_count=nx.number_of_selfloops(graph)
    coverage_graph=graph.copy() if loop_count else graph
    if loop_count:coverage_graph.remove_edges_from(nx.selfloop_edges(coverage_graph))
    n=len(graph)
    tc,mc=MT.load_model_params('HIM_WIC',dataset)
    evaluator=DiffusionTools(dataset,graph,diffusion='WIC',random_type='random')
    hashes={'graph':sha(HOME/f'data/{dataset}/{dataset}.G'),
        'probabilities':sha(HOME/f'configs/diffusion/{dataset}_WIC_random_probs.pkl'),
        'selector':sha(HOME/'methods/HIM/coverage_selection.py'),
        'legacy_selector':sha(HOME/'methods/HIM/seed_selection.py'),
        'simulator':sha(HOME/'modules/diffusion/IndependentCascadesModel.py')}
    cache={};runs=[]
    for seed in protocol['models'][dataset]:
        if dataset=='ciao':
            source=OUT/f'paired_selection_ciao_wic_train30_seed{seed}_2026-10-11.json'
        else:source=OUT/f'expanded_{dataset}_s{seed}_2026-10-30.json'
        archived=json.loads(source.read_text());k=archived['seed_budget']
        checkpoint=ROOT/archived['checkpoint']
        assert sha(checkpoint)==archived['checkpoint_sha256']
        if dataset=='ciao':
            assert hashes['graph']==archived['graph_sha256']
            assert hashes['probabilities']==archived['probability_file_sha256']
        else:
            assert hashes['graph']==archived['input_hashes']['graph']
            assert hashes['probabilities']==archived['input_hashes']['probabilities']
        output=OUT/f'coverage_{dataset}_s{seed}_2026-10-30.json'
        if output.exists():
            saved=json.loads(output.read_text());assert saved['protocol_sha256']==ph and saved['input_hashes']==hashes
            assert saved['checkpoint_sha256']==sha(checkpoint)
            assert saved['source_artifact']==source.name and saved['source_artifact_sha256']==sha(source)
            for method,detail in saved['methods'].items():cache[tuple(detail['seeds'])]=(detail['activated_per_replicate'],output.name+':'+method)
            runs.append(saved);print('Reused',output.name,flush=True);continue
        model=HIM(n,list(dict(graph.degree()).values()),mc.c,mc.num_dim,mc.init_size,mc.num_negs,torch.device('cpu'))
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
        points=Lorentzian_vector(model.embs.weight.detach())
        origin=torch.zeros_like(points);origin[:,0]=1
        ldo=squared_lorentzian_distance(points,origin).numpy()
        selected={};timings={};surrogates={}
        for method in protocol['methods']:
            started=time.perf_counter()
            if method=='ASW':seeds=Runner(model,None,0,torch.device('cpu')).seed_set_selection(k,'ASW',tc.window_weight,graph)
            else:
                mix=protocol['him_mixture'] if method=='HIM-C' else 0.
                probs=him_edge_prior(ldo,protocol['edge_prior'],mix)
                seeds=greedy_coverage(coverage_graph,probs,k)
                surrogates[method]=coverage_value(coverage_graph,seeds,probs)
            selected[method]=sorted(map(int,seeds));timings[method]=time.perf_counter()-started
            assert len(selected[method])==len(set(selected[method]))==k
        assert selected['ASW']==archived['methods']['ASW']['seeds']
        print(dataset,seed,'selected; HIM-C changed',len(set(selected['HIM-C'])-set(selected['ASW'])),flush=True)
        methods={}
        for method,seeds in selected.items():
            recovery=HOME/f'local_checks/coverage_{dataset}_s{seed}_{method}_2026-10-30.json'
            if recovery.exists():
                saved=json.loads(recovery.read_text())
                assert saved['protocol_sha256']==ph and saved['input_hashes']==hashes
                assert saved['checkpoint_sha256']==sha(checkpoint) and saved['seeds']==seeds
                values=saved['outcomes'];reused=saved['reused_outcomes_from']
                assert len(values)==protocol['evaluation_rounds']
            elif tuple(seeds) in cache:
                values,reused=cache[tuple(seeds)]
            else:
                values=[];reused=None
                for rep in range(protocol['evaluation_rounds']):
                    rng_seed=protocol['evaluation_seed_base']+rep
                    random.seed(rng_seed);np.random.seed(rng_seed)
                    value,_=evaluator.spread_evaluation(seeds,1)
                    values.append(int(value))
                    if (rep+1)%250==0:print(dataset,seed,method,rep+1,flush=True)
            cache[tuple(seeds)]=(values,output.name+':'+method)
            write(recovery,{'protocol_sha256':ph,'input_hashes':hashes,'checkpoint_sha256':sha(checkpoint),
                'seeds':seeds,'outcomes':values,'reused_outcomes_from':reused})
            methods[method]={'seeds':seeds,'activated_per_replicate':values,'selection_seconds':timings[method],
                'mean_activated':float(np.mean(values)),'coverage_percent':float(np.mean(values)*100/n),
                'reused_outcomes_from':reused}
            print(dataset,seed,method,'coverage',round(methods[method]['coverage_percent'],4),flush=True)
        contrasts={}
        for comparison in ['ASW','CO']:
            d=np.array(methods['HIM-C']['activated_per_replicate'])-np.array(methods[comparison]['activated_per_replicate'])
            contrasts['HIM-C-'+comparison]={'mean_activated':float(d.mean()),'ci95_conditional':ci(d)}
        result={'dataset':dataset,'training_seed':seed,'nodes':n,'edges':graph.number_of_edges(),
            'seed_budget':k,'protocol_sha256':ph,'input_hashes':hashes,'source_artifact':source.name,
            'source_artifact_sha256':sha(source),'checkpoint':str(checkpoint.relative_to(ROOT)),
            'checkpoint_sha256':sha(checkpoint),'probability_id':evaluator.prob_id,
            'methods':methods,'contrasts':contrasts,'surrogate_values':surrogates,
            'HIM-C_changed_from_ASW':len(set(selected['HIM-C'])-set(selected['ASW'])),
            'HIM-C_changed_from_CO':len(set(selected['HIM-C'])-set(selected['CO'])),
            'self_loops_ignored_in_coverage_only':loop_count}
        write(output,result);runs.append(result)
    aggregate={'dataset':dataset,'nodes':n,'seed_budget':k,
        'methods':{m:{'coverage_percent_mean':float(np.mean([r['methods'][m]['coverage_percent'] for r in runs])),
            'per_training_seed':[r['methods'][m]['coverage_percent'] for r in runs]} for m in protocol['methods']},
        'contrasts':{},'HIM-C_changed_from_ASW':[r['HIM-C_changed_from_ASW'] for r in runs],
        'HIM-C_changed_from_CO':[r['HIM-C_changed_from_CO'] for r in runs]}
    for comparison in ['ASW','CO']:
        d=[r['contrasts']['HIM-C-'+comparison]['mean_activated'] for r in runs]
        aggregate['contrasts']['HIM-C-'+comparison]={'mean_activated':float(np.mean(d)),
            'per_training_seed':d,'ci95_t_df2':ci(d)}
    write(OUT/f'coverage_summary_{dataset}_2026-10-30.json',{'protocol_sha256':ph,'result':aggregate})
    print(json.dumps(aggregate,indent=2),flush=True)


if __name__=='__main__':main()
