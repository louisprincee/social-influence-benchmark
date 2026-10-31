"""Recompute report statistics and verify all archived reproduction inputs."""
from pathlib import Path
import hashlib
import json
import random

import numpy as np
from scipy.stats import t
from selection_provenance import validate_selector_fingerprint

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/HIM'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def ci(a):
 a=np.asarray(a,dtype=float)
 return a.mean()+np.array([-1,1])*t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a))

def main():
 summary=json.loads((OUT/'expanded_summary_2026-10-30.json').read_text())
 protocol=json.loads((OUT/'expanded_protocol_2026-10-30.json').read_text())
 ph=sha(OUT/'expanded_protocol_2026-10-30.json')
 assert summary['protocol']==protocol and summary['protocol_sha256']==ph
 groups={d:[] for d in protocol['datasets']}
 simulations=0
 for filename in summary['run_artifacts']:
  r=json.loads((OUT/filename).read_text());d=r['dataset'];groups[d].append(r)
  assert r['protocol_sha256']==ph and sha(ROOT/r['checkpoint'])==r['checkpoint_sha256']
  assert r['input_hashes']['graph']==sha(ROOT/f'methods/HIM/data/{d}/{d}.G')
  assert r['input_hashes']['probabilities']==sha(ROOT/f'methods/HIM/configs/diffusion/{d}_WIC_random_probs.pkl')
  assert r['input_hashes']['config']==sha(ROOT/'methods/HIM/configs/HIM_WIC.yaml')
  assert r['input_hashes']['training_script']==sha(ROOT/'methods/HIM/run.py')
  validate_selector_fingerprint(r['input_hashes']['selector'], OUT/filename)
  assert r['training_metadata']['training_instances']==30
  assert r['training_metadata']['seed_count']==r['seed_budget']==int(r['nodes']*.1)
  assert r['training_metadata']['training_seed']==r['training_seed']
  cache=ROOT/'methods/HIM'/r['training_metadata']['training_cache']
  assert sha(cache)==r['training_metadata']['training_cache_sha256']
  for m,detail in r['methods'].items():
   seeds=detail['seeds'];a=np.asarray(detail['activated_per_replicate'])
   assert len(seeds)==len(set(seeds))==r['seed_budget']
   assert min(seeds)>=0 and max(seeds)<r['nodes']
   assert len(a)==protocol['evaluation_rounds'] and np.all((a>=len(seeds))&(a<=r['nodes']))
   assert np.isclose(a.mean(),detail['mean_activated'])
   assert np.isclose(a.mean()*100/r['nodes'],detail['coverage_percent'])
   if detail['reused_outcomes_from'] is None:simulations+=len(a)
   else:
    part,method=detail['reused_outcomes_from'].split(':')
    prev=json.loads((OUT/part).read_text())['methods'][method]
    assert prev['seeds']==seeds and prev['activated_per_replicate']==detail['activated_per_replicate']
   if m=='Random':assert seeds==sorted(random.Random(r['training_seed']).sample(list(range(r['nodes'])),r['seed_budget']))
  assert r['alpha_selection_diagnostic']['0.0']['seeds']==r['methods']['ASW']['seeds']
  assert r['alpha_selection_diagnostic']['0.5']['seeds']==r['methods']['ASW-J']['seeds']
  for m in ['MD','ASW-J']:
   a=np.asarray(r['methods'][m]['activated_per_replicate'])-np.asarray(r['methods']['ASW']['activated_per_replicate'])
   assert np.isclose(a.mean(),r['differences'][m+'-ASW']['mean_activated'])
   np.testing.assert_allclose(ci(a),r['differences'][m+'-ASW']['ci95_conditional'])
 for d,rows in groups.items():
  assert sorted(r['training_seed'] for r in rows)==protocol['training_seeds']
  assert len({r['checkpoint_sha256'] for r in rows})==3
  assert len({r['training_metadata']['training_cache_sha256'] for r in rows})==3
  assert len({tuple(r['methods']['Degree']['seeds']) for r in rows})==1
  assert len({tuple(r['methods']['Degree']['activated_per_replicate']) for r in rows})==1
  for m in protocol['methods']:
   a=[r['methods'][m]['coverage_percent'] for r in rows]
   np.testing.assert_allclose(a,summary['datasets'][d]['methods'][m]['coverage_percent_per_training_seed'])
   assert np.isclose(np.mean(a),summary['datasets'][d]['methods'][m]['coverage_percent_mean'])
  for m in ['MD','ASW-J']:
   a=[r['differences'][m+'-ASW']['mean_activated'] for r in rows]
   np.testing.assert_allclose(ci(a),summary['datasets'][d]['contrasts'][m+'-ASW']['ci95_t_df2'])
 report={'status':'pass','training_runs':9,'datasets':protocol['datasets'],
         'evaluation_rounds_per_set':500,'archived_nonreused_simulations':simulations,
         'protocol_sha256':ph,'summary_sha256':sha(OUT/'expanded_summary_2026-10-30.json'),
         'verifier_sha256':sha(Path(__file__)),
         'checks':['Input, cache and checkpoint hashes','Nine independent training seeds and caches',
                   'Exact seed budgets','Raw outcomes and means','Conditional and across-training intervals',
                   'Random seed reproducibility','Identical-set reuse provenance','Alpha=0 and alpha=.5 identities']}
 (OUT/'expanded_verification_2026-10-30.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))

if __name__=='__main__':main()
