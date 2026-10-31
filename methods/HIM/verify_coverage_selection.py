"""Verify archived HIM-C selections, all paired outcomes and report statistics."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
import networkx as nx
from scipy.stats import t
from dataload.dataset import DataLoad
from methods.HIM.HIM import HIM
from methods.HIM.coverage_selection import greedy_coverage,him_edge_prior
from modules.hyperbolic.lorentz_function import Lorentzian_vector,squared_lorentzian_distance
from utils.tools import MyTool as MT

ROOT=Path(__file__).resolve().parents[2];HOME=Path(__file__).resolve().parent;OUT=ROOT/'results/HIM'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ci(values):
 a=np.asarray(values,dtype=float)
 return a.mean()+np.array([-1,1])*t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a))

def main():
 torch.set_num_threads(1)
 protocol=json.loads((OUT/'coverage_protocol_2026-10-30.json').read_text());ph=sha(OUT/'coverage_protocol_2026-10-30.json')
 summary=json.loads((OUT/'coverage_summary_2026-10-30.json').read_text());assert summary['protocol']==protocol and summary['protocol_sha256']==ph
 groups={d:[] for d in protocol['datasets']};new_simulations=0
 for filename in summary['run_artifacts']:
  r=json.loads((OUT/filename).read_text());d=r['dataset'];groups[d].append(r)
  assert r['protocol_sha256']==ph and sha(ROOT/r['checkpoint'])==r['checkpoint_sha256']
  assert sha(OUT/r['source_artifact'])==r['source_artifact_sha256']
  assert r['input_hashes']['graph']==sha(HOME/f'data/{d}/{d}.G')
  assert r['input_hashes']['probabilities']==sha(HOME/f'configs/diffusion/{d}_WIC_random_probs.pkl')
  assert r['input_hashes']['selector']==sha(HOME/'methods/HIM/coverage_selection.py')
  assert r['input_hashes']['legacy_selector']==sha(HOME/'methods/HIM/seed_selection.py')
  assert r['input_hashes']['simulator']==sha(HOME/'modules/diffusion/IndependentCascadesModel.py')
  graph=DataLoad(d,'WIC').social_G;tc,mc=MT.load_model_params('HIM_WIC',d)
  proxy=graph.copy();proxy.remove_edges_from(nx.selfloop_edges(proxy))
  model=HIM(len(graph),list(dict(graph.degree()).values()),mc.c,mc.num_dim,mc.init_size,mc.num_negs,torch.device('cpu'))
  model.load_state_dict(torch.load(ROOT/r['checkpoint'],map_location='cpu',weights_only=True))
  points=Lorentzian_vector(model.embs.weight.detach());origin=torch.zeros_like(points);origin[:,0]=1
  ldo=squared_lorentzian_distance(points,origin).numpy()
  for m,detail in r['methods'].items():
   seeds=detail['seeds'];a=np.asarray(detail['activated_per_replicate'])
   assert len(seeds)==len(set(seeds))==r['seed_budget'] and 0<=min(seeds)<=max(seeds)<len(graph)
   assert len(a)==protocol['evaluation_rounds'] and np.all((a>=len(seeds))&(a<=len(graph)))
   assert np.isclose(a.mean(),detail['mean_activated']) and np.isclose(a.mean()*100/len(graph),detail['coverage_percent'])
   if m!='ASW':
    mix=protocol['him_mixture'] if m=='HIM-C' else 0
    expected=sorted(greedy_coverage(proxy,him_edge_prior(ldo,protocol['edge_prior'],mix),r['seed_budget']))
    assert seeds==expected
   else:
    assert seeds==json.loads((OUT/r['source_artifact']).read_text())['methods']['ASW']['seeds']
   if detail['reused_outcomes_from'] is None:new_simulations+=len(a)
   else:
    part,method=detail['reused_outcomes_from'].split(':');previous=json.loads((OUT/part).read_text())['methods'][method]
    assert previous['seeds']==seeds and previous['activated_per_replicate']==detail['activated_per_replicate']
  for m in ['ASW','CO']:
   a=np.asarray(r['methods']['HIM-C']['activated_per_replicate'])-np.asarray(r['methods'][m]['activated_per_replicate'])
   assert np.isclose(a.mean(),r['contrasts']['HIM-C-'+m]['mean_activated'])
   np.testing.assert_allclose(ci(a),r['contrasts']['HIM-C-'+m]['ci95_conditional'])
 for d,rows in groups.items():
  assert [r['training_seed'] for r in rows]==protocol['models'][d]
  assert len({r['checkpoint_sha256'] for r in rows})==3
  assert len({tuple(r['methods']['CO']['seeds']) for r in rows})==1
  assert len({tuple(r['methods']['CO']['activated_per_replicate']) for r in rows})==1
  for m in protocol['methods']:
   values=[r['methods'][m]['coverage_percent'] for r in rows]
   assert np.isclose(np.mean(values),summary['datasets'][d]['methods'][m]['coverage_percent_mean'])
  for m in ['ASW','CO']:
   a=[r['contrasts']['HIM-C-'+m]['mean_activated'] for r in rows]
   np.testing.assert_allclose(ci(a),summary['datasets'][d]['contrasts']['HIM-C-'+m]['ci95_t_df2'])
 report={'status':'pass','checkpoints':12,'archived_nonreused_simulations':new_simulations,
  'protocol_sha256':ph,'summary_sha256':sha(OUT/'coverage_summary_2026-10-30.json'),
  'verifier_sha256':sha(Path(__file__)),
  'checks':['Graph, probability, checkpoint, selector, simulator and source-result hashes',
   'Exact HIM-C and CO re-selection from all twelve checkpoints','Original ASW seed sets unchanged',
   'Cardinality, raw outcome bounds, means and paired intervals','Across-training intervals and deterministic CO reuse']}
 (OUT/'coverage_verification_2026-10-30.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
