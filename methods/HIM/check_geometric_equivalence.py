"""Ensure the selector refactor alone does not change these fixed seed sets."""
from pathlib import Path
import json
import torch
from dataload.dataset import DataLoad
from methods.HIM.HIM import HIM
import methods.HIM.structural_selection as structural
from utils.runtime import Runner
from utils.tools import MyTool as MT

ROOT=Path(__file__).resolve().parents[2]
HOME=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(1)
 records=[]
 inputs=[]
 for dataset in ['cora_ml','power_grid','facebook']:
  for seed in [142,143,144]:
   manifest=json.loads((HOME/f'local_checks/expanded_{dataset}_s{seed}_2026-10-30_training.json').read_text())
   inputs.append((dataset,seed,ROOT/manifest['checkpoint'],int({'cora_ml':2810,'power_grid':4941,'facebook':5908}[dataset]*.1)))
 for seed in [42,43,44]:
  archived=json.loads((ROOT/f'results/HIM/paired_selection_ciao_wic_train30_seed{seed}_2026-10-11.json').read_text())
  inputs.append(('ciao',seed,ROOT/archived['checkpoint'],366))
 for dataset,seed,checkpoint,k in inputs:
  graph=DataLoad(dataset,'WIC').social_G;tc,mc=MT.load_model_params('HIM_WIC',dataset)
  model=HIM(len(graph),list(dict(graph.degree()).values()),mc.c,mc.num_dim,mc.init_size,mc.num_negs,torch.device('cpu'))
  model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
  original=set(Runner(model,None,0,torch.device('cpu')).seed_set_selection(k,'ASW',tc.window_weight,graph))
  saved=structural.blend_overlap_weights
  structural.blend_overlap_weights=lambda geometric,overlap,alpha:geometric
  try:control=structural.select_structural_asw(graph,model.embs.weight,k,int(k*tc.window_weight),.5)
  finally:structural.blend_overlap_weights=saved
  changed=len(control-original)
  records.append({'dataset':dataset,'training_seed':seed,'budget':k,'geometric_only_changed_nodes':changed})
  assert changed==0,records[-1]
 report={'status':'pass','records':records,'interpretation':'On all 12 study checkpoints, the refactored selector with structural blending forced off selects exactly the legacy ASW set. This controls numerical/refactoring effects for these checkpoints, not every possible graph or embedding.'}
 (ROOT/'results/HIM/geometric_equivalence_2026-10-30.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
