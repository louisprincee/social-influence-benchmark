"""Course extension: use a frozen HIM prior in cumulative one-hop coverage.

This is a probabilistic coverage surrogate, not the true multi-hop diffusion
objective. Greedy coverage and lazy evaluation are established techniques.
The selector receives no evaluation probability bank or test cascades.
"""
from __future__ import annotations

import heapq
import math
import numpy as np


def him_edge_prior(ldo, probability=.1, mixture=.5):
    """Map lower LDO to a larger fixed edge prior, averaging tied ranks."""
    scores=np.asarray(ldo,dtype=float)
    if scores.ndim!=1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('LDO must be a nonempty finite vector')
    if not math.isfinite(probability) or not 0<=probability<=1:
        raise ValueError('probability must lie in [0,1]')
    if not math.isfinite(mixture) or not 0<=mixture<=1:
        raise ValueError('mixture must lie in [0,1]')
    _,inverse,counts=np.unique(scores,return_inverse=True,return_counts=True)
    start=np.cumsum(counts)-counts
    average_rank=start+(counts-1)/2
    strength=np.ones(1) if len(scores)==1 else 1-average_rank[inverse]/(len(scores)-1)
    return probability*((1-mixture)+mixture*strength)


def _inputs(graph,probabilities):
    n=graph.number_of_nodes()
    p=np.asarray(probabilities,dtype=float)
    if graph.is_directed() or set(graph)!=set(range(n)) or p.shape!=(n,):
        raise ValueError('requires an undirected graph with matching IDs 0..n-1')
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('edge priors must be finite and in [0,1]')
    if any(graph.has_edge(v,v) for v in graph):
        raise ValueError('self-loops are not supported')
    neighbors=[np.asarray(list(map(int,graph.neighbors(v))),dtype=int) for v in range(n)]
    return n,p.copy(),neighbors


def coverage_value(graph,seeds,probabilities):
    """Expected one-hop union size under independent coverage attempts."""
    n,p,neighbors=_inputs(graph,probabilities)
    remaining=np.ones(n)
    for v in set(map(int,seeds)):
        if not 0<=v<n:
            raise ValueError('seed outside graph')
        remaining[v]=0.
        remaining[neighbors[v]]*=1-p[v]
    return float(n-remaining.sum())


def greedy_coverage(graph,probabilities,seed_num):
    """Exact greedy on the surrogate using decreasing marginal upper bounds."""
    n,p,neighbors=_inputs(graph,probabilities)
    if not isinstance(seed_num,int) or not 0<=seed_num<=n:
        raise ValueError('invalid seed budget')
    remaining=np.ones(n)
    def gain(v):
        return float(remaining[v]+p[v]*remaining[neighbors[v]].sum())
    heap=[(-gain(v),v,0) for v in range(n)]
    heapq.heapify(heap)
    selected=[]
    while len(selected)<seed_num:
        _,v,stamp=heapq.heappop(heap)
        if stamp!=len(selected):
            heapq.heappush(heap,(-gain(v),v,len(selected)))
            continue
        selected.append(v)
        remaining[v]=0.
        remaining[neighbors[v]]*=1-p[v]
    return selected
