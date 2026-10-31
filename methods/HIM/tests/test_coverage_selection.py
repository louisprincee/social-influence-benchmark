"""Behavior tests for cumulative coverage; exact small-graph enumeration is the oracle."""
import importlib
import unittest
import networkx as nx
import numpy as np

try:
    coverage = importlib.import_module('methods.HIM.coverage_selection')
except ModuleNotFoundError:
    coverage = None


def objective(graph, seeds, probabilities):
    total = 0.
    for target in graph:
        remaining = 1.
        for source in seeds:
            q = 1. if source == target else probabilities[source] if graph.has_edge(source,target) else 0.
            remaining *= 1-q
        total += 1-remaining
    return total


def full_scan(graph, probs, k):
    selected=[]
    for _ in range(k):
        base=objective(graph,selected,probs)
        choice=min((v for v in graph if v not in selected), key=lambda v:(-(objective(graph,selected+[v],probs)-base),v))
        selected.append(choice)
    return selected


class CoverageSelectionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(coverage,'Cumulative coverage selector is not implemented')

    def test_lazy_greedy_matches_exhaustive_marginal_gain(self):
        graph=nx.Graph([(0,1),(0,2),(1,2),(2,3),(3,4),(4,5),(5,6)])
        probs=np.array([.07,.1,.08,.065,.09,.055,.1])
        selected=coverage.greedy_coverage(graph,probs,5)
        self.assertEqual(selected,full_scan(graph,probs,5))

    def test_selects_an_uncovered_region_after_a_redundant_hub(self):
        graph=nx.Graph([(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(5,6),(5,7)])
        selected=coverage.greedy_coverage(graph,np.ones(8),2)
        self.assertEqual(selected,[0,5])

    def test_coverage_uses_union_probability_instead_of_adding_twice(self):
        graph=nx.path_graph(3)
        self.assertAlmostEqual(coverage.coverage_value(graph,[0,2],np.full(3,.5)),2.75)

    def test_no_him_mix_is_independent_of_embedding_ranks(self):
        a=coverage.him_edge_prior([0,1,2],.1,0)
        b=coverage.him_edge_prior([20,-1,3],.1,0)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_allclose(a,[.1,.1,.1])

    def test_lower_ldo_has_a_stronger_prior_and_ties_match(self):
        np.testing.assert_allclose(coverage.him_edge_prior([0,2,2,4],.1,.5),[.1,.075,.075,.05])

    def test_zero_full_budgets_and_isolates(self):
        graph=nx.empty_graph(4)
        self.assertEqual(coverage.greedy_coverage(graph,np.full(4,.1),0),[])
        self.assertEqual(coverage.greedy_coverage(graph,np.full(4,.1),4),[0,1,2,3])

    def test_invalid_probability_or_node_mapping_is_rejected(self):
        for probs in [[.2,-.1],[.2,1.1],[.2,float('nan')]]:
            with self.assertRaises(ValueError):coverage.greedy_coverage(nx.path_graph(2),probs,1)
        with self.assertRaises(ValueError):coverage.greedy_coverage(nx.Graph([(1,2)]),[.1,.1],1)
        with self.assertRaises(ValueError):coverage.greedy_coverage(nx.path_graph(2),[.1,.1],3)

    def test_inputs_are_unchanged_and_prior_is_rank_based(self):
        graph=nx.path_graph(5);before=list(graph.edges());scores=np.arange(5,dtype=float)
        a=coverage.him_edge_prior(scores,.1,.5)
        b=coverage.him_edge_prior(3*scores+2,.1,.5)
        np.testing.assert_allclose(a,b)
        coverage.greedy_coverage(graph,a,3)
        np.testing.assert_array_equal(scores,np.arange(5))
        self.assertEqual(before,list(graph.edges()))
