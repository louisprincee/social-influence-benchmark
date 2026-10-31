"""Behavior checks for the course extension; run with unittest discovery."""
import importlib
import unittest

import networkx as nx
import torch
from methods.HIM.seed_selection import adaptive_HBO_window_selection

try:
    structural = importlib.import_module('methods.HIM.structural_selection')
except ModuleNotFoundError:
    structural = None


class StructuralSelectionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(structural, 'The structural ASW extension is not implemented')

    def test_overlap_uses_neighborhood_union_not_degree_product(self):
        graph = nx.Graph([(0, 2), (0, 3), (1, 2), (1, 3), (1, 4)])
        self.assertAlmostEqual(structural.jaccard_overlap(graph, 0, 1), 2 / 3)
        graph.add_nodes_from([5, 6])
        self.assertEqual(structural.jaccard_overlap(graph, 5, 6), 0)

    def test_structure_promotes_the_candidate_with_shared_neighbors(self):
        result = structural.blend_overlap_weights(torch.tensor([.5, .5]), torch.tensor([1., 0.]), .5)
        torch.testing.assert_close(result, torch.tensor([.75, .25]))

    def test_no_structural_evidence_keeps_geometric_weights(self):
        result = structural.blend_overlap_weights(torch.tensor([.2, .8]), torch.zeros(2), .5)
        torch.testing.assert_close(result, torch.tensor([.2, .8]))

    def test_invalid_mixture_cannot_silently_change_penalty_sign(self):
        for alpha in [-.1, 1.1, float('nan')]:
            with self.assertRaises(ValueError):
                structural.blend_overlap_weights(torch.tensor([1.]), torch.tensor([1.]), alpha)

    def test_zero_mixture_reproduces_the_legacy_selector(self):
        graph = nx.barabasi_albert_graph(20, 3, seed=7)
        embs = torch.randn(20, 4, generator=torch.Generator().manual_seed(7)) * .1
        expected = adaptive_HBO_window_selection(graph, embs, 4, w_size=4)
        self.assertEqual(structural.select_structural_asw(graph, embs, 4, 4, alpha=0), expected)

    def test_exact_budget_is_unique_and_does_not_modify_inputs(self):
        graph = nx.complete_graph(12)
        embs = torch.randn(12, 4, generator=torch.Generator().manual_seed(9)) * .1
        before = embs.clone()
        selected = structural.select_structural_asw(graph, embs, 4, 4, alpha=.5)
        self.assertEqual(len(selected), 4)
        self.assertTrue(selected <= set(graph))
        torch.testing.assert_close(embs, before)
        self.assertEqual(graph.number_of_edges(), 66)
        self.assertEqual(selected, structural.select_structural_asw(graph, embs, 4, 4, alpha=.5))

    def test_equal_embeddings_and_isolates_do_not_create_nan_scores(self):
        graph = nx.empty_graph(8)
        self.assertEqual(len(structural.select_structural_asw(graph, torch.zeros(8, 4), 3, 3)), 3)
        graph = nx.complete_graph(8)
        self.assertEqual(len(structural.select_structural_asw(graph, torch.zeros(8, 4), 3, 3)), 3)

    def test_zero_and_full_budgets_and_oversized_window(self):
        graph = nx.path_graph(6)
        embs = torch.zeros(6, 4)
        self.assertEqual(structural.select_structural_asw(graph, embs, 0, 2), set())
        self.assertEqual(structural.select_structural_asw(graph, embs, 6, 30), set(range(6)))
        self.assertEqual(len(structural.select_structural_asw(graph, embs, 4, 30)), 4)

    def test_open_neighborhood_ignores_self_loops(self):
        graph = nx.Graph([(0, 0), (0, 1), (0, 2), (1, 2)])
        self.assertAlmostEqual(structural.jaccard_overlap(graph, 0, 1), 1 / 3)

    def test_looped_graph_keeps_original_degree_penalty(self):
        import sys
        from modules.hyperbolic.lorentz_function import Lorentzian_vector, squared_lorentzian_distance
        graph = nx.Graph([(0, 0), (0, 1), (0, 2), (1, 3), (2, 3)])
        embs = torch.tensor([[.1, .0], [.2, .0], [.3, .0], [.4, .0]])
        points = Lorentzian_vector(embs)
        origin = torch.zeros_like(points); origin[:, 0] = 1
        expected = float(squared_lorentzian_distance(points, origin)[0]) / graph.degree(0)
        observed = []
        def trace(frame, event, arg):
            if frame.f_code.co_name == 'select_structural_asw' and event == 'line':
                if frame.f_locals.get('current') == 0 and 'scale' in frame.f_locals:
                    observed.append(frame.f_locals['scale'])
            return trace
        previous = sys.gettrace()
        try:
            sys.settrace(trace)
            structural.select_structural_asw(graph, embs, 2, 2, alpha=.5)
        finally:
            sys.settrace(previous)
        self.assertTrue(observed)
        self.assertAlmostEqual(observed[0], expected)

    def test_mismatched_node_identity_is_rejected(self):
        with self.assertRaises(ValueError):
            structural.select_structural_asw(nx.Graph([(1, 2)]), torch.zeros(2, 4), 1, 1)


if __name__ == '__main__':
    unittest.main()
