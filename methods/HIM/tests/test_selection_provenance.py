"""Reject stale experiment inputs before reusing propagation outcomes."""
import importlib
import unittest
from pathlib import Path
import hashlib

try:
    provenance = importlib.import_module('selection_provenance')
except ModuleNotFoundError:
    provenance = None


class SelectionProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(provenance, 'Selection cache validation is not implemented')
        selector = Path(__file__).parents[1] / 'methods/HIM/structural_selection.py'
        self.expected = {'protocol_sha256': 'protocol', 'graph_sha256': 'graph',
                         'checkpoint_sha256': 'model', 'selector_sha256': hashlib.sha256(selector.read_bytes()).hexdigest(),
                         'evaluation_rounds': 2, 'nodes': 4}
        self.selections = {'ASW-J': [0, 1]}
        self.saved = dict(self.expected, methods={'ASW-J': {'seeds': [0, 1], 'activated_per_replicate': [2, 3]}})

    def test_matching_inputs_and_reselected_set_can_reuse_outcomes(self):
        provenance.validate_cached_structural(self.saved, self.expected, self.selections)

    def test_changed_graph_cannot_reuse_outcomes(self):
        self.saved['graph_sha256'] = 'stale'
        with self.assertRaises(ValueError):
            provenance.validate_cached_structural(self.saved, self.expected, self.selections)

    def test_unknown_selector_revision_cannot_reuse_outcomes(self):
        self.saved['selector_sha256'] = 'unknown'
        with self.assertRaises(ValueError):
            provenance.validate_cached_structural(self.saved, self.expected, self.selections)

    def test_changed_reselected_set_cannot_reuse_outcomes(self):
        with self.assertRaises(ValueError):
            provenance.validate_cached_structural(self.saved, self.expected, {'ASW-J': [0, 2]})
