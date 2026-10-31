"""Check the independent-unit interval used in the reproduction report."""
import unittest

from expanded_reproduction import interval


class ReportingStatisticsTests(unittest.TestCase):
    def test_three_training_runs_use_two_degrees_of_freedom(self):
        lo, hi = interval([-1, 0, 1])
        self.assertAlmostEqual(lo, -2.4841377117)
        self.assertAlmostEqual(hi, 2.4841377117)

    def test_identical_runs_have_a_finite_zero_width_interval(self):
        self.assertEqual(interval([2, 2, 2]), [2, 2])

    def test_single_training_run_cannot_claim_training_interval(self):
        with self.assertRaises(ValueError):
            interval([1])
