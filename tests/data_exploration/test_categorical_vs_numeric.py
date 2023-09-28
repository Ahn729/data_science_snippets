from unittest import TestCase

import numpy as np
from scipy.stats import kruskal

from data_science_snippets.data_exploration.categorical_vs_numeric import \
    kruskal_significant_difference


class TestKruskalSignificantDifference(TestCase):
    test_array_1 = None
    shifted_test_array_1 = None
    shuffled_test_array_1 = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_array_1 = np.arange(100)
        cls.shifted_test_array_1 = cls.test_array_1 + 10
        cls.shuffled_test_array_1 = np.random.permutation(cls.test_array_1)
        rng = np.random.default_rng(42)
        cls.test_array_2 = rng.normal(loc=40, scale=30, size=100)
        cls.test_array_3 = rng.normal(loc=50, scale=20, size=1000)

    def test_shift_is_recognized(self):
        result = kruskal_significant_difference(self.test_array_1, self.shifted_test_array_1,
                                                p=0.05)
        self.assertEqual(result.mean_diff, 10)

    def test_if_shift_is_present_there_is_a_difference(self):
        result = kruskal_significant_difference(self.test_array_1, self.shifted_test_array_1,
                                                p=0.05)
        self.assertGreater(result.significant_difference, 0)
        self.assertLess(result.significant_difference, 10)

    def test_if_array_is_shuffled_there_is_no_difference(self):
        result = kruskal_significant_difference(self.test_array_1, self.shuffled_test_array_1,
                                                p=0.05)
        self.assertIn(result.significant_difference, [0, np.nan])

    def test_p_val_does_not_exceed_limit(self):
        p = 0.05
        tol = 1e-3
        result = kruskal_significant_difference(
            self.test_array_1,
            self.shifted_test_array_1,
            p=p,
            tol=tol
        )
        self.assertLess(result.p_value, p + tol)

    def test_random_arrays(self):
        p = 0.01
        tol = 1e-4
        result = kruskal_significant_difference(self.test_array_2, self.test_array_3, p=p, tol=tol)
        self.assertGreater(result.mean_diff, 0)  # should be approx. 10
        # Due to sample size and mean_diff, this will be around 5.
        self.assertGreater(result.significant_difference, 0)
        self.assertLess(result.significant_difference, 10)
        self.assertLess(result.p_value, p + tol)

    def test_assert_p_value_is_correct(self):
        result = kruskal_significant_difference(self.test_array_2, self.test_array_3)
        shift = result.significant_difference
        kruskal_result = kruskal(self.test_array_2 + shift, self.test_array_3)
        self.assertAlmostEqual(kruskal_result[1], result.p_value)
