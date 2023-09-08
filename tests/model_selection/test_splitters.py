from unittest import TestCase

from data_science_snippets.model_selection import RepeatedGroupKFold, RepeatedStratifiedGroupKFold

import pandas as pd


class TestRepeatedGroupKFold(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.df = pd.DataFrame({
            "X": [i for i in range(16)],  # X
            "Y": [i for i in range(16)],  # y
            "G": [i // 2 for i in range(10)] + [5, 5, 5, 6, 6, 6],  # group variable
            "S": [i // 8 for i in range(16)],  # stratification variable
        })

    def test_split_returns_correct_number_of_splits(self):
        kf = RepeatedGroupKFold(n_splits=3, n_repeats=5)
        self.assertEqual(len(list(kf.split(self.df, groups=self.df["G"]))), 3 * 5)

    def test_split_returns_correct_number_of_splits_stratified(self):
        kf = RepeatedStratifiedGroupKFold(n_splits=3, n_repeats=5, strat_var=self.df["S"])
        self.assertEqual(len(list(kf.split(self.df, groups=self.df["G"]))), 3 * 5)

    def _test_splitter_respects_groups(self, kf):
        for train_index, test_index in kf.split(self.df, groups=self.df["G"]):
            train_groups = self.df.loc[train_index, "G"]
            test_groups = self.df.loc[test_index, "G"]
            intersection = set(train_groups).intersection(set(test_groups))
            self.assertSetEqual(intersection, set())

    def test_splitters_respect_groups(self):
        kf = RepeatedGroupKFold(n_splits=3, n_repeats=5)
        self._test_splitter_respects_groups(kf)

    def test_splitters_respect_groups_stratified(self):
        kf = RepeatedStratifiedGroupKFold(n_splits=3, n_repeats=5, strat_var=self.df["S"])
        self._test_splitter_respects_groups(kf)



