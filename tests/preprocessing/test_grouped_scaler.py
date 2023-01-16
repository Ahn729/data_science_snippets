from math import sqrt
from unittest import TestCase

import pandas as pd

from data_science_snippets.preprocessing import GroupedScaler


class TestGroupedScaler(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.df = pd.DataFrame({
            "cat_col": ["A"] * 3 + ["B"] * 2 + ["C"] * 4,
            "num_col_1": range(9),
            "num_col_2": [0] * 4 + [1] * 5
        })

    def test_fit(self):
        gs = GroupedScaler(by="cat_col")
        gs.fit(self.df)

    def test_fit_transform_has_correct_mean(self):
        gs = GroupedScaler(by="cat_col")
        result = gs.fit_transform(self.df)
        self.assertAlmostEqual(result.loc[0:2, "num_col_1"].mean(), 0)
        self.assertAlmostEqual(result.loc[3:4, "num_col_1"].mean(), 0)
        self.assertAlmostEqual(result.loc[5:8, "num_col_1"].mean(), 0)

    def test_fit_transform_has_correct_std(self):
        gs = GroupedScaler(by="cat_col")
        result = gs.fit_transform(self.df)
        self.assertAlmostEqual(result.loc[0:2, "num_col_1"].std(ddof=0), 1)
        self.assertAlmostEqual(result.loc[3:4, "num_col_1"].std(ddof=0), 1)
        self.assertAlmostEqual(result.loc[5:8, "num_col_1"].std(ddof=0), 1)

    def test_fit_transform_is_correct_c1(self):
        gs = GroupedScaler(by="cat_col")
        result = gs.fit_transform(self.df)
        self.assertTrue((
            result.loc[:, "num_col_1"] == [
                -sqrt(3/2), 0, sqrt(3/2), -1, 1,
                -1.5 / sqrt(5 / 4), -.5 / sqrt(5/4),
                .5 / sqrt(5 / 4), 1.5 / sqrt(5/4)
            ]).all()
        )

    def test_fit_transform_is_correct_c2(self):
        gs = GroupedScaler(by="cat_col")
        result = gs.fit_transform(self.df)
        self.assertTrue((result.loc[0:2, "num_col_2"] == 0).all())
        self.assertAlmostEqual(result.at[3, "num_col_2"], -1)
        self.assertAlmostEqual(result.at[4, "num_col_2"], 1)
        self.assertTrue((result.loc[5:8, "num_col_2"] == 0).all())

    def test_get_feature_names_out_with_grouping_var(self):
        gs = GroupedScaler(by="cat_col", output_grouping_var=True)
        gs.fit(self.df)
        feature_names = gs.get_feature_names_out()
        self.assertListEqual(feature_names, ["cat_col", "num_col_1", "num_col_2"])

    def test_get_feature_names_out_without_grouping_var(self):
        gs = GroupedScaler(by="cat_col", output_grouping_var=False)
        gs.fit(self.df)
        feature_names = gs.get_feature_names_out()
        self.assertListEqual(feature_names, ["num_col_1", "num_col_2"])



