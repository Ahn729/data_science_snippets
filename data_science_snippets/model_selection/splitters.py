__all__ = ["RepeatedGroupKFold", "RepeatedStratifiedGroupKFold"]

from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
import numpy as np


class _BaseRepeatedGroupKFold:
    def __init__(self, cv_class, n_splits=5, n_repeats=10, random_state=None, strat_var=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.cv = cv_class
        self.y = strat_var

    def split(self, X, y=None, groups=None):
        if self.y is not None:
            y = self.y
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None")
        n_samples = len(groups)

        # Validate the number of groups
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        if n_groups < self.n_splits:
            raise ValueError("Number of groups is less than n_splits")

        # RNG
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_repeats):
            shuffled_groups = rng.permutation(unique_groups)
            bins = np.zeros(n_samples, dtype=np.uint)
            for idx, group in enumerate(shuffled_groups):
                bins[groups == group] = idx
            yield from self.cv(n_splits=self.n_splits).split(X, y, bins)


class RepeatedGroupKFold(_BaseRepeatedGroupKFold):
    """A combination of sklearn's GroupKFold and RepeatedKFold.

    Repeatedly splits the dataset into train and test sets, respecting groups
    (similar to GroupKFold)"""
    def __init__(self, n_splits=5, n_repeats=10, random_state=None):
        super().__init__(GroupKFold, n_splits=n_splits, n_repeats=n_repeats,
                         random_state=random_state)


class RepeatedStratifiedGroupKFold(_BaseRepeatedGroupKFold):
    """A combination of sklearn's StratifiedGroupKFold and RepeatedKFold.

    Repeatedly splits the dataset into train and test sets, respecting groups and
    preserving stratifications as well as possible (similar to StratifiedGroupKFold).
    Additionally, you can specify a stratification variable to use as the target for
    stratification (not necessarily the same as the response variable y).
    """
    def __init__(self, n_splits=5, n_repeats=10, random_state=None, strat_var=None):
        super().__init__(StratifiedGroupKFold, n_splits=n_splits, n_repeats=n_repeats,
                         random_state=random_state, strat_var=strat_var)
