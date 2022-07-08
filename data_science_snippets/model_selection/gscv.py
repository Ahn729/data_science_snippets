from typing import Dict, List, Any

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV


def plot_gscv_results(gscv: GridSearchCV,
                      hyperparams: Dict[str, List[Any]],
                      min_score: float = None) -> plt.Figure:
    """Plots the results of a fitted GridSearchCV instance as boxplots

    Args:
        gscv: A fitted sklearn GridSearchCV instance
        hyperparams: Parameter grid that was used to fit the gscv
        min_score: Minimum score for a datapoint to be included in the
            plot. Use if there are outliers in a negetive sense which
            mess up the plots

    Returns:

    """
    n_plots = len(hyperparams)
    results = pd.DataFrame(gscv.cv_results_).fillna('default')
    if min_score is not None:
        results = results[results['mean_test_score'] >= min_score]
    fig, axs = plt.subplots(nrows=n_plots, figsize=(8, 6*n_plots))
    for i, param in enumerate(hyperparams.keys()):
        sns.boxplot(data=results, x=f'param_{param}', y="mean_test_score", ax=axs[i])
    return fig
