import pandas as pd

__all__ = ['handle_nans', 'reduce_dataset']


def handle_nans(df: pd.DataFrame,
                max_nan_proportion: float = 0.1,
                nan_fill_method: str = "mean") -> pd.DataFrame:
    """Handles nan columns by either dropping or filling them

    Drops columns with more nans than max_nan_proportion, fills
    the rest using the method specified

    Args:
        df: dataframe to process
        max_nan_proportion: threshold which determines which
            columns to keep
        nan_fill_method:
            method to fill nan values (e.g. mean, or some
            other callable attribute of df)

    Returns:
        dataframe without nans
    """
    nan_proportions = df.isna().mean()
    nan_cols = nan_proportions[nan_proportions > max_nan_proportion].index
    ret = df.drop(columns=nan_cols)
    fill_func = getattr(df, nan_fill_method)
    return ret.fillna(fill_func())


def reduce_dataset(df: pd.DataFrame, remove_nans: bool = False, **kwargs) -> pd.DataFrame:
    print(f'Dataframe shape before starting reduce: {df.shape}')
    # Drop columns which only have na entries
    ret = df.loc[:, df.notna().any().values]
    print(f'Dataframe shape after dropping all-nan cols: {ret.shape}')
    # Drop columns which have all the same entries
    ret = ret.loc[:, ret.nunique(dropna=False) != 1]
    print(f'Dataframe shape after dropping all-equal cols: {ret.shape}')
    if remove_nans:
        ret = handle_nans(ret, **kwargs)
        print(f'Dataframe shape after handling nans: {ret.shape}')
    return ret
