import pandas as pd


def reduce_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print(f'Dataframe shape before starting reduce {df.shape}')
    # Drop columns which only have na entries
    ret = df.loc[:, df.notna().any().values]
    print(f'Dataframe shape after dropping all-nan cols {ret.shape}')
    # Drop columns which have all the same entries
    ret = ret.loc[:, ret.nunique(dropna=False) != 1]
    print(f'Dataframe shape after dropping all-equal cols {ret.shape}')
    return ret
