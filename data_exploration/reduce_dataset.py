# noqa
# TODO: Refactor and cleanup

def reduce_dataset(df):
    print(f'{df.shape}')
    # Drop columns which only have na entries
    ret = df.loc[:, df.notna().any().values]
    print(f'{ret.shape}')
    # Drop columns which have all the same entries
    ret = ret.loc[:, ret.nunique(dropna=False) != 1]
    print(f'{ret.shape}')
    return ret