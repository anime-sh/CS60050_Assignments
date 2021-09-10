import pandas as pd
def get_data(path):
    """
    Loads the data from the given path.
    :param path: the path to the data
    :return: the data as a pandas dataframe
    """
    return pd.read_csv(path)
