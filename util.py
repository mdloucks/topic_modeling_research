import pickle
from os import path

'''
A list of utility functions and tools
'''

def cache(data, name):
    """Cache a given set of data to a pickle file

    Args:
        data (Object): the data to be cached
        name (str): name to be stored under
    """
    with open(f"pkl/{name}.pkl", 'wb') as f:
        pickle.dump(data, f)


def load(name):
    """Load a cached pickle

    Args:
        name (str): name of file

    Returns:
        Object/bool: de-serialized pickle or false on not found
    """
    filename = f"pkl/{name}.pkl"

    if not(path.exists(filename)):
        return False

    with open(filename, 'rb') as f:
        return pickle.load(f)
