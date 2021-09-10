import pandas as pd
import numpy as np

def get_data(path):
    """
    Loads the data from the given path.
    Assumes last columns of the data are the target values.
    :param path: the path to the data
    :return: the data as X and y numpy arrays
    """
    data = pd.read_csv(path)
    X = data.drop(data.columns[-1], axis=1).to_numpy()
    y = data[data.columns[-1]].to_numpy()
    return X, y

def train_test_split(X,y,train_size,shuffle=True,seed=42):

    """
    Splits the data into training and test sets.
    :param X: the data
    :param y: the target values
    :param train_size: the size of the training set
    :param shuffle: whether to shuffle the data
    :param seed: the seed for the random generator
    :return: X_train, X_test, y_train, y_test
    """
    length=len(X)
    n_train = int(np.ceil(length*train_size))
    n_test = length - n_train

    if shuffle:
        perm = np.random.RandomState(seed).permutation(length)
        test_indices = perm[:n_test]
        train_indices = perm[n_test:]
    else:
        train_indices = np.arange(n_train)
        test_indices = np.arange(n_train, length)

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    return X_train, X_test, y_train, y_test


def train_val_test_split(X,y,train_size,val_size,shuffle=True,seed=42):
    '''
    Splits the data into training, validation and test sets.
    :param X: the data
    :param y: the target values
    :param train_size: the size of the training set
    :param val_size: the size of the validation set
    :param shuffle: whether to shuffle the data
    :param seed: the seed for the random generator
    :return: X_train, X_val, X_test, y_train, y_val, y_test
    '''
    length=len(X)
    n_train = int(np.ceil(length*train_size))
    n_val = int(np.ceil(length*val_size))
    n_test = length - n_train - n_val

    if shuffle:
        perm = np.random.RandomState(seed).permutation(length)
        test_indices = perm[:n_test]
        val_indices = perm[n_test:n_test+n_val]
        train_indices = perm[n_test+n_val:]
    else:
        train_indices = np.arange(n_train)
        val_indices = np.arange(n_train, n_train + n_val)
        test_indices = np.arange(n_train + n_val, length)

    X_train = X[train_indices]
    X_val = X[val_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_val = y[val_indices]
    y_test = y[test_indices]
    return X_train, X_val, X_test, y_train, y_val, y_test

def check_purity(y):
    """
    Checks if the given array is pure.
    :param y: the array
    :return: True if the array is pure, False otherwise
    """
    return len(set(y)) == 1

def classify_array(y):
    """
    Classifies the array into a single class.
    find most common number and return that
    :param y: the array
    :return: the class
    """
    classes, counts = np.unique(y, return_counts=True)
    return classes[counts.argmax()]
