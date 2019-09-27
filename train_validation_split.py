import json
import logging
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import GroupKFold
from sklearn.model_selection._split import _RepeatedSplits #define our own generator class
from collections import defaultdict

import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class RepeatedGroupKFold(_RepeatedSplits):
    """Repeated Group K-Fold cross validator.

    Repeats Group K-Fold n times with different randomization in each repetition.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    n_repeats : int, default=10
        Number of times cross-validator needs to be repeated.

    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 1, 1])
    >>> rkf = RepeatedGroupKFold(n_splits=2, n_repeats=2, random_state=2652124)
    >>> for train_index, test_index in rkf.split(X):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    ...
    TRAIN: [0 1] TEST: [2 3]
    TRAIN: [2 3] TEST: [0 1]
    TRAIN: [1 2] TEST: [0 3]
    TRAIN: [0 3] TEST: [1 2]

    Notes
    -----
    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting ``random_state``
    to an integer.

    See also
    --------
    RepeatedStratifiedKFold: Repeats Stratified K-Fold n times.
    """
    def __init__(self, n_splits=5, n_repeats=10, random_state=None):
        super(RepeatedGroupKFold, self).__init__(
            GroupKFold, n_repeats, random_state, n_splits=n_splits)


def generate_train_test_split_cv(df, k_folds=None, n_repeats=None, random_seed=None):
    """
    split train dev sets using cross validation
    To-do: integrate more CV methods, e.g., loo, lop, etc
    :param df: data frame
    :param k_folds: number of folds specified in cross validation
    :param n_repeats: number of iterations through K-Folds Strategy specified in cross validation
    :param random_seed: the random seed (if any) to use
    :return: data_train, target_train, data_test, target_test
    """
    X = df.iloc[:, df.columns != 'target']
    y = df.target.to_frame()

    if not k_folds:
        k_folds = 5
        logger.warning('Applied {k}-fold cross validation by default'.format(k=k_folds))

    if not n_repeats:
        n_repeats = 10
        logger.warning('Applied n_repeats = {n} cross validation by default'.format(n=n_repeats))


    # Grouped K Fold CV (all members of a group fall in same Fold)
    group_kfold = RepeatedGroupKFold(n_splits=k_folds, n_repeats=n_repeats, random_state=random_seed)

    for train_index, test_index in group_kfold.split(X, y, groups=X.PID):
        df_subset = {}
        df_subset['data_train'], df_subset['data_test'] = X.iloc[train_index], X.iloc[test_index]
        df_subset['target_train'], df_subset['target_test'] = y.iloc[train_index], y.iloc[test_index]

        yield df_subset['data_train'], df_subset['target_train'], \
              df_subset['data_test'], df_subset['target_test']

