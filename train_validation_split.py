import json
import logging
import mlflow
import numpy as np
import pandas as pd
import os

from sklearn.utils.validation import check_array
from sklearn.model_selection._split import _BaseKFold, _RepeatedSplits, NSPLIT_WARNING #define our own generator class
import warnings
from collections import defaultdict

import os
import pandas as pd
import itertools
import logging

logger = logging.getLogger(__name__)


L2_GRID = [0.0, .1, .5]
DROPOUT_GRID = [0.0, .1, .5]
USETIME_GRID = [True, False]

HYPERPARAM_SEARCH = list(itertools.product(*[L2_GRID, DROPOUT_GRID, USETIME_GRID]))

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
            CustomGroupKFold, n_repeats, random_state=random_state, n_splits=n_splits)

class CustomGroupKFold(_BaseKFold):
    """K-fold iterator variant with non-overlapping groups.

    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).

    The folds are approximately balanced in the sense that the number of
    distinct groups is approximately the same in each fold.

    Parameters
    ----------
    n_splits : int, default=3
        Number of folds. Must be at least 2.

        .. versionchanged:: 0.20
            ``n_splits`` default value will change from 3 to 5 in v0.22.

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([1, 2, 3, 4])
    >>> groups = np.array([0, 0, 2, 2])
    >>> group_kfold = CustomGroupKFold(n_splits=2)
    >>> group_kfold.get_n_splits(X, y, groups)
    2
    >>> print(group_kfold)
    CustomGroupKFold(n_splits=2)
    >>> for train_index, test_index in group_kfold.split(X, y, groups):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    ...     print(X_train, X_test, y_train, y_test)
    ...
    TRAIN: [0 1] TEST: [2 3]
    [[1 2]
     [3 4]] [[5 6]
     [7 8]] [1 2] [3 4]
    TRAIN: [2 3] TEST: [0 1]
    [[5 6]
     [7 8]] [[1 2]
     [3 4]] [3 4] [1 2]

    See also
    --------
    LeaveOneGroupOut
        For splitting the data according to explicit domain-specific
        stratification of the dataset.
    """
    def __init__(self, n_splits='warn', shuffle=False, random_state=None):
        if n_splits == 'warn':
            warnings.warn(NSPLIT_WARNING, FutureWarning)
            n_splits = 3
        super(CustomGroupKFold, self).__init__(n_splits, shuffle=shuffle,
                                     random_state=random_state)
    def _iter_test_indices(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, ensure_2d=False, dtype=None)

        unique_groups, groups = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)

        if self.n_splits > n_groups:
            raise ValueError("Cannot have number of splits n_splits=%d greater"
                             " than the number of groups: %d."
                             % (self.n_splits, n_groups))

        # Weight groups by their number of occurrences
        n_samples_per_group = np.bincount(groups)

        # Distribute the most frequent groups first
        indices = np.argsort(n_samples_per_group)[::-1]
        n_samples_per_group = n_samples_per_group[indices]

        # Total weight of each fold
        n_samples_per_fold = np.zeros(self.n_splits)

        # Mapping from group index to fold index
        group_to_fold = np.zeros(len(unique_groups))

        # Distribute samples by adding the largest weight to the lightest fold
        for group_index, weight in enumerate(n_samples_per_group):
            lightest_fold = np.argmin(n_samples_per_fold)
            n_samples_per_fold[lightest_fold] += weight
            group_to_fold[indices[group_index]] = lightest_fold

        indices = group_to_fold[groups]

        for f in range(self.n_splits):
            yield np.where(indices == f)[0]

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,), optional
            The target variable for supervised learning problems.

        groups : array-like, with shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        return super(CustomGroupKFold, self).split(X, y, groups)

def generate_train_test_split_cv(df, ARGS, k_folds=None, n_repeats=None, random_seed=None):
    """
    split train dev sets using cross validation
    To-do: integrate more CV methods, e.g., loo, lop, etc
    :param df: data frame
    :param k_folds: number of folds specified in cross validation
    :param n_repeats: number of iterations through K-Folds Strategy specified in cross validation
    :param random_seed: the random seed (if any) to use
    :return: data_train, target_train, data_test, target_test
    """
    def xform_dataframes(data_train, target_train, data_test, target_test, ARGS):
        """Read the data from provided paths and assign it into lists"""
        y_train = target_train['target'].values
        y_test = target_test['target'].values
        data_output_train = [data_train['codes'].values]
        data_output_test = [data_test['codes'].values]

        if ARGS.numeric_size:
            data_output_train.append(data_train['numerics'].values)
            data_output_test.append(data_test['numerics'].values)
        if ARGS.use_time:
            data_output_train.append(data_train['to_event'].values)
            data_output_test.append(data_test['to_event'].values)

        mlflow.log_artifact(ARGS.path_dataset, artifact_path='input/train')

        return (data_output_train, y_train, data_output_test, y_test)

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

        yield xform_dataframes(df_subset['data_train'], df_subset['target_train'],
              df_subset['data_test'], df_subset['target_test'], ARGS)




