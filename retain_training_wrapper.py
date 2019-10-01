import argparse
import logging
import mlflow
import pandas as pd
from retain_train import parse_arguments, model_create, train_model
from train_validation_split import generate_train_test_split_cv, HYPERPARAM_SEARCH

logger = logging.getLogger(__name__)


def train_wrapper(dataset, ARGS):


    k_folds = ARGS.k_folds
    n_repeats = ARGS.n_repeats
    i = 0
    j = 0
    hyperparam_list = [(ARGS.l2, ARGS.dropout_input, ARGS.use_time)]
    if ARGS.hyperparam_search is True:
        hyperparam_list = HYPERPARAM_SEARCH
    for param_set in hyperparam_list:
        logger.info("Hyperparam search is {}. "
                    "hyperparam set is: L2 {}, embedding dropout {}, use time {}".format(ARGS.hyperparam_search,
                                                                                         param_set[0],
                                                                                         param_set[1],
                                                                                         param_set[2]))
        ARGS.l2, ARGS.dropout_input, ARGS.use_time = param_set #hacky, but lets us leave retain code mostly unmodified
        for data_train, y_train, data_test, y_test in generate_train_test_split_cv(dataset,
                                                                                   ARGS,
                                                                                   k_folds=k_folds,
                                                                                   n_repeats=n_repeats,
                                                                                   random_seed=12345):
            k = i % k_folds
            n = k % n_repeats
            print('Creating Model for K={k}, N={n}'.format(k=k, n=n))
            model = model_create(ARGS)
            print('Training Model')

            train_model(model=model, data_train=data_train, y_train=y_train,
                        data_test=data_test, y_test=y_test, ARGS=ARGS)

            mlflow.keras.log_model(model, "model_{k}_{n}".format(k=k, n=n))
            i+=1
            if i % k_folds == k_folds-1:
                j+=1


if __name__ == '__main__':

    PARSER = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    PARSER.add_argument('--path_dataset', type=str, default='data/training_df.pkl',
                      help='Path to training data, unsplit')
    PARSER.add_argument('--hyperparam_search', action='store_true',
                        help='If argument is present then hyperparamet grid search will be performed. '
                             'WARNING: this will multiply the number of total runs that are done!')
    PARSER.add_argument('--k_folds', type=int, default=5,
                        help='Number of folds to be used in Cross Validation')
    PARSER.add_argument('--n_repeats', type=int, default=1,
                        help='Number of random iterations of Cross Validation to perform. '
                             'WARNING: this will multiply the number of total runs performed by the value chosen!')



    ARGS = parse_arguments(PARSER)

    with  mlflow.start_run() as run:
        training_dataset = pd.read_pickle(ARGS.path_dataset)
        train_wrapper(training_dataset, ARGS)