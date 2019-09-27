import argparse
import mlflow
import pandas as pd
from retain_train import parse_arguments, model_create, train_model
from train_validation_split import generate_train_test_split_cv

def train_wrapper(dataset, ARGS):


    k_folds = 5
    n_repeats = 10
    i = 0
    for data_train, y_train, data_test, y_test in generate_train_test_split_cv(dataset,
                                                                               k_folds=k_folds,
                                                                               n_repeats=n_repeats,
                                                                              random_seed=12345):
        k = i % k_folds
        n = i % n_repeats
        print('Creating Model for K={k}, N={n}'.format(k=k, n=n))
        model = model_create(ARGS)
        print('Training Model')

        train_model(model=model, data_train=data_train, y_train=y_train,
                    data_test=data_test, y_test=y_test, ARGS=ARGS)

        mlflow.keras.log_model(model, "model_{k}_{n}".format(k=k, n=n))
        i+=1


if __name__ == '__main__':

    PARSER = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    PARSER.add_argument('--path_dataset', type=str, default='data/training_df.pkl',
                      help='Path to training data, unsplit')
    ARGS = parse_arguments(PARSER)

    with  mlflow.start_run() as run:
        training_dataset = pd.read_pickle(ARGS.path_dataset)
        train_wrapper(training_dataset, ARGS)