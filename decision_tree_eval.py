import argparse
import functools
import itertools
import numpy as np
import pandas as pd
import pickle
from sklearn import tree

from retain_evaluation import precision_recall, probability_calibration, lift, roc


def convert_data_to_code_df(data, ARGS):
    def _code_in_set(whitelist, code):
        if code in whitelist:
            return (code, 1)
        else:
            return (code, 0)
        
    codes = data[0]
    code_sets = [set(list(itertools.chain(*code))) for code in codes]
    
    full_set = set(list(itertools.chain(*code_sets)))
    print(full_set)
    print('before')
    wide_code_list = []
    for patient_codes in code_sets:
        patient_code_map = {code:0 for code in full_set}
        for code in patient_codes:
            patient_code_map[code] = 1
        wide_code_list.append(patient_code_map)
    
        
    #code_dict = [{code:exists for (code, exists) in [_code_in_set(existing_codes, check_code) for check_code in full_set]} for existing_codes in code_sets]
    print('after')
    #code_df = pd.DataFrame(code_dict)
    code_df = pd.DataFrame(wide_code_list)

    return code_df

def make_a_tree(data, target, ARGS):
    model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=ARGS.depth)

    model.fit(data, target)

    return model


def read_data(ARGS):
    """Read the data from provided paths and assign it into lists"""

    #train
    data = pd.read_pickle(ARGS.path_train_data)
    if ARGS.path_train_target.endswith('.csv'):
        y_train = pd.read_csv(ARGS.path_train_target)['target'].values
    else:
        y_train = pd.read_pickle(ARGS.path_train_target)['target'].values
    x_train = [data['codes'].values]

    #test
    data = pd.read_pickle(ARGS.path_test_data)
    if ARGS.path_test_target.endswith('.csv'):
        y_test = pd.read_csv(ARGS.path_test_target)['target'].values
    else:
        y_test = pd.read_pickle(ARGS.path_test_target)['target'].values
    x_test = [data['codes'].values]

    with open(ARGS.path_features, 'rb') as f:
        feature_dict = pickle.load(f)
    
    '''
    if ARGS.numeric_size:
        data_output.append(data['numerics'].values)
    if ARGS.use_time:
        data_output.append(data['to_event'].values)
    '''
    return (feature_dict, x_train, y_train, x_test, y_test)


def main(ARGS):
    """Main Body of the code"""
    print('Reading Data')
    feature_dict, x_train, y_train, x_test, y_test = read_data(ARGS)

    print('Converting data to code sets')
    x_df = convert_data_to_code_df(x_train, ARGS)

    print('Training Decision Tree')
    tree = make_a_tree(x_df, y_train, ARGS)

    print('Getting predictions')
    x_df = convert_data_to_code_df(x_test, ARGS)
    y_predict = tree.predict(x_df)

    print('Evaluating Results')
    roc(y_test, y_predict, ARGS.omit_graphs)
    precision_recall(y_test, y_predict, ARGS.omit_graphs)
    lift(y_test, y_predict, ARGS.omit_graphs)
    probability_calibration(y_test, y_predict, ARGS.omit_graphs)

    print('Printing Decision Tree')
    feature_list = list(feature_dict.values())
    get_code(tree, feature_list)


def parse_arguments(parser):
    """Read user arguments"""

    parser.add_argument('--path_train_data', type=str, default='/home/whiteau/Data/retain/non_partial_data/claims_visits_ED_1_data.pkl', help='Path to train data')
    parser.add_argument('--path_train_target', type=str, default='/home/whiteau/Data/retain/non_partial_data/claims_visits_ED_1_target.pkl', help='Path to train target')
    parser.add_argument('--path_test_data', type=str, default='/home/whiteau/Data/retain/non_partial_data/claims_visits_ED_2_data.pkl', help='Path to evaluation data')
    parser.add_argument('--path_test_target', type=str, default='/home/whiteau/Data/retain/non_partial_data/claims_visits_ED_2_target.pkl',
                        help='Path to evaluation target')
    parser.add_argument('--path_features', type=str, default='/home/whiteau/Data/retain/non_partial_data/dictionary.pkl',
                        help='Path to feature dictionary')
    parser.add_argument('--depth', type=int, default=10,
                        help='The maximum depth of the decision tree')
    parser.add_argument('--omit_graphs', action='store_false',
                        help='Does not output graphs if argument is present')
    args = parser.parse_args()

    return args



if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ARGS = parse_arguments(PARSER)
    main(ARGS)
