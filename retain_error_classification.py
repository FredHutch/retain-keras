
import argparse
import functools
import itertools
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score, average_precision_score,\
                            precision_recall_curve, roc_curve
from sklearn.calibration import calibration_curve
from sklearn import tree

def get_code(tree, feature_names):
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    def recurse(left, right, threshold, features, node, indentation=0):
        INDENT = "  "
        if (threshold[node] != -2):
            print( "{}if (  {}  <= {}  ) {{".format(INDENT*indentation, features[node], str(threshold[node])))
            if left[node] != -1:
                recurse (left, right, threshold, features,left[node], indentation=indentation+1)
                print( "{}}} else {{".format(INDENT*indentation))
            if right[node] != -1:
                recurse (left, right, threshold, features,right[node], indentation=indentation+1)
                print( "{}}}".format(INDENT*indentation))
        else:
            print( "{}return {}".format(INDENT*indentation, str(value[node])))

    recurse(left, right, threshold, features, 0)
            


def convert_data_to_code_df(data, ARGS):
    def _code_in_set(whitelist, code):
        if code in whitelist:
            return (code, 1)
        else:
            return (code, 0)
        
    codes = data[0]
    code_sets = [set(list(itertools.chain(*code))) for code in codes]
    
    full_set = set(list(itertools.chain(*code_sets)))
    code_dict = [{code:exists for (code, exists) in [_code_in_set(existing_codes, check_code) for check_code in full_set]} for existing_codes in code_sets]

    code_df = pd.DataFrame(code_dict)

    return code_df


def make_a_tree(data, target, ARGS):
    model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=ARGS.depth)

    model.fit(data, target)

    return model
    

def read_data(ARGS):
    """Read the data from provided paths and assign it into lists"""    
    data = pd.read_pickle(ARGS.path_data)
    if ARGS.path_target.endswith('.csv'):
        y = pd.read_csv(ARGS.path_target)['target'].values
    else:
        y = pd.read_pickle(ARGS.path_target)['target'].values
    data_output = [data['codes'].values]
    with open(ARGS.path_features, 'rb') as f:
        feature_dict = pickle.load(f)
    
    '''
    if ARGS.numeric_size:
        data_output.append(data['numerics'].values)
    if ARGS.use_time:
        data_output.append(data['to_event'].values)
    '''
    return (data_output, y, feature_dict)


def main(ARGS):
    """Main Body of the code"""
    print('Reading Data')
    data, y, feature_dict = read_data(ARGS)

    print('Converting data to code sets')
    x_df = convert_data_to_code_df(data, ARGS)

    tree = make_a_tree(x_df, y, ARGS)

    feature_list = list(feature_dict.values())
    get_code(tree, feature_list)
    
    

def parse_arguments(parser):
    """Read user arguments"""

    parser.add_argument('--path_data', type=str, default='data/data_test.pkl',
                        help='Path to evaluation data')
    parser.add_argument('--path_target', type=str, default='data/target_test.pkl',
                        help='Path to evaluation target')
    parser.add_argument('--path_features', type=str, default='data/dictionary.pkl',
                        help='Path to feature dictionary')
    parser.add_argument('--depth', type=int, default=5,
                        help='The maximum depth of the decision tree')

    args = parser.parse_args()

    return args



if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ARGS = parse_arguments(PARSER)
    main(ARGS)
