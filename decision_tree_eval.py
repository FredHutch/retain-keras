import argparse
import functools
import itertools
import numpy as np
import pandas as pd
import pickle
from sklearn import tree
from sklearn.metrics import roc_auc_score, average_precision_score,\
                            precision_recall_curve, roc_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def precision_recall(y_true, y_prob, graph):
    """Print Precision Recall Statistics and Graph"""
    average_precision = average_precision_score(y_true, y_prob)
    if graph:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.style.use('ggplot')
        plt.clf()
        plt.plot(recall, precision,
                 label='Precision-Recall Curve  (Area = %0.3f)' % average_precision)
        plt.xlabel('Recall: P(predicted+|true+)')
        plt.ylabel('Precision: P(true+|predicted+)')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.legend(loc="lower left")
        print('Precision-Recall Curve saved to pr.png')
        plt.savefig('pr.png')
    else:
        print('Average Precision %0.3f' % average_precision)

def probability_calibration(y_true, y_prob,graph):
    if graph:
        fig_index = 1
        name = 'My pred'
        n_bins = 20
        fig = plt.figure(fig_index, figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))

        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_true, y_prob, n_bins=n_bins, normalize=True)

        ax1.plot(mean_predicted_value, fraction_of_positives,
                 label=name)

        ax2.hist(y_prob, range=(0, 1), bins=n_bins, label=name,
                 histtype="step", lw=2)

        ax1.set_ylabel("Fraction of Positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title('Calibration Plots  (Reliability Curve)')

        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)
        print('Probability Calibration Curves saved to calibration.png')
        plt.tight_layout()
        plt.savefig('calibration.png')

def lift(y_true, y_prob, graph):
    """Print Precision Recall Statistics and Graph"""
    prevalence = sum(y_true)/len(y_true)
    average_lift = average_precision_score(y_true, y_prob) / prevalence
    if graph:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        lift_values = precision/prevalence
        plt.style.use('ggplot')
        plt.clf()
        plt.plot(recall, lift_values,
                 label='Lift-Recall Curve  (Area = %0.3f)' % average_lift)
        plt.xlabel('Recall: P(predicted+|true+)')
        plt.ylabel('Lift')
        plt.xlim([0.0, 1.0])
        plt.legend(loc="lower left")
        print('Lift-Recall Curve saved to lift.png')
        plt.savefig('lift')
    else:
        print('Average Lift %0.3f' % average_lift)

def roc(y_true, y_prob, graph):
    """Print ROC Statistics and Graph"""
    roc_auc = roc_auc_score(y_true, y_prob)
    if graph:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label='ROC curve (Area = %0.3f)'% roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specifity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        print('ROC Curve saved to roc.png')
        plt.savefig('roc.png')
    else:
        print('ROC-AUC %0.3f' % roc_auc)

def good_print_tree(estimator, feature_names, x_test):
    # The decision estimator has an attribute called tree_  which stores the entire
    # tree structure and allows access to low level attributes. The binary tree
    # tree_ is represented as a number of parallel arrays. The i-th element of each
    # array holds information about the node `i`. Node 0 is the tree's root. NOTE:
    # Some of the arrays only apply to either leaves or split nodes, resp. In this
    # case the values of nodes of the other type are arbitrary!
    #
    # Among those arrays, we have:
    #   - left_child, id of the left child of the node
    #   - right_child, id of the right child of the node
    #   - feature, feature used for splitting the node
    #   - threshold, threshold value at the node
    #

    # Using those arrays, we can parse the tree structure:

    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature  = [feature_names[i] for i in estimator.tree_.feature]
    #feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold


    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has %s nodes and has "
          "the following tree structure:"
          % n_nodes)
    for i in range(n_nodes):
        if is_leaves[i]:
            print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
        else:
            print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                  "node %s."
                  % (node_depth[i] * "\t",
                     i,
                     children_left[i],
                     feature[i],
                     threshold[i],
                     children_right[i],
                     ))
    print()

    # First let's retrieve the decision path of each sample. The decision_path
    # method allows to retrieve the node indicator functions. A non zero element of
    # indicator matrix at the position (i, j) indicates that the sample i goes
    # through the node j.

    node_indicator = estimator.decision_path(x_test)

    # Similarly, we can also have the leaves ids reached by each sample.

    leave_id = estimator.apply(x_test)

    # Now, it's possible to get the tests that were used to predict a sample or
    # a group of samples. First, let's make it for the sample.

    sample_id = 0
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]

    print('Rules used to predict sample %s: ' % sample_id)
    for node_id in node_index:
        if leave_id[sample_id] == node_id:
            continue

        if (x_test[sample_id][feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        print("decision id node %s : (x_test[%s, %s] (= %s) %s %s)"
              % (node_id,
                 sample_id,
                 feature[node_id],
                 x_test[sample_id, feature[node_id]],
                 threshold_sign,
                 threshold[node_id]))

    # For a group of samples, we have the following common node.
    sample_ids = [0, 1]
    common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) ==
                    len(sample_ids))

    common_node_id = np.arange(n_nodes)[common_nodes]

    print("\nThe following samples %s share the node %s in the tree"
          % (sample_ids, common_node_id))
    print("It is %s %% of all nodes." % (100 * len(common_node_id) / n_nodes,))

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

def convert_data_to_code_df(data, full_set, ARGS):
    def _code_in_set(whitelist, code):
        if code in whitelist:
            return (code, 1)
        else:
            return (code, 0)
        
    codes = data[0]
    code_sets = [set(list(itertools.chain(*code))) for code in codes]

    print('before')
    wide_code_list = []
    for patient_codes in code_sets:
        patient_code_map = {code:0 for code in full_set}
        for code in patient_codes:
            if code in full_set:
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
    codes = itertools.chain(x_train[0], x_test[0])
    code_sets = [set(list(itertools.chain(*code))) for code in codes]
    full_set = set(list(itertools.chain(*code_sets)))
    x_df = convert_data_to_code_df(x_train, full_set, ARGS)

    print('Training Decision Tree')
    dt = make_a_tree(x_df, y_train, ARGS)

    print('Getting predictions')
    x_df = convert_data_to_code_df(x_test, full_set, ARGS)
    y_predict = dt.predict(x_df)

    print('Evaluating Results')
    roc(y_test, y_predict, ARGS.omit_graphs)
    precision_recall(y_test, y_predict, ARGS.omit_graphs)
    lift(y_test, y_predict, ARGS.omit_graphs)
    probability_calibration(y_test, y_predict, ARGS.omit_graphs)

    print('Printing Decision Tree')
    feature_list = list(feature_dict.values())
    features  = [feature_dict[i] for i in list(x_df.columns.values)]   
    print(len(feature_list))
    print(len(features))
    #good_print_tree(tree, feature_list, x_df)
    get_code(dt, features)
    with open("decisiontree_D{}.dot".format(ARGS.depth), 'w') as dotfile:
        tree.export_graphviz(dt, out_file=dotfile, feature_names=features)


def parse_arguments(parser):
    """Read user arguments"""

    parser.add_argument('--path_train_data', type=str, default='C:/Users/whiteau/Documents/non_partial_data/claims_visits_ED_1_data.pkl', help='Path to train data')
    parser.add_argument('--path_train_target', type=str, default='C:/Users/whiteau/Documents/non_partial_data/claims_visits_ED_1_target.pkl', help='Path to train target')
    parser.add_argument('--path_test_data', type=str, default='C:/Users/whiteau/Documents/non_partial_data/claims_visits_ED_2_data.pkl', help='Path to evaluation data')
    parser.add_argument('--path_test_target', type=str, default='C:/Users/whiteau/Documents/non_partial_data/claims_visits_ED_2_target.pkl',
                        help='Path to evaluation target')
    parser.add_argument('--path_features', type=str, default='C:/Users/whiteau/Documents/non_partial_data/dictionary.pkl',
                        help='Path to feature dictionary')
    parser.add_argument('--depth', type=int, default=4,
                        help='The maximum depth of the decision tree')
    parser.add_argument('--omit_graphs', action='store_false',
                        help='Does not output graphs if argument is present')
    args = parser.parse_args()

    return args



if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ARGS = parse_arguments(PARSER)
    main(ARGS)
