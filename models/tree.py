import csv
from typing import Tuple
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree as treeViz
import graphviz
import sys
sys.path.append('../')
from clean.number_cleaning import process_string
from clean.onehot_cleaning import process_multihot_non_matrix, process_onehot
from clean.drink_cleaning import process_drink, parse_common_drinks, parse_drinks_list
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree#, DecisionTreeClassifier
from inference_naive_bayes import predict

q3_options = ['Week day lunch', 'Week day dinner', 'Weekend lunch', 'Weekend dinner', 'At a party', 'Late night snack']
q7_options = ['Friends', 'Teachers', 'Siblings', 'Parents', 'None', 'Strangers']
q8_options = ['None', 'A little (mild)', 'A moderate amount (medium)', 'A lot (hot)',
              'I will have some of this food with my hot sauce']

X_train, t_train = None, None
X_val, t_val     = None, None
X_test, t_test   = None, None

def get_data(data_path): 
    file = open(data_path, mode='r', encoding='utf-8')
    reader = csv.reader(file)
    data = np.array(list(reader))
    file.close()

    data = data[1:, :]  # Remove first question row
    train_data, val_data, test_data = split_data(data)

    global X_train, t_train, X_val, t_val, X_test, t_test
    X_train, t_train = parse_data(train_data)
    X_val, t_val = parse_data(val_data)
    X_test, t_test = parse_data(test_data)

def split_data(data):
    """Returns rows for training/val/test data separately.
    """
    train_indices, val_indices, test_indices = load_data_split('../datasets/train_test_split.csv')
    return data[train_indices], data[val_indices], data[test_indices]


def load_data_split(filepath='data_split.csv'):
    """Load train/val/test split indices from CSV file"""
    train_indices = []
    val_indices = []
    test_indices = []
    
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if row[0] == 'train':
                train_indices.append(int(row[1]))
            elif row[0] == 'val':
                val_indices.append(int(row[1]))
            else:
                test_indices.append(int(row[1]))
    
    return train_indices, val_indices, test_indices

def parse_data(data):
    common_drinks = parse_common_drinks(input_simple_file="../clean/common_drinks.simple")
    drinks_list = parse_drinks_list(input_simple_file="../clean/common_drinks.simple")
    process_string_vectorized = np.vectorize(process_string)

    q1 = data[:, 1]  # How complex? (1-5)
    q2 = data[:, 2]  # How many ingredients? (open ended, numeric)
    q3 = data[:, 3]  # In what setting? (multi select)
    q4 = data[:, 4]  # Cost? (open ended, numeric)
    q5 = data[:, 5]  # Movie? (open ended, any)
    q6 = data[:, 6]  # Drink? (open ended, drink)
    q7 = data[:, 7]  # Who does it remind you of? (multi select)
    q8 = data[:, 8]  # How much hot sauce? (single select)
    q9 = np.zeros((data.shape[0],))
    q10 = np.zeros((data.shape[0],))
    q11 = np.zeros((data.shape[0],))
    t = data[:, 9]   # Label

    for i in range(data.shape[0]):
        naive_bayes_q = ''
        for j in range(1, 9):
            naive_bayes_q += data[i][j]
            naive_bayes_q += '.'
        _, logits = predict(naive_bayes_q, model_dir='../saved_model', verbose=False)
        q9[i] = logits['Pizza']
        q10[i] = logits['Shawarma']
        q11[i] = logits['Sushi']

    q1p = q1.astype(np.int64)
    q2p = process_string_vectorized(q2)
    q3p = np.array([process_multihot_non_matrix(x, q3_options) for x in q3])
    q4p = process_string_vectorized(q4)
    
    q6p = np.array([process_drink(x, common_drinks) for x in q6])
    q6phot = np.array([process_multihot_non_matrix(x, drinks_list) for x in q6p])

    q7p = np.array([process_multihot_non_matrix(x, q7_options) for x in q7])
    q8p = np.array([process_onehot(x, q8_options) for x in q8])
    
    columns = []
    columns.extend([np.reshape(q1p, (q1p.shape[0], 1))])
    columns.extend([np.reshape(q2p, (q2p.shape[0], 1))])
    columns.extend(np.hsplit(q3p, q3p.shape[1]))
    columns.extend([np.reshape(q4p, (q4p.shape[0], 1))])
    columns.extend(np.hsplit(q6phot, q6phot.shape[1]))
    columns.extend(np.hsplit(q7p, q7p.shape[1]))
    columns.extend(np.hsplit(q8p, q8p.shape[1]))
    columns.extend([np.reshape(q9, (q9.shape[0], 1))])
    columns.extend([np.reshape(q10, (q10.shape[0], 1))])
    columns.extend([np.reshape(q11, (q11.shape[0], 1))])

    return np.hstack(columns), t

def build_tree(criterion, d, s):
    return RandomForestClassifier(criterion=criterion, max_depth=d, min_samples_split=s)
    #return DecisionTreeClassifier(criterion=criterion, max_depth=d, min_samples_split=s)

def visualize_tree(tree):
    dot_data = treeViz.export_graphviz(tree,
                                       max_depth=3,
                                       class_names=["pizza", "shawarma", "sushi"],
                                       filled=True,
                                       rounded=True)
    graphviz.Source(dot_data).view(cleanup=True)


def score_models(max_depths,
                 min_samples_split,
                 criterion,
                 data):
    """
    Parameters:
        `max_depths` - A list of values representing the max_depth values to be
                       try as hyperparameter values
        `min_samples_split` - An list of values representing the min_samples_split
                       values to try as hyperpareameter values
        `criterion` -  A string; either "entropy" or "gini"
        `data` - a tuple of (X_train, t_train, X_valid, t_valid)

    Returns a dictionary, `out`, whose keys are the the hyperparameter choices, and whose values are
    the training and validation accuracies, as well as the tree itself.
    In other words, out[(max_depth, min_samples_split)]['val'] = validation score and
                    out[(max_depth, min_samples_split)]['train'] = training score
                    out[(max_depth, min_samples_split)]['tree'] = fitted tree
    For that combination of (max_depth, min_samples_split) hyperparameters.
    """
    X_train, X_valid, t_train, t_valid = data
    out = {}

    for d in max_depths:
        for s in min_samples_split:
            out[(d, s)] = {}
            # Create a DecisionTreeClassifier based on the given hyperparameters and fit it to the data
            tree = build_tree(criterion, d, s)
            tree.fit(X_train, t_train)
            
            out[(d, s)]['val'] = tree.score(X_valid, t_valid) 
            out[(d, s)]['train'] = tree.score(X_valid, t_valid)
            out[(d, s)]['tree'] = tree
    return out

def grid_search(data) -> Tuple[int, int, str, DecisionTreeClassifier]:
    # Hyperparameters values to try in our grid search
    criterions = ['entropy', 'gini', 'log_loss']
    max_depths = [1, 5, 10, 15, 20, 25, 30, 50, 100]
    min_samples_split = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    # we found that best_hyperparams = [(max_depth, min_samples_split), val_accuracy, criterion]
    best_hyperparams = [(-1, -1), -1, criterions[0]]
    print("criterion,max_depth,min_samples_split,valid_acc")
    for criterion in criterions:
        res = score_models(max_depths, min_samples_split, criterion, data)
    
        for d, s in res:
            if best_hyperparams[1] < res[(d, s)]['val']:
                best_hyperparams = [(d, s), res[(d, s)]['val'], criterion, res[(d, s)]['tree']]
            print(criterion + "," + str(d) + "," + str(s) + "," + str(res[(d, s)]['val']))

    d, s = best_hyperparams[0]
    criterion = best_hyperparams[2]
    tree = best_hyperparams[3]

    return d, s, criterion, tree

def train_decision_tree():
    # print(X.shape) # (1644, 66)
    # print(t.shape) # (1644, )
    d, s, criterion, tree = grid_search((X_train, X_val, t_train, t_val))
    print(d, s, criterion)
    print(tree.score(X_train, t_train))
    print(tree.score(X_val, t_val))
    print(tree.score(X_test, t_test))

    return tree

def build_best_tree():
    # print(X.shape) # (1644, 66)
    # print(t.shape) # (1644, )
    d, s, criterion = 15, 8, 'entropy'
    tree = build_tree(criterion, d, s)
    tree.fit(X_train, t_train)

    print(tree.score(X_test, t_test))

    return tree

#def make_prediction():
#    visualize_tree(tree)
#    X, t = 

if __name__ == '__main__':
    np.random.seed(0)

    get_data('../data/cleaned_data_combined_modified.csv')

    tree = train_decision_tree()    
    #tree = build_best_tree()