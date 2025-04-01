# Run this file from the models folder
# cd models; python3 tree.py
import csv
from typing import Tuple
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import sys
import json
sys.path.append('../')
from clean.number_cleaning import process_string
from clean.onehot_cleaning import process_multihot_non_matrix, process_onehot
from clean.drink_cleaning import process_drink, parse_common_drinks, parse_drinks_list
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree, DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
from inference_naive_bayes import predict, predict_smart
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

q3_options = ['Week day lunch', 'Week day dinner', 'Weekend lunch', 'Weekend dinner', 'At a party', 'Late night snack']
q7_options = ['Friends', 'Teachers', 'Siblings', 'Parents', 'None', 'Strangers']
q8_options = ['None', 'A little (mild)', 'A moderate amount (medium)', 'A lot (hot)',
              'I will have some of this food with my hot sauce']

X_train, t_train = None, None
X_val, t_val     = None, None
X_test, t_test   = None, None

# These flags enable/disable Naive Bayes chaining and
# choose one of the models: decision tree, random forest, gradient boosting classifier.
chaining = True # enables/disables Naive Bayes chaining
decision_tree = True
random_forest = False
gradient_boost = False
train = True # if False, builds the tree from the known hyperparameters
save_tree_flag = True

def get_data(data_path): 
    file = open(data_path, mode='r', encoding='utf-8')
    reader = csv.reader(file)
    data = np.array(list(reader))
    file.close()

    data = data[1:, :]  # Remove first question row
    train_data, val_data, test_data = split_data(data)

    #import pandas as pd 
    #test_data = test_data[:, -1]
    #df = pd.DataFrame(test_data)
    #df.to_csv("test_targets.csv", index=False)
    #exit()

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

    if chaining:
        q9 = np.zeros((data.shape[0],))
        q10 = np.zeros((data.shape[0],))
        q11 = np.zeros((data.shape[0],))
    t = data[:, 9]   # Label

    for i in range(data.shape[0]):
        # naive_bayes_q = ''
        # for j in range(1, 9):
        #     naive_bayes_q += data[i][j]
        #     naive_bayes_q += '.'
        naive_bayes_data = data[i]
        _, logits = predict_smart(naive_bayes_data, model_dir='../saved_model', verbose=False)
        if chaining:
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
    if chaining:
        columns.extend([np.reshape(q9, (q9.shape[0], 1))])
        columns.extend([np.reshape(q10, (q10.shape[0], 1))])
        columns.extend([np.reshape(q11, (q11.shape[0], 1))])

    return np.hstack(columns), t

def build_tree(criterion, d, s):
    # select the model here
    if decision_tree:
      return DecisionTreeClassifier(criterion=criterion, max_depth=d, min_samples_split=s)
    if random_forest:  
        return RandomForestClassifier(criterion=criterion, max_depth=d, min_samples_split=s)
    if gradient_boost:
        return GradientBoostingClassifier(criterion=criterion, max_depth=d, min_samples_split=s)

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
    if not gradient_boost:
        criterions = ['entropy', 'gini', 'log_loss']
    else:
        criterions = ['squared_error', 'friedman_mse']
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


def convert_regressor(regressor):
    """
    Converts one of the estimators inside the gradient boosting classifier to JSON.
    """
    tree = regressor.tree_

    v = tree.value.tolist()
    for i in range(len(v)):
        v[i] = np.squeeze(v[i]).max()

    return {
        "children_left": tree.children_left.tolist(),
        "children_right": tree.children_right.tolist(),
        "feature": tree.feature.tolist(),
        "threshold": tree.threshold.tolist(),
        "value": v
    }

def save_gbc(gbc):
    """
    Writes gradient boosting classifier to JSON file.
    """

    tree_list = []
    # 3 classes: pizza, sushi, shawarma
    for c in range(0, 3):
        trees = [convert_regressor(est) for est in gbc.estimators_[c]]
        tree_list.append(trees)
    
    model_data = {
        "n_classes": gbc.n_classes_,
        "learning_rate": gbc.learning_rate,
        "classes": gbc.classes_.tolist(),
        "tree_list": tree_list
    }

    # Save as JSON
    with open("gbc_model.json", "w") as f:
        json.dump(model_data, f)

def train_tree():
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
    d, s, criterion = 5, 2, 'squared_error'
    tree = build_tree(criterion, d, s)
    tree.fit(X_train, t_train)

    np.set_printoptions(threshold=np.inf)
    print(tree.score(X_test, t_test))
    
    labels = np.unique(t_test)
    c = confusion_matrix(tree.predict(X_test), t_test, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=c, display_labels=labels)
    disp.plot()
    plt.show()

    return tree

def save_tree(tree):
    if decision_tree or random_forest:
        raise NotImplementedError

    save_gbc(tree)

def save_decision_tree(decision_tree):
    tree = decision_tree.tree_

    v = tree.value.tolist()

    model_data = {
        "children_left": tree.children_left.tolist(),
        "children_right": tree.children_right.tolist(),
        "feature": tree.feature.tolist(),
        "threshold": tree.threshold.tolist(),
        "value": v
    }

    with open("decision_model.json", "w") as f:
        json.dump(model_data, f)

def test_classifier(classifier):
    #estimators = classifier.estimators_
    #for c in range(3):
    #    for j in range(len(estimators)):
    #        d = estimators[j][0] # d = decision tree regressor
    #        print(d)
    # tree = estimators[0][0].tree_
    
    # print({
    #     "children_left": tree.children_left.tolist(),
    #     "children_right": tree.children_right.tolist(),
    #     "feature": tree.feature.tolist(),
    #     "threshold": tree.threshold.tolist(),
    #     "value": tree.value.tolist(),
    #     "classes": classifier.classes_.tolist(),
    # })

    
    #plot_tree(estimators[0][0], fontsize=7)
    print(classifier.tree_)
    #plot_tree(classifier, fontsize=7)
    #plt.show()
    

#def make_prediction():
#    visualize_tree(tree)
#    X, t = 

if __name__ == '__main__':
    np.random.seed(0)

    get_data('../data/cleaned_data_combined_modified.csv')

    if train:
        tree = train_tree()
    else:
        tree = build_best_tree()
    
    if save_tree_flag:
        save_decision_tree(tree)
    test_classifier(tree)