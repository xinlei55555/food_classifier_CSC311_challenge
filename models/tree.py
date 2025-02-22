from typing import List, Tuple
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree as treeViz
import graphviz
import pydotplus
from IPython.display import display

# Generate synthetic dataset
def generate_data(n_samples):
    # 8 features with different distributions for each food class
    X = np.random.randn(n_samples, 8)
    t = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        # Simple pattern-based classification (replace with real data)
        if np.sum(X[i, :4]) > 0.5:  # Pizza pattern
            t[i] = 0
        elif np.sum(X[i, 4:]) < -0.2:  # Shawarma pattern
            t[i] = 1
        else:  # Sushi pattern
            t[i] = 2
    return X, t

def build_tree(criterion, d, s):
    return DecisionTreeClassifier(criterion=criterion, max_depth=d, min_samples_split=s)

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
    criterions = ["entropy", "gini"]
    max_depths = [1, 5, 10, 15, 20, 25, 30, 50, 100]
    min_samples_split = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    # we found that best_hyperparams = [(max_depth, min_samples_split), val_accuracy, criterion]
    best_hyperparams = [(-1, -1), -1, 'entropy']
    for criterion in criterions:
        print("\nUsing criterion {}".format(criterion))
        res = score_models(max_depths, min_samples_split, criterion, data)
    
        for d, s in res:
            if best_hyperparams[1] < res[(d, s)]['val']:
                best_hyperparams = [(d, s), res[(d, s)]['val'], criterion, res[(d, s)]['tree']]
                print("Best hyperparameters updated: ", best_hyperparams)

    d, s = best_hyperparams[0]
    criterion = best_hyperparams[2]
    tree = best_hyperparams[3]

    return d, s, criterion, tree

def visualize_tree(tree):
    dot_data = treeViz.export_graphviz(tree,
                                       max_depth=3,
                                       class_names=["pizza", "shawarma", "sushi"],
                                       filled=True,
                                       rounded=True)
    graphviz.Source(dot_data).view(cleanup=True)

if __name__ == '__main__':
    np.random.seed(0)

    X, t = generate_data(8000)
    X_tv, X_test, t_tv, t_test = train_test_split(X, t, test_size=1500/8000, random_state=1)
    X_train, X_valid, t_train, t_valid = train_test_split(X_tv, t_tv, test_size=1500/6500, random_state=1)

    d, s, criterion, fitted_tree = grid_search((X_train, X_valid, t_train, t_valid))
    visualize_tree(fitted_tree)