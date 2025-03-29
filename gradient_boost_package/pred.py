import json
import numpy as np
from number_cleaning import process_string 
from onehot_cleaning import process_multihot, process_onehot
from drink_cleaning import process_drink, parse_common_drinks

chaining = False

def predict_single(tree, x):
    """Traverse a single gradient boost tree and return its output."""
    node = 0
    while tree["children_left"][node] != -1:
        feature = tree["feature"][node]
        threshold = tree["threshold"][node]
        if x[feature] <= threshold:
            node = tree["children_left"][node]
        else:
            node = tree["children_right"][node]
    return np.array(tree["value"][node]).squeeze()

def predict(X):
    """Predict using the extracted model."""
    X = np.array(X)
    learning_rate = model_data["learning_rate"]
    n_classes = model_data["n_classes"]
    
    # Initialize predictions
    predictions = np.zeros((X.shape[0], n_classes))
    
    # Aggregate predictions from all trees
    for tree_data in model_data["trees"]:
        for i, x in enumerate(X):
            predictions[i] += learning_rate * predict_single(tree_data, x)
    
    return np.argmax(predictions, axis=1)

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
        naive_bayes_q = ''
        for j in range(1, 9):
            naive_bayes_q += data[i][j]
            naive_bayes_q += '.'
        _, logits = predict(naive_bayes_q, model_dir='../saved_model', verbose=False)
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

if __name__ == '__main__':
    if chaining:
        raise NotImplementedError
    # Load JSON model
    with open("saved_model/gbc_model.json", "r") as f:
        model_data = json.load(f)

    print("hello")