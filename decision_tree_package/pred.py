import csv
import json
import numpy as np
from number_cleaning import process_string 
from onehot_cleaning import process_multihot, process_onehot, process_multihot_non_matrix
from drink_cleaning import process_drink, parse_common_drinks, parse_drinks_list
from inference_naive_bayes import predict_smart

chaining = True

q3_options = ['Week day lunch', 'Week day dinner', 'Weekend lunch', 'Weekend dinner', 'At a party', 'Late night snack']
q7_options = ['Friends', 'Teachers', 'Siblings', 'Parents', 'None', 'Strangers']
q8_options = ['None', 'A little (mild)', 'A moderate amount (medium)', 'A lot (hot)',
              'I will have some of this food with my hot sauce']

# # predicts from a weak learner
# def predict_gb_single(tree, x):
#     """Traverse a single gradient boost tree and return its output."""
#     print(x)
#     node = 0
#     while tree["children_left"][node] != -1 or tree["children_right"][node] != -1:
#         feature = tree["feature"][node]
#         threshold = tree["threshold"][node]
#         if x[feature] <= threshold:
#             node = tree["children_left"][node]
#         else:
#             node = tree["children_right"][node]
#     return tree["value"][node]

def predict_gb_single(t, x):
    node = 0
    while t["children_left"][node] != -1:  # Not a leaf
        if x[t["feature"][node]] <= t["threshold"][node]:
            node = t["children_left"][node]
        else:
            node = t["children_right"][node]
    return np.array(t["value"][node]).argmax()

def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / np.sum(e_x)

def predict_gb_multiple(X):
    """Predict using the extracted model."""
    global gb_data

    n_classes = gb_data["n_classes"]
    classes = gb_data["classes"]    
    tree_list = gb_data["tree_list"]
    
    predictions = np.zeros((X.shape[0], n_classes))

    for j, x in enumerate(X):
        for c in range(n_classes):
            for i, t in enumerate(tree_list[c]):
                predictions[j][c] += predict_gb_single(t, x)
    
    for j in range(X.shape[0]):
        predictions[j] = softmax(predictions[j])

    #f = lambda x: classes[x]
    res = np.argmax(predictions, axis=1)
    res = res.tolist()
    for i in range(len(res)):
        res[i] = classes[res[i]]
    
    return res

def parse_data(data):
    """
    Parses data from the CSV file and returns matrix X.
    """
    common_drinks = parse_common_drinks(input_simple_file="common_drinks.simple")
    drinks_list = parse_drinks_list(input_simple_file="common_drinks.simple")
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

    for i in range(data.shape[0]):
        # naive_bayes_q = ''
        # for j in range(1, 9):
        #     naive_bayes_q += data[i][j]
        #     naive_bayes_q += '.'
        naive_bayes_data = data[i]
        _, logits = predict_smart(naive_bayes_data, model_dir='.', verbose=False)
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

    return np.hstack(columns)

def get_data(data_path):
    """
    Gets and parses data from the CSV file.
    """
    file = open(data_path, mode='r', encoding='utf-8')
    reader = csv.reader(file)
    data = np.array(list(reader))
    file.close()

    data = data[1:, :]  # Remove first question row
    return parse_data(data)

def predict_all(csv_filename):
    # if chaining:
    #     raise NotImplementedError
    
    # # Load gradient boost model
    # global gb_data
    # with open("gbc_model.json", "r") as f:
    #     gb_data = json.load(f)

    # X = get_data(csv_filename)
    
    # return predict_gb_multiple(X)
    t = None
    with open("decision_model.json", "r") as f:
        t = json.load(f)
    X = get_data(csv_filename)
    result = []
    classes = ["Pizza", "Shawarma", "Sushi"]
    for x in X:
        result.append(classes[predict_gb_single(t, x)])
    return result


if __name__ == '__main__':
    # predictions = predict_all('../data/test.csv')
    # with open('../data/test_targets.csv', mode='r', encoding='utf-8') as file:
    #     reader = csv.reader(file)
    #     targets = list(reader)
    
    # correct = 0
    # for i in range(1, len(targets)):
    #     print(targets[i][0], predictions[i-1])
    #     if targets[i][0] == predictions[i-1]:
    #         correct += 1
    # print(float(correct) / len(targets))
    # #print(predictions)
    pass