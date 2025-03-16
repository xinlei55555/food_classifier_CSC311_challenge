import csv
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree as treeViz
import graphviz
import sys
sys.path.append('../')
from clean.number_cleaning import process_string
from clean.onehot_cleaning import process_multihot_non_matrix, process_onehot
from clean.drink_cleaning import process_drink, parse_common_drinks, parse_drinks_list

q3_options = ['Week day lunch', 'Week day dinner', 'Weekend lunch', 'Weekend dinner', 'At a party', 'Late night snack']
q7_options = ['Friends', 'Teachers', 'Siblings', 'Parents', 'None', 'Strangers']
q8_options = ['None', 'A little (mild)', 'A moderate amount (medium)', 'A lot (hot)',
              'I will have some of this food with my hot sauce']

def train_decision_tree(data_path): 
    file = open(data_path, mode='r', encoding='utf-8')
    reader = csv.reader(file)
    data = np.array(list(reader))
    file.close()

    common_drinks = parse_common_drinks(input_simple_file="../clean/common_drinks.simple")
    drinks_list = parse_drinks_list(input_simple_file="../clean/common_drinks.simple")
    process_string_vectorized = np.vectorize(process_string)

    data = data[1:, :]  # Remove first question row

    q1 = data[:, 1]  # How complex? (1-5)
    q2 = data[:, 2]  # How many ingredients? (open ended, numeric)
    q3 = data[:, 3]  # In what setting? (multi select)
    q4 = data[:, 4]  # Cost? (open ended, numeric)
    q5 = data[:, 5]  # Movie? (open ended, any)
    q6 = data[:, 6]  # Drink? (open ended, drink)
    q7 = data[:, 7]  # Who does it remind you of? (multi select)
    q8 = data[:, 8]  # How much hot sauce? (single select)
    t = data[:, 9]   # Label

    q2p = process_string_vectorized(q2)
    q3p = np.array([process_multihot_non_matrix(x, q3_options) for x in q3])
    q4p = process_string_vectorized(q4)
    
    q6p = np.array([process_drink(x, common_drinks) for x in q6])
    q6phot = np.array([process_multihot_non_matrix(x, drinks_list) for x in q6p])

    #np.set_printoptions(threshold=np.inf)

    q7p = np.array([process_multihot_non_matrix(x, q7_options) for x in q7])
    q8p = np.array([process_onehot(x, q8_options) for x in q8])
    
    columns = []
    columns.extend([np.reshape(q2p, (q2p.shape[0], 1))])
    columns.extend(np.hsplit(q3p, q3p.shape[1]))
    columns.extend([np.reshape(q4p, (q4p.shape[0], 1))])
    columns.extend(np.hsplit(q6phot, q6phot.shape[1]))
    columns.extend(np.hsplit(q7p, q7p.shape[1]))
    columns.extend(np.hsplit(q8p, q8p.shape[1]))

    training = np.hstack(columns)
    #np.set_printoptions(threshold=np.inf)
    print(training)
    

def build_tree(criterion, d, s):
    return DecisionTreeClassifier(criterion=criterion, max_depth=d, min_samples_split=s)

def visualize_tree(tree):
    dot_data = treeViz.export_graphviz(tree,
                                       max_depth=3,
                                       class_names=["pizza", "shawarma", "sushi"],
                                       filled=True,
                                       rounded=True)
    graphviz.Source(dot_data).view(cleanup=True)

if __name__ == '__main__':
    np.random.seed(0)

    train_decision_tree('../data/cleaned_data_combined_modified.csv')