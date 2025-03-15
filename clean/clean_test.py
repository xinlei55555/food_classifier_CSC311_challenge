import csv
import numpy as np
from number_cleaning import process_string
from onehot_cleaning import process_multihot, process_onehot
from drink_cleaning import process_drink, parse_common_drinks

csv_path = '../datasets/cleaned_data_combined_modified.csv'
q3_options = ['Week day lunch', 'Week day dinner', 'Weekend lunch', 'Weekend dinner', 'At a party', 'Late night snack']
q7_options = ['Friends', 'Teachers', 'Siblings', 'Parents', 'None', 'Strangers']
q8_options = ['None', 'A little (mild)', 'A moderate amount (medium)', 'A lot (hot)',
              'I will have some of this food with my hot sauce']

if __name__ == '__main__':
    file = open(csv_path, mode='r', encoding='utf-8')
    reader = csv.reader(file)
    data = np.array(list(reader))
    file.close()

    common_drinks = parse_common_drinks()
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
    q3p = np.array([process_multihot(x, q3_options) for x in q3])
    q4p = process_string_vectorized(q4)
    q6p = np.array([process_drink(x, common_drinks) for x in q6])
    q7p = np.array([process_multihot(x, q7_options) for x in q7])
    q8p = np.array([process_onehot(x, q8_options) for x in q8])
