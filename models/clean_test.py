import csv
import numpy as np
from number_cleaning import process_string
from onehot_cleaning import process_multihot, process_onehot
from drink_cleaning import process_drink, parse_common_drinks

csv_path = 'food_val.csv'

# Define categorical options for encoding
q3_options = ['Week day lunch', 'Week day dinner', 'Weekend lunch', 'Weekend dinner', 'At a party', 'Late night snack']
q7_options = ['Friends', 'Teachers', 'Siblings', 'Parents', 'None', 'Strangers']
q8_options = ['None', 'A little (mild)', 'A moderate amount (medium)', 'A lot (hot)',
              'I will have some of this food with my hot sauce']
drink_categories = [
    "diet-coke", "zero", "coke", "mango", "barbican", "none", "sparkling", "boba", 
    "milk-shake", "smoothie", "dr-pepper", "yoghurt", "yakult", "crush", "root-beer", 
    "ramune", "coffee", "iced-tea", "gingerale", "tea", "saporo", "soy-sauce", "yuzu", 
    "jarritos", "baijiu", "water", "juice", "wine", "beer", "gatorade", "soju", "sake", 
    "soup", "calpis", "milk", "lemonade", "pepsi", "soda", "sprite", "fanta", 
    "mountain-dew", "alcohol"
]

def one_hot_encode_drink(drink, categories):
    """Convert a mapped drink category into a one-hot encoded vector."""
    vector = np.zeros(len(categories))  # Initialize zero vector
    if drink in categories:
        index = categories.index(drink)  # Find index of drink category
        vector[index] = 1  # Set corresponding index to 1
    return vector

if __name__ == '__main__':
    file = open(csv_path, mode='r', encoding='utf-8')
    reader = csv.reader(file)
    data = np.array(list(reader))
    file.close()

    common_drinks = parse_common_drinks()
    process_string_vectorized = np.vectorize(process_string)

    data = data[1:, :]  # Remove header row

    # Extract raw columns
    q1 = data[:, 1]  # Numeric
    q2 = process_string_vectorized(data[:, 2])  # Numeric
    q3 = np.array([process_multihot(x, q3_options) for x in data[:, 3]])  # Multi-hot
    q4 = process_string_vectorized(data[:, 4])  # Numeric
    q5 = data[:, 5]  # Categorical, needs encoding
    q6 = np.array([process_drink(x, common_drinks) for x in data[:, 6]])  # Encoded drinks
    q6 = np.array([one_hot_encode_drink(drink, drink_categories) for drink in q6])
    q7 = np.array([process_multihot(x, q7_options) for x in data[:, 7]])  # Multi-hot
    q8 = np.array([process_onehot(x, q8_options) for x in data[:, 8]])  # One-hot
    t = data[:, 9]  # Labels (Pizza, Shawarma, Sushi)

    # Convert labels to numeric
    label_map = {'Pizza': 0, 'Shawarma': 1, 'Sushi': 2}
    t_numeric = np.array([label_map[label] for label in t])

    # Stack all features into a single numeric dataset
    X_processed = np.column_stack((
    q1.reshape(-1, 1),  # Ensure 2D
    q2.reshape(-1, 1),  # Ensure 2D
    q3.reshape(q3.shape[0], -1),  # Flatten multi-hot encoding
    q4.reshape(-1, 1),  # Ensure 2D
    q6.reshape(len(q6), -1),  # Ensure 2D
    q7.reshape(q7.shape[0], -1),  # Flatten multi-hot encoding
    q8.reshape(q8.shape[0], -1)   # Flatten one-hot encoding
))


    # Save cleaned data
    X_processed = X_processed.astype(np.float64)
    t_numeric = t_numeric.astype(np.int64)
    np.save('X_clean_val.npy', X_processed)
    np.save('t_clean_val.npy', t_numeric)
    print(X_processed[:5])
    print(q1[:5])
    print(q2[:5])
    print(q3[:5])
    print(q4[:5])
    print(q6[:5])
    print(q7[:5])
    print(q8[:5])
    print(t_numeric[:5])
