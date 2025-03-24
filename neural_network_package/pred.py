import numpy as np
import pandas as pd
from number_cleaning import process_string
from onehot_cleaning import process_multihot, process_onehot
from drink_cleaning import process_drink, parse_common_drinks

# Activation Functions
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Categorical encoding options
q3_options = ['Week day lunch', 'Week day dinner', 'Weekend lunch', 
             'Weekend dinner', 'At a party', 'Late night snack']
q7_options = ['Friends', 'Teachers', 'Siblings', 'Parents', 'None', 'Strangers']
q8_options = ['None', 'A little (mild)', 'A moderate amount (medium)', 
             'A lot (hot)', 'I will have some of this food with my hot sauce']
drink_categories = [
    "diet-coke", "zero", "coke", "mango", "barbican", "none", "sparkling",
    "boba", "milk-shake", "smoothie", "dr-pepper", "yoghurt", "yakult",
    "crush", "root-beer", "ramune", "coffee", "iced-tea", "gingerale", "tea",
    "saporo", "soy-sauce", "yuzu", "jarritos", "baijiu", "water", "juice",
    "wine", "beer", "gatorade", "soju", "sake", "soup", "calpis", "milk",
    "lemonade", "pepsi", "soda", "sprite", "fanta", "mountain-dew", "alcohol"
]

class PredictionNetwork:
    def __init__(self):
        # Load saved parameters
        self.W1 = np.load('W1.npy')
        self.b1 = np.load('b1.npy')
        self.W2 = np.load('W2.npy')
        self.b2 = np.load('b2.npy')
        self.W3 = np.load('W3.npy')
        self.b3 = np.load('b3.npy')
        self.mean = np.load('mean.npy')
        self.std = np.load('std.npy')

    def forward(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = relu(z2)
        z3 = np.dot(a2, self.W3) + self.b3
        return softmax(z3)

    def predict(self, X):
        X_std = (X - self.mean) / (self.std + 1e-8)
        y_hat = self.forward(X_std)
        return np.argmax(y_hat, axis=1)

def predict_all(csv_filename):
    # Load and preprocess data
    df = pd.read_csv(csv_filename)
    data = df.values
    
    # Process columns using imported functions
    q1 = data[:, 1].astype(float)
    # if np.issubdtype(data[:, 2].dtype, np.number):
    #     q2 = np.array(data[:, 2], dtype=float)
    # else:
    #     q2 = np.vectorize(process_string)(data[:, 2])
    q2 = np.array(data[:, 2], dtype=float)
    q3 = np.array([process_multihot(str(x), q3_options) for x in data[:, 3]])
    q4 = np.vectorize(process_string)(data[:, 4])
    
    # Process drinks
    common_drinks = parse_common_drinks()
    processed_drinks = [process_drink(x, common_drinks) for x in data[:, 6]]
    q6 = np.array([process_onehot(drink, drink_categories) for drink in processed_drinks])
    
    q7 = np.array([process_multihot(str(x), q7_options) for x in data[:, 7]])
    q8 = np.array([process_onehot(str(x), q8_options) for x in data[:, 8]])

    # Combine features
    X_processed = np.column_stack((
        q1.reshape(-1, 1),
        q2.reshape(-1, 1),
        q3.reshape(q3.shape[0], -1),
        q4.reshape(-1, 1),
        q6.reshape(len(q6), -1),
        q7.reshape(q7.shape[0], -1),
        q8.reshape(q8.shape[0], -1)
    )).astype(np.float64)
    
    # Make predictions
    network = PredictionNetwork()
    preds = network.predict(X_processed)
    
    # Convert to labels
    food_labels = ["Pizza", "Shawarma", "Sushi"]
    return [food_labels[p] for p in preds]

# Required supporting files:
# W1.npy, b1.npy, W2.npy, b2.npy, W3.npy, b3.npy
# mean.npy, std.npy