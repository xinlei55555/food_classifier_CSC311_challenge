import numpy as np
import matplotlib.pyplot as plt

# Load preprocessed data
X_train = np.load('../data_cleaning/X_clean.npy')
t_train = np.load('../data_cleaning/t_clean.npy')
num_features = X_train.shape[1]  # Ensure correct number of input features

# Split data into training and validation sets
N_TRAIN = int(len(X_train) * 0.8)
X_valid, t_valid = X_train[N_TRAIN:], t_train[N_TRAIN:]
X_train, t_train = X_train[:N_TRAIN], t_train[:N_TRAIN]

class MLPModel(object):
    def __init__(self, num_features, num_hidden=50, num_classes=3):
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.num_classes = num_classes

        # Initialize weights
        self.W1 = np.random.normal(0, 2/num_features, (num_hidden, num_features))
        self.b1 = np.random.normal(0, 2/num_features, num_hidden)
        self.W2 = np.random.normal(0, 2/num_hidden, (num_classes, num_hidden))
        self.b2 = np.random.normal(0, 2/num_hidden, num_classes)

    def forward(self, X):
        self.m = X @ self.W1.T + self.b1
        self.h = np.maximum(0, self.m)  # ReLU activation
        self.z = self.h @ self.W2.T + self.b2
        return self.softmax(self.z)

    def softmax(self, z):
        z_max = np.max(z, axis=1, keepdims=True)
        z_exp = np.exp(z - z_max)
        return z_exp / np.sum(z_exp, axis=1, keepdims=True)

    def predict(self, X):
        y = self.forward(X)
        return np.argmax(y, axis=1)

# Train the model
num_features = X_train.shape[1]
model = MLPModel(num_features=num_features, num_hidden=50, num_classes=3)

# Example test prediction
test_sample = X_train[0].reshape(1, -1)
pred = model.predict(test_sample)
foods = ["Pizza", "Shawarma", "Sushi"]
predictions = [foods[pred] for pred in model.predict(X_valid)]
pred_X_valid = model.predict(X_valid)
print(f"Predictions: {predictions[:10]}")
accuracy = np.mean(pred_X_valid == [label for label in t_valid]) 
print(f"Validation accuracy: {accuracy:.2f}")
print(pred_X_valid[:100])
print(t_valid[:100])
