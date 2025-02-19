# -*- coding: utf-8 -*-
import numpy as np
import random
import matplotlib.pyplot as plt

class MLPModel(object):
    def __init__(self, num_features=8, num_hidden=50, num_classes=3):
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.num_classes = num_classes

        # Weights and biases
        self.W1 = np.random.normal(0, 2/num_features, (num_hidden, num_features))
        self.b1 = np.random.normal(0, 2/num_features, num_hidden)
        self.W2 = np.random.normal(0, 2/num_hidden, (num_classes, num_hidden))
        self.b2 = np.random.normal(0, 2/num_hidden, num_classes)
        
        # Initialize intermediate variables
        self.cleanup()

    def cleanup(self):
        self.X = None
        self.m = None
        self.h = None
        self.z = None
        self.y = None
        self.W1_bar = None
        self.b1_bar = None
        self.W2_bar = None
        self.b2_bar = None

    def forward(self, X):
        self.X = X
        # Hidden layer
        self.m = X @ self.W1.T + self.b1
        self.h = np.maximum(0, self.m)  # ReLU activation
        # Output layer
        self.z = self.h @ self.W2.T + self.b2
        self.y = self.softmax(self.z)
        return self.y

    def backward(self, ts):
        # Backpropagation
        self.z_bar = (self.y - ts) / self.X.shape[0]
        self.W2_bar = self.z_bar.T @ self.h
        self.b2_bar = np.sum(self.z_bar, axis=0)
        
        h_bar = self.z_bar @ self.W2
        self.m_bar = h_bar * (self.m > 0)  # ReLU derivative
        
        self.W1_bar = self.m_bar.T @ self.X
        self.b1_bar = np.sum(self.m_bar, axis=0)

    def update(self, alpha):
        self.W1 -= alpha * self.W1_bar
        self.b1 -= alpha * self.b1_bar
        self.W2 -= alpha * self.W2_bar
        self.b2 -= alpha * self.b2_bar

    @staticmethod
    def softmax(z):
        z_max = np.max(z, axis=1, keepdims=True)
        z_exp = np.exp(z - z_max)
        return z_exp / np.sum(z_exp, axis=1, keepdims=True)

    def predict(self, X):
        y = self.forward(X)
        return np.argmax(y, axis=1)

def make_onehot(indices, total):
    return np.eye(total)[indices]

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

# Training parameters
N_TRAIN = 2000
N_VALID = 500
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.1

# Generate and split data
X_train, t_train = generate_data(N_TRAIN)
X_valid, t_valid = generate_data(N_VALID)

# Initialize model
model = MLPModel(num_features=8, num_hidden=50, num_classes=3)

# Training loop
train_loss = []
valid_acc = []

for epoch in range(EPOCHS):
    # Shuffle training data
    indices = np.random.permutation(N_TRAIN)
    X_train = X_train[indices]
    t_train = t_train[indices]
    
    # Mini-batch training
    epoch_loss = []
    for i in range(0, N_TRAIN, BATCH_SIZE):
        X_batch = X_train[i:i+BATCH_SIZE]
        t_batch = make_onehot(t_train[i:i+BATCH_SIZE], 3)
        
        # Forward and backward pass
        model.forward(X_batch)
        model.backward(t_batch)
        model.update(LEARNING_RATE)
        
        # Calculate loss
        loss = -np.sum(t_batch * np.log(model.y + 1e-8)) / BATCH_SIZE
        epoch_loss.append(loss)
    
    # Validation
    val_pred = model.predict(X_valid)
    accuracy = np.mean(val_pred == t_valid)
    
    train_loss.append(np.mean(epoch_loss))
    valid_acc.append(accuracy)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss[-1]:.4f} | Val Acc: {accuracy:.4f}")

# Plot training curve
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss)
plt.title("Training Loss")
plt.xlabel("Epoch")

plt.subplot(1, 2, 2)
plt.plot(valid_acc)
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.show()

# Example prediction
test_sample = np.random.randn(1, 8)
pred = model.predict(test_sample)
foods = ["Pizza", "Shawarma", "Sushi"]
print(f"Predicted food class: {foods[pred[0]]}")