import numpy as np
import pandas as pd
import random


# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Activation Functions
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Prevent overflow
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, alpha=0.01):
        """
        Initialize a simple neural network with two hidden layers.
        """
        self.alpha = alpha  # Learning rate

        # Layer sizes
        self.input_size = input_size
        self.hidden1_size, self.hidden2_size = hidden_sizes
        self.output_size = output_size

        # Initialize weights and biases using Xavier initialization
        self.W1 = np.random.randn(self.input_size, self.hidden1_size) * np.sqrt(1 / self.input_size)
        self.b1 = np.zeros((1, self.hidden1_size))

        self.W2 = np.random.randn(self.hidden1_size, self.hidden2_size) * np.sqrt(1 / self.hidden1_size)
        self.b2 = np.zeros((1, self.hidden2_size))

        self.W3 = np.random.randn(self.hidden2_size, self.output_size) * np.sqrt(1 / self.hidden2_size)
        self.b3 = np.zeros((1, self.output_size))

    def forward(self, X):
        """
        Forward pass of the network.
        """
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = tanh(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = tanh(self.z2)

        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.y_hat = softmax(self.z3)

        return self.y_hat

    def backward(self, X, y):
        """
        Backward propagation to compute gradients and update weights.
        """
        m = X.shape[0]  # Number of examples

        # Compute error at output layer
        dZ3 = self.y_hat - y
        dW3 = np.dot(self.a2.T, dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        # Compute error at second hidden layer
        dZ2 = np.dot(dZ3, self.W3.T) * tanh_derivative(self.z2)
        dW2 = np.dot(self.a1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Compute error at first hidden layer
        dZ1 = np.dot(dZ2, self.W2.T) * tanh_derivative(self.z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.W1 -= self.alpha * dW1
        self.b1 -= self.alpha * db1
        self.W2 -= self.alpha * dW2
        self.b2 -= self.alpha * db2
        self.W3 -= self.alpha * dW3
        self.b3 -= self.alpha * db3

    def train(self, X, y, epochs=100, batch_size=16):
        """
        Train the neural network using mini-batch gradient descent.
        """
        for epoch in range(epochs):
            # Shuffle data
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            # Mini-batch gradient descent
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                # Forward and backward pass
                self.forward(X_batch)
                self.backward(X_batch, y_batch)

            if epoch % 10 == 0 or epoch == epochs - 1:
                loss = -np.sum(y_batch * np.log(self.y_hat + 1e-9)) / y_batch.shape[0]
  # Cross-entropy loss
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        """
        Make predictions on new data.
        """
        y_hat = self.forward(X)
        return np.argmax(y_hat, axis=1)  # Return predicted class indices

# Load Processed Data
X = np.load('../data_cleaning/X_clean.npy')
t = np.load('../data_cleaning/t_clean.npy')

# Convert labels to one-hot encoding
num_classes = 3
t_one_hot = np.zeros((t.shape[0], num_classes))
t_one_hot[np.arange(t.shape[0]), t] = 1  # One-hot encode labels

# Split into training and validation sets
split = int(0.8 * X.shape[0])
X_train, X_valid = X[:split], X[split:]
t_train, t_valid = t_one_hot[:split], t_one_hot[split:]

# Initialize and Train Model
input_size = X_train.shape[1]
hidden_sizes = (64, 32)  # Best from GridSearchCV
output_size = num_classes
alpha = 0.01  # Learning rate from GridSearchCV
epochs = 500  # Number of iterations
batch_size = 16  # Batch size from GridSearchCV

nn = NeuralNetwork(input_size, hidden_sizes, output_size, alpha)
nn.train(X_train, t_train, epochs=epochs, batch_size=batch_size)

# Evaluate Accuracy
train_predictions = nn.predict(X_train)
valid_predictions = nn.predict(X_valid)

train_acc = np.mean(np.argmax(t_train, axis=1) == train_predictions)
valid_acc = np.mean(np.argmax(t_valid, axis=1) == valid_predictions)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {valid_acc:.4f}")

# Save Model Weights
np.save('W1.npy', nn.W1)
np.save('b1.npy', nn.b1)
np.save('W2.npy', nn.W2)
np.save('b2.npy', nn.b2)
np.save('W3.npy', nn.W3)
np.save('b3.npy', nn.b3)

# Function to Load Model and Make Predictions
def predict_new(X_test):
    """Load model weights and make predictions on new test data."""
    W1, b1 = np.load('W1.npy'), np.load('b1.npy')
    W2, b2 = np.load('W2.npy'), np.load('b2.npy')
    W3, b3 = np.load('W3.npy'), np.load('b3.npy')

    nn.W1, nn.b1 = W1, b1
    nn.W2, nn.b2 = W2, b2
    nn.W3, nn.b3 = W3, b3

    predictions = nn.predict(X_test)
    
    # Convert numeric labels back to food names
    food_labels = ["Pizza", "Shawarma", "Sushi"]
    return [food_labels[pred] for pred in predictions]

# Example Test Predictions
X_test_sample = X_valid[:5]  # Take first 5 validation samples
predictions = predict_new(X_test_sample)
print("Predictions:", predictions)
