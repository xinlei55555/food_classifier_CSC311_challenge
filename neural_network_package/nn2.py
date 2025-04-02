import numpy as np
import pandas as pd
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Activation Functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, alpha=0.005, l2_lambda=0.05):
        self.alpha = alpha  # Learning rate
        self.l2_lambda = l2_lambda  # L2 regularization strength
        
        # Layer sizes
        self.input_size = input_size
        self.hidden1_size, self.hidden2_size = hidden_sizes
        self.output_size = output_size

        # Initialize weights with He initialization
        self.W1 = np.random.randn(self.input_size, self.hidden1_size) * np.sqrt(2 / self.input_size)
        self.b1 = np.zeros((1, self.hidden1_size))
        
        self.W2 = np.random.randn(self.hidden1_size, self.hidden2_size) * np.sqrt(2 / self.hidden1_size)
        self.b2 = np.zeros((1, self.hidden2_size))
        
        self.W3 = np.random.randn(self.hidden2_size, self.output_size) * np.sqrt(2 / self.hidden2_size)
        self.b3 = np.zeros((1, self.output_size))

        # Adam optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0
        
        # Initialize Adam moment estimates
        self.m_W1, self.v_W1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.m_b1, self.v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.m_W2, self.v_W2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.m_b2, self.v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.m_W3, self.v_W3 = np.zeros_like(self.W3), np.zeros_like(self.W3)
        self.m_b3, self.v_b3 = np.zeros_like(self.b3), np.zeros_like(self.b3)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = relu(self.z2)

        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.y_hat = softmax(self.z3)
        return self.y_hat

    def backward(self, X, y):
        m = X.shape[0]
        self.t += 1

        # Output layer error
        dZ3 = self.y_hat - y
        dW3 = (np.dot(self.a2.T, dZ3) / m) + (self.l2_lambda * self.W3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        # Hidden layer 2 error
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * relu_derivative(self.z2)
        dW2 = (np.dot(self.a1.T, dZ2)) / m + (self.l2_lambda * self.W2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Hidden layer 1 error
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_derivative(self.z1)
        dW1 = (np.dot(X.T, dZ1)) / m + (self.l2_lambda * self.W1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Adam updates for weights
        for param, d_param, m, v in zip(
            [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3],
            [dW1, db1, dW2, db2, dW3, db3],
            [self.m_W1, self.m_b1, self.m_W2, self.m_b2, self.m_W3, self.m_b3],
            [self.v_W1, self.v_b1, self.v_W2, self.v_b2, self.v_W3, self.v_b3]
        ):
            m[:] = self.beta1 * m + (1 - self.beta1) * d_param
            v[:] = self.beta2 * v + (1 - self.beta2) * (d_param ** 2)
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            param -= self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def compute_loss(self, X, y):
        y_hat = self.forward(X)
        return -np.sum(y * np.log(y_hat + 1e-9)) / y.shape[0] + (
            self.l2_lambda * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2) + np.sum(self.W3 ** 2))) / (2 * y.shape[0])

    def train(self, X_train, y_train, X_val, y_val, epochs=500, batch_size=32, patience=10):
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Shuffle training data
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            # Mini-batch training
            epoch_train_loss = 0
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                self.forward(X_batch)
                self.backward(X_batch, y_batch)
                epoch_train_loss += self.compute_loss(X_batch, y_batch) * X_batch.shape[0]

            # Calculate metrics
            train_loss = epoch_train_loss / X_train.shape[0]
            val_loss = self.compute_loss(X_val, y_val)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best weights
                self.best_W1 = np.copy(self.W1)
                self.best_W2 = np.copy(self.W2)
                self.best_W3 = np.copy(self.W3)
                self.best_b1 = np.copy(self.b1)
                self.best_b2 = np.copy(self.b2)
                self.best_b3 = np.copy(self.b3)
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == epochs-1:
                print(f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best weights
        self.W1 = self.best_W1
        self.W2 = self.best_W2
        self.W3 = self.best_W3
        self.b1 = self.best_b1
        self.b2 = self.best_b2
        self.b3 = self.best_b3

    def predict(self, X):
        y_hat = self.forward(X)
        return np.argmax(y_hat, axis=1)

# Load and prepare data
X = np.load('../data_cleaning/X_clean.npy')
t = np.load('../data_cleaning/t_clean.npy')

# Standardize features
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / (std + 1e-8)

# Shuffle data before splitting
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
t = t[indices]

# Convert to one-hot
num_classes = 3
t_one_hot = np.eye(num_classes)[t]

# Split data
split = int(0.8 * X.shape[0])
X_train, X_valid = X[:split], X[split:]
t_train, t_valid = t_one_hot[:split], t_one_hot[split:]

# Handle class imbalance
class_counts = np.bincount(t)
class_weights = 1 / class_counts
class_weights /= class_weights.sum()
sample_weights = class_weights[t]

# Initialize and train model
input_size = X_train.shape[1]
hidden_sizes = (128, 64)
output_size = num_classes

nn = NeuralNetwork(
    input_size=input_size,
    hidden_sizes=hidden_sizes,
    output_size=output_size,
    alpha=0.01,
    l2_lambda=0.3
)

nn.train(
    X_train=X_train,
    y_train=t_train,
    X_val=X_valid,
    y_val=t_valid,
    epochs=300,
    batch_size=16,
    patience=10
)

# Evaluate
train_preds = nn.predict(X_train)
valid_preds = nn.predict(X_valid)

train_acc = np.mean(train_preds == np.argmax(t_train, axis=1))
valid_acc = np.mean(valid_preds == np.argmax(t_valid, axis=1))

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {valid_acc:.4f}")

# Save best model
np.save('W1.npy', nn.W1)
np.save('b1.npy', nn.b1)
np.save('W2.npy', nn.W2)
np.save('b2.npy', nn.b2)
np.save('W3.npy', nn.W3)
np.save('b3.npy', nn.b3)
# Add these to your training script after training
np.save('mean.npy', mean)
np.save('std.npy', std)

def predict_new(X_test):
    # Standardize using training stats
    X_test = (X_test - mean) / (std + 1e-8)
    
    # Load weights
    nn = NeuralNetwork(
        input_size=X_test.shape[1],
        hidden_sizes=hidden_sizes,
        output_size=output_size
    )
    nn.W1, nn.b1 = np.load('W1.npy'), np.load('b1.npy')
    nn.W2, nn.b2 = np.load('W2.npy'), np.load('b2.npy')
    nn.W3, nn.b3 = np.load('W3.npy'), np.load('b3.npy')
    
    preds = nn.predict(X_test)
    food_labels = ["Pizza", "Shawarma", "Sushi"]
    return [food_labels[p] for p in preds]

# Example test
X_test_sample = X_valid[:5]
print("Predictions:", predict_new(X_test_sample))