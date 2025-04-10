import numpy as np
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
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes=(256, 128, 64), output_size=3, alpha=0.1, lr=0.0005):
        # Best parameters from grid search
        self.alpha = alpha  # L2 regularization
        self.lr = lr         # Learning rate
        self.batch_size = 16
        
        # Layer architecture
        self.layer_sizes = [input_size] + list(hidden_sizes) + [output_size]
        
        # Adam parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        
        # Initialize weights and Adam states
        self.params = {}
        self.m = {}
        self.v = {}
        for i in range(1, len(self.layer_sizes)):
            scale = np.sqrt(2 / self.layer_sizes[i-1])
            self.params[f'W{i}'] = np.random.randn(
                self.layer_sizes[i-1], self.layer_sizes[i]) * scale
            self.params[f'b{i}'] = np.zeros((1, self.layer_sizes[i]))
            
            self.m[f'W{i}'] = self.m[f'b{i}'] = 0
            self.v[f'W{i}'] = self.v[f'b{i}'] = 0

    def forward(self, X, training=True):
        """Forward pass with dropout"""
        self.cache = {'A0': X}
        
        for i in range(1, len(self.layer_sizes)):
            # Linear transformation
            z = np.dot(self.cache[f'A{i-1}'], self.params[f'W{i}']) + self.params[f'b{i}']
            
            # Tanh activation for hidden layers
            if i < len(self.layer_sizes)-1:
                a = tanh(z)
                # Dropout (20% rate)
                if training:
                    mask = (np.random.rand(*a.shape) < 0.8) / 0.8
                    a *= mask
                self.cache[f'A{i}'] = a
            else:
                self.cache[f'A{i}'] = softmax(z)
        
        return self.cache[f'A{len(self.layer_sizes)-1}']

    def backward(self, X, y, t):
        """Adam optimizer with L2 regularization"""
        grads = {}
        m = X.shape[0]
        
        # Output layer gradient
        dZ = self.cache[f'A{len(self.layer_sizes)-1}'] - y
        grads[f'W{len(self.layer_sizes)-1}'] = np.dot(
            self.cache[f'A{len(self.layer_sizes)-2}'].T, dZ) / m
        grads[f'b{len(self.layer_sizes)-1}'] = np.sum(dZ, axis=0, keepdims=True) / m
        
        # Backprop through hidden layers
        for i in reversed(range(1, len(self.layer_sizes)-1)):
            dA = np.dot(dZ, self.params[f'W{i+1}'].T)
            dZ = dA * tanh_derivative(self.cache[f'A{i}'])
            grads[f'W{i}'] = np.dot(self.cache[f'A{i-1}'].T, dZ) / m
            grads[f'b{i}'] = np.sum(dZ, axis=0, keepdims=True) / m
        
        # Adam update with L2 regularization
        for key in self.params:
            if 'W' in key:  # Apply L2 regularization to weights only
                grads[key] += self.alpha * self.params[key]
            
            # Update moments
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            
            # Bias-corrected moments
            m_hat = self.m[key] / (1 - self.beta1 ** t)
            v_hat = self.v[key] / (1 - self.beta2 ** t)
            
            # Update parameters
            self.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def train(self, X, y, epochs=500):
        """Training with learning rate decay"""
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            # Learning rate decay
            self.lr = 0.0005 * np.exp(-0.001 * epoch)
            
            # Shuffle data
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch training
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                # Forward and backward pass
                self.forward(X_batch)
                self.backward(X_batch, y_batch, epoch+1)
            
            # Early stopping check
            val_loss = self.compute_loss(X_val, t_val)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            
            if epoch % 50 == 0:
                train_acc = self.evaluate(X_train, t_train)
                val_acc = self.evaluate(X_val, t_val)
                print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    def compute_loss(self, X, y):
        y_hat = self.forward(X, training=False)
        return -np.mean(y * np.log(y_hat + 1e-8))

    def evaluate(self, X, y):
        y_hat = self.forward(X, training=False)
        return np.mean(np.argmax(y_hat, axis=1) == np.argmax(y, axis=1))

    def predict(self, X):
        y_hat = self.forward(X, training=False)
        return np.argmax(y_hat, axis=1)

def load_and_preprocess():
    # Load and split data with standardization
    X = np.load('../data_cleaning/X_clean.npy')
    t = np.load('../data_cleaning/t_clean.npy')
    
    # Split data
    indices = np.random.permutation(X.shape[0])
    split = int(0.8 * X.shape[0])
    X_train, X_test = X[indices[:split]], X[indices[split:]]
    t_train, t_test = t[indices[:split]], t[indices[split:]]
    
    # Standardize and save params
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)
    np.save('train_mean.npy', train_mean)
    np.save('train_std.npy', train_std)
    
    return (X_train - train_mean)/(train_std+1e-8), \
           np.eye(3)[t_train.astype(int)], \
           (X_test - train_mean)/(train_std+1e-8), \
           np.eye(3)[t_test.astype(int)]

if __name__ == "__main__":
    # Load data and train
    X_train, t_train, X_test, t_test = load_and_preprocess()
    
    # Initialize best model
    model = NeuralNetwork(
        input_size=X_train.shape[1],
        hidden_sizes=(256, 128, 64),
        output_size=3,
        alpha=0.1,
        lr=0.0005
    )
    
    # Train and save weights
    model.train(X_train, t_train, epochs=500)
    np.savez('best_model_weights.npz',
             W1=model.params['W1'],
             b1=model.params['b1'],
             W2=model.params['W2'],
             b2=model.params['b2'],
             W3=model.params['W3'],
             b3=model.params['b3'])