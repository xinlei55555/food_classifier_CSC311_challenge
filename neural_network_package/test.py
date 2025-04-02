import numpy as np
from models.neural_network_model2 import NeuralNetwork

class NeuralNetworkLoader:
    def __init__(self, input_size):
        # Initialize architecture with best parameters
        self.model = NeuralNetwork(
            input_size=input_size,
            hidden_sizes=(256, 128, 64),
            output_size=3,
            alpha=0.1,
            lr=0.0005
        )
        
        # Load weights
        weights = np.load('best_model_weights.npz')
        self.model.params = {
            'W1': weights['W1'],
            'b1': weights['b1'],
            'W2': weights['W2'],
            'b2': weights['b2'],
            'W3': weights['W3'],
            'b3': weights['b3']
        }

    def predict(self, X):
        return self.model.predict(X)

if __name__ == "__main__":
    # Load and preprocess test data
    X = np.load('../data_cleaning/X_clean.npy')
    t = np.load('../data_cleaning/t_clean.npy')
    
    # Apply preprocessing
    train_mean = np.load('train_mean.npy')
    train_std = np.load('train_std.npy')
    X_test = (X - train_mean) / (train_std + 1e-8)
    t_test = np.eye(3)[t.astype(int)]
    
    # Initialize and test
    loader = NeuralNetworkLoader(X_test.shape[1])
    preds = loader.predict(X_test)
    
    # Calculate accuracy
    accuracy = np.mean(preds == np.argmax(t_test, axis=1))
    print(f"Test Accuracy: {accuracy:.4f}")