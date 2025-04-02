import numpy as np
import itertools
from nnB import NeuralNetwork
import random

random.seed(42)
np.random.seed(42)
# Load and preprocess data
X = np.load('X_clean_train.npy')
t = np.load('t_clean_train.npy')
X_valid = np.load('X_clean_val.npy')
t_valid = np.load('t_clean_val.npy')

# Shuffle data before splitting
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
t = t[indices]

# Split into raw train/validation sets
split = int(0.8 * X.shape[0])
X_train_raw, X_valid_raw = X, X_valid
t_train, t_valid = t, t_valid

# Standardize using training stats
mean = X_train_raw.mean(axis=0)
std = X_train_raw.std(axis=0) + 1e-8
X_train = (X_train_raw - mean) / std
X_valid = (X_valid_raw - mean) / std

# Convert to one-hot encoding
num_classes = 3
t_train_one_hot = np.eye(num_classes)[t_train]
t_valid_one_hot = np.eye(num_classes)[t_valid]

# Define parameter grid
param_grid = {
    'alpha': [ 0.01, 0.005],
    'l2_lambda': [0.25,0.3,0.35],
    'hidden_sizes': [(64, 32), (128, 64), (32, 16),(64,64),(32,32),(16,16)],
    'batch_size': [32, 64,16],
    'patience': [10]
}

# Generate all combinations
keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

best_accuracy = 0
best_params = {}
history = []

print(f"Testing {len(combinations)} parameter combinations...")

for i, params in enumerate(combinations):
    print(f"\nTesting combination {i+1}/{len(combinations)}")
    print(params)
    
    # Initialize model
    nn = NeuralNetwork(
        input_size=X_train.shape[1],
        hidden_sizes=params['hidden_sizes'],
        output_size=num_classes,
        alpha=params['alpha'],
        l2_lambda=params['l2_lambda']
    )
    
    # Train with current parameters
    nn.train(
        X_train=X_train,
        y_train=t_train_one_hot,
        X_val=X_valid,
        y_val=t_valid_one_hot,
        epochs=300,
        batch_size=params['batch_size'],
        patience=params['patience']
    )
    
    # Evaluate
    valid_preds = nn.predict(X_valid)
    accuracy = np.mean(valid_preds == t_valid)
    history.append((params, accuracy))
    
    # Update best parameters
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params.copy()
        best_params['accuracy'] = accuracy
        print(f"New best accuracy: {best_accuracy:.4f}")

# Sort results by accuracy
history.sort(key=lambda x: x[1], reverse=True)

# Save best parameters
print("\nTop 5 parameter combinations:")
for params, acc in history[:5]:
    print(f"Accuracy: {acc:.4f}")
    print(params)
    print("---")

print("\nBest parameters found:")
print(best_params)

# Save best model configuration
with open('best_params.txt', 'w') as f:
    f.write(str(best_params))

# Optional: Save full history
import pandas as pd
df = pd.DataFrame([{'params': p, 'accuracy': a} for p, a in history])
df.to_csv('param_search_history.csv', index=False)