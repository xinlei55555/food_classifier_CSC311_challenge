import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Load preprocessed data
X = np.load('../data_cleaning/X_clean.npy')
t = np.load('../data_cleaning/t_clean.npy')

# Split data into training and validation sets
X_train, X_valid, t_train, t_valid = train_test_split(X, t, test_size=0.2, random_state=42, stratify=t)

# Normalize features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

# Define hyperparameter grid
param_grid = {
    'hidden_layer_sizes': [(64, 32), (128, 64, 32), (256, 128, 64),(32,16)],  # Different architectures
    'learning_rate_init': [0.0001, 0.001, 0.01,0.005,0.0005],  # Learning rates
    'alpha': [0.0001, 0.001, 0.01,0.005,0.0005],  # L2 regularization
    'batch_size': [16, 32, 64,128],  # Batch sizes
    'activation': ['relu', 'tanh'],  # Activation functions
}

# Initialize the model
model = MLPClassifier(solver='adam', max_iter=500, early_stopping=True, random_state=42)

# Perform Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, t_train)

# Get the best model and hyperparameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print("Best Hyperparameters:", best_params)

# Evaluate on validation set
train_acc = accuracy_score(t_train, best_model.predict(X_train))
valid_acc = accuracy_score(t_valid, best_model.predict(X_valid))

print(f"Best Training Accuracy: {train_acc:.4f}")
print(f"Best Validation Accuracy: {valid_acc:.4f}")

# Save best model and scaler
joblib.dump(best_model, 'best_mlp_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Function to make predictions on new data
def predict_new(X_test):
    """Load best model and make predictions on new test data."""
    model = joblib.load('best_mlp_model.pkl')  # Load trained model
    scaler = joblib.load('scaler.pkl')  # Load scaler

    X_test = scaler.transform(X_test)  # Normalize test data
    predictions = model.predict(X_test)  # Predict class labels

    # Convert numeric labels back to food names
    food_labels = ["Pizza", "Shawarma", "Sushi"]
    return [food_labels[pred] for pred in predictions]

# Example test sample
X_test_sample = X_valid[:5]  # Take first 5 validation samples
predictions = predict_new(X_test_sample)
print("Predictions:", predictions)
