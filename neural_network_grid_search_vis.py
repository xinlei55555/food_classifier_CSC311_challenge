import itertools
import numpy as np
import pandas as pd
from nnBB import NeuralNetwork, augment_with_nb_features
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Create directories for saving results
import os
os.makedirs('grid_search_results', exist_ok=True)
os.makedirs('confusion_matrices', exist_ok=True)

def run_grid_search():
    # Load augmented data
    X_train_augmented, mean, std = augment_with_nb_features('food_train.csv', is_train=True)
    X_val_augmented, _, _ = augment_with_nb_features('food_val.csv', mean=mean, std=std, is_train=False)

    # Load labels
    t_train = np.load('t_clean_train.npy')
    t_valid = np.load('t_clean_val.npy')
    num_classes = 3
    t_train_one_hot = np.eye(num_classes)[t_train]
    t_valid_one_hot = np.eye(num_classes)[t_valid]

    # Define parameter grid
    param_grid = {
        'alpha': [ 0.01, 0.05, 0.005],
        'l2_lambda': [0.25, 0.3],
        'hidden_sizes': [(128, 64), (256, 128), (64, 32)],
        'batch_size': [16, 32],
        'patience': [10],
        'clip_threshold': [4]  # Add weight clipping thresholds
    }

    # Generate all combinations
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_accuracy = 0
    best_params = {}
    history = []

    print(f"Testing {len(combinations)} parameter combinations...")

    history = []
    cm_files = []

    for i, params in enumerate(combinations):
        print(f"\nCombination {i+1}/{len(combinations)}")
        print(params)
        
        # Initialize model
        nn = NeuralNetwork(
            input_size=X_train_augmented.shape[1],
            hidden_sizes=params['hidden_sizes'],
            output_size=num_classes,
            alpha=params['alpha'],
            l2_lambda=params['l2_lambda'],
            clip_threshold=params['clip_threshold']  # Pass clip_threshold to the model
        )
        
        # Train model
        
        nn.train(
            X_train=X_train_augmented,
            y_train=t_train_one_hot,
            X_val=X_val_augmented,
            y_val=t_valid_one_hot,
            epochs=300,
            batch_size=params['batch_size'],
            patience=params['patience']
        )
        
        # Evaluate
        valid_preds = nn.predict(X_val_augmented)
        accuracy = np.mean(valid_preds == t_valid)

        # Calculate and save confusion matrix
        cm = confusion_matrix(t_valid, valid_preds)
        cm_file = f'confusion_matrices/cm_{i}.npy'
        np.save(cm_file, cm)
        
        # Store results
        history.append({
            'params': params,
            'accuracy': accuracy,
            'cm_file': cm_file
        })
        
        # Update best parameters
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params.copy()
            best_params['accuracy'] = best_accuracy
            print(f"New best accuracy: {best_accuracy:.4f}")

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Confusion matrix saved to {cm_file}")
        print(f"Best parameters so far: {best_params}")
        print(f"best_accuracy: {best_accuracy:.4f}")

        # Save full history
        df = pd.DataFrame(history)
        df.to_csv('grid_search_results/full_results_small.csv', index=False)
        
        # Generate visualizations
        plot_hyperparameter_relationships(df)
            
    return df

    

def plot_hyperparameter_relationships(df):
    # Convert params dict to separate columns
    param_df = pd.json_normalize(df['params'])
    param_df['accuracy'] = df['accuracy']
    
    # Create visualizations for each parameter
    plt.figure(figsize=(15, 10))
    
    # Plot for numerical parameters
    numerical_params = ['alpha', 'l2_lambda', 'clip_threshold']
    for i, param in enumerate(numerical_params, 1):
        plt.subplot(2, 3, i)
        sns.regplot(x=param, y='accuracy', data=param_df, order=2)
        plt.title(f'{param} vs Accuracy')
    
    # Plot for categorical parameters
    categorical_params = ['hidden_sizes', 'batch_size']
    for i, param in enumerate(categorical_params, len(numerical_params)+1):
        plt.subplot(2, 3, i)
        
        # Convert tuples to strings for plotting
        param_df[param] = param_df[param].apply(lambda x: str(x) if isinstance(x, tuple) else x)
        
        sns.boxplot(x=param, y='accuracy', data=param_df)
        plt.xticks(rotation=45)
        plt.title(f'{param} vs Accuracy')
    
    plt.tight_layout()
    plt.savefig('grid_search_results/hyperparameter_relationships_small.png')
    plt.close()
    
    # Create pairplot for numerical parameters
    sns.pairplot(param_df, 
                 vars=numerical_params+['accuracy'],
                 kind='reg',
                 plot_kws={'order': 2})
    plt.savefig('grid_search_results/numerical_relationships_small.png')
    plt.close()
# Run the enhanced grid search
results_df = run_grid_search()

# Show top results
print("\nTop 5 configurations:")
print(results_df.sort_values('accuracy', ascending=False).head(5))