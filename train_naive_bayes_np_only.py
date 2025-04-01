# -------------------- New code with Numpy only ---------------
import csv
import random
import numpy as np
import json
import os

from models.np_naive_bayes_utils import grid_search, train_naive_bayes, make_inference, load_data, train_val_test_split, evaluate_accuracy
from clean.drink_cleaning import process_drink, parse_common_drinks

def save_model(class_priors, class_probs, vocab, model_dir='saved_model'):
    """Save trained model components to files"""
    os.makedirs(model_dir, exist_ok=True)
    
    # Save class priors
    with open(f'{model_dir}/class_priors.json', 'w') as f:
        json.dump(class_priors, f)
    
    # Save class probabilities
    np.savez(f'{model_dir}/class_probs.npz', **class_probs)
    
    # Save vocabulary
    with open(f'{model_dir}/vocab.json', 'w') as f:
        json.dump(vocab, f)
    
    print(f"Model saved to {os.path.join(os.getcwd(), model_dir)}")

def load_model(model_dir='saved_model'):
    """Load trained model components from files"""
    # Load class priors
    with open(f'{model_dir}/class_priors.json', 'r') as f:
        class_priors = json.load(f)
    
    # Load class probabilities
    class_probs_loaded = np.load(f'{model_dir}/class_probs.npz')
    class_probs = {class_name: class_probs_loaded[class_name] for class_name in class_probs_loaded.files}
    
    # Load vocabulary
    with open(f'{model_dir}/vocab.json', 'r') as f:
        vocab = json.load(f)
    
    print(f"Model loaded from {model_dir}")
    return class_priors, class_probs, vocab

def set_random_seed(seed):
    """Sets random seed for training reproducibility"""
    random.seed(seed)
    np.random.seed(seed)

def main():
    # Load and train model
    data_path = 'data/cleaned_data_combined_modified.csv'

    # Read and preprocess data
    data = load_data(data_path)

    data = np.array(data)

    column_6 = data[:, 6]
    common_drinks = parse_common_drinks('clean/common_drinks.simple')
    processed_column_6 = np.array([process_drink(x, common_drinks) for x in column_6])
    data[:, 6] = processed_column_6

    unwanted_indexes = []  # [0, 1, 2, 4]
    data = np.delete(data, unwanted_indexes, axis=1)

    print(data[0], 'data[0]')

    (X_train, y_train), (X_val, y_val), (X_test, y_test), vocab = train_val_test_split(data, split_file='datasets/train_test_split.csv')
    
    print(list(X_train[0]), 'X_train[0:5]')
    # exit()
    # Grid search for best (a, b) values
    a_range = b_range = range(0, 10, 1)
    a, b = grid_search(X_train, y_train, vocab, X_val, y_val, a_range, b_range)
    print(f'Best (a, b) values: ({a}, {b})')

    class_priors, class_probs = train_naive_bayes(X_train, y_train, vocab, a, b)

    # Save the trained model
    save_model(class_priors, class_probs, vocab)

    # Evaluate
    accuracy = evaluate_accuracy(class_priors, class_probs, vocab, X_train, y_train)
    print(f'Validation Accuracy: {accuracy:.5f}')

    # Evaluate
    accuracy = evaluate_accuracy(class_priors, class_probs, vocab, X_val, y_val)
    print(f'Validation Accuracy: {accuracy:.5f}')

    # Evaluate final accuracy for test set
    accuracy = evaluate_accuracy(class_priors, class_probs, vocab, X_test, y_test, confusion_matrix=True)
    print(f'Testing Accuracy: {accuracy: .5f}')
    print(X_test[0:5], 'X_test[0:5]')

    # Example inference
    inference, _ = make_inference(
        class_priors, class_probs, vocab, 
        'I eat this with friends, on weekends, with little hot sauce, and at a party', 
        verbose=True)
    print(f'Inference: {inference}')

if __name__ == '__main__':
    set_random_seed(42)
    main()

    