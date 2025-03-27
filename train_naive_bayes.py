# ------------------- Original Code with sklearn ----------------
# from models.naive_bayes import train_naive_bayes, make_inference
# from sklearn.feature_extraction.text import CountVectorizer
# import pandas as pd

# data_path = 'data/cleaned_data_combined_modified.csv'
# model = train_naive_bayes(data_path)

# # Create a CountVectorizer instance
# vectorizer = CountVectorizer()

# # Fit the vectorizer on the training data
# data = pd.read_csv(data_path, delimiter=',', quotechar='"')
# X = data.iloc[:, :-1]
# X = X.fillna('')
# vectorizer.fit(X.apply(lambda row: ' '.join(row.values.astype(str)), axis=1))
# # accuracy 0.86

# inference = make_inference(
#     model, vectorizer, 'I love eating this product on Saturday evening with 3-4 ingredients, and I have little hot sauce')
# print(f'Inference: {inference}')
# ------------------- Original Code with sklearn ----------------

# -------------------- New code with Numpy only ---------------
import csv
import random
import numpy as np
import json
import os
from models.np_naive_bayes_utils import grid_search, train_naive_bayes, make_inference, load_data, train_val_test_split, evaluate_accuracy

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
    
    print(f"Model saved to {model_dir}")

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
    unwanted_indexes = [0, 1, 2, 4]
    data = np.delete(data, unwanted_indexes, axis=1)

    (X_train, y_train), (X_val, y_val), (X_test, y_test), vocab = train_val_test_split(data, split_file='datasets/train_test_split.csv')
    
    # Grid search for best (a, b) values
    a_range = b_range = range(0, 20, 2)
    a, b = grid_search(X_train, y_train, vocab, X_val, y_val, a_range, b_range)
    print(f'Best (a, b) values: ({a}, {b})')

    class_priors, class_probs = train_naive_bayes(X_train, y_train, vocab, a, b)

    # Save the trained model
    save_model(class_priors, class_probs, vocab)

    # Evaluate
    accuracy = evaluate_accuracy(class_priors, class_probs, vocab, X_val, y_val)
    print(f'Validation Accuracy: {accuracy:.2f}')

    # Example inference
    inference, _ = make_inference(
        class_priors, class_probs, vocab, 
        'I eat this with friends, on weekends, with little hot sauce, and at a party', 
        verbose=True)
    print(f'Inference: {inference}')

if __name__ == '__main__':
    set_random_seed(42)
    main()