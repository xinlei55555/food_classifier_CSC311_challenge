import numpy as np
import csv
import re
import os
from collections import defaultdict
from itertools import product

# Tokenization function


def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# Vectorizer functions


def fit_vectorizer(data):
    vocabulary = set()
    for row in data:
        words = tokenize(row)
        vocabulary.update(words)
    return {word: idx for idx, word in enumerate(sorted(vocabulary))}


def transform_vectorizer(data, vocab):
    vectors = np.zeros((len(data), len(vocab)), dtype=int)
    for i, row in enumerate(data):
        words = tokenize(row)
        for word in words:
            if word in vocab:
                vectors[i, vocab[word]] += 1
    # Convert to binary features (presence/absence)
    vectors = (vectors > 0).astype(int)
    return vectors


def load_data(data_path):
    with open(data_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        data = [row for row in reader]
    return data


def save_data_split(indices, train_size, val_size, test_size, filepath='data_split.csv'):
    """Save train/val/test split indices to a CSV file"""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['type', 'index'])  # header
        
        # Write train indices
        train_end = int(len(indices) * train_size)
        writer.writerows([('train', idx) for idx in indices[:train_end]])
        
        # Write validation indices
        val_end = train_end + int(len(indices) * val_size)
        writer.writerows([('val', idx) for idx in indices[train_end:val_end]])
        
        # Write test indices
        writer.writerows([('test', idx) for idx in indices[val_end:]])

def load_data_split(filepath='data_split.csv'):
    """Load train/val/test split indices from CSV file"""
    train_indices = []
    val_indices = []
    test_indices = []
    
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if row[0] == 'train':
                train_indices.append(int(row[1]))
            elif row[0] == 'val':
                val_indices.append(int(row[1]))
            else:
                test_indices.append(int(row[1]))
    
    return train_indices, val_indices, test_indices

def train_val_test_split(data, train_size=0.7, val_size=0.15, test_size=0.15, 
                        split_file='datasets/train_test_split.csv', random_seed=42):
    """
    Split data into train/validation/test sets with reproducible splits
    
    Args:
        data: Input data to split
        train_size: Proportion of data for training
        val_size: Proportion of data for validation
        test_size: Proportion of data for testing
        split_file: Path to save/load split indices
        random_seed: Seed for reproducibility
        
    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), vocab
    """
    # Verify split proportions
    assert np.isclose(train_size + val_size + test_size, 1.0), \
           "Split sizes must sum to 1"
    
    # Prepare text data
    texts = [','.join(row[:-1]) for row in data]
    labels = [row[-1] for row in data]
    vocab = fit_vectorizer(texts)
    X = transform_vectorizer(texts, vocab)
    y = np.array(labels)
    
    # Try to load existing split
    if os.path.exists(split_file):
        train_indices, val_indices, test_indices = load_data_split(split_file)
        print(f"Loaded existing split from {split_file}")
    else:
        # Create new random split
        np.random.seed(random_seed)
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        
        # Calculate split points
        train_end = int(len(X) * train_size)
        val_end = train_end + int(len(X) * val_size)
        
        # Save the split for future use
        save_data_split(indices, train_size, val_size, test_size, split_file)
        print(f"Created new split and saved to {split_file}")
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
    
    # Split the data
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), vocab



def train_naive_bayes(X_train, y_train, vocab, a=1, b=1):
    '''Added laplacian smoothing to avoid zero probabilities'''
    classes = np.unique(y_train)
    class_counts = np.array([np.sum(y_train == c) for c in classes])
    total_samples = len(y_train)

    # Class priors (with Dirichlet smoothing)
    class_priors = (class_counts + a - 1) / \
        (total_samples + (a + b - 2) * len(classes))

    # Likelihoods (with Laplace smoothing)
    class_probs = {}
    epsilon = 1e-10  # Avoid log(0)

    for c in classes:
        X_class = X_train[y_train == c]
        theta = (np.sum(X_class, axis=0) + a - 1) / \
            (len(X_class) + a + b - 2)  # MAP prior 1 and 2 to avoid 0
        theta = np.clip(theta, epsilon, 1 - epsilon)  # Avoid 0 or 1
        class_probs[c] = theta  # probability of each word to appear

    return dict(zip(classes, class_priors)), class_probs


def predict_class(class_priors, class_probs, feature_vector, verbose=False):
    """
    Unified prediction function used by both evaluate_accuracy and make_inference.

    Args:
        class_priors: Dict of class priors {class_label: prior_prob}
        class_probs: Dict of word probabilities {class_label: theta_vector}
        feature_vector: Binary feature vector of shape (vocab_size,)
        verbose: If True, prints debug info

    Returns:
        Predicted class label
    """
    max_log_prob = -np.inf
    best_class = None
    logits = defaultdict(float)

    for class_label, prior in class_priors.items():
        theta = class_probs[class_label]
        theta_clipped = np.clip(theta, 1e-10, 1 - 1e-10)  # Numerical stability

        log_prob = np.log(prior)
        log_prob += np.sum(
            feature_vector * np.log(theta_clipped) +
            (1 - feature_vector) * np.log(1 - theta_clipped)
        )

        if verbose:
            print(f'Class: {class_label}, Log Prob: {log_prob:.2f}')

        if log_prob > max_log_prob:
            max_log_prob = log_prob
            best_class = class_label
        
        logits[class_label] = log_prob

    return best_class, logits


def evaluate_accuracy(class_priors, class_probs, vocab, X_val, y_val, verbose=False):
    """
    Evaluates accuracy using the unified predict_class function.
    """
    correct = 0
    for i in range(len(X_val)):
        prediction, _ = predict_class(
            class_priors,
            class_probs,
            X_val[i],  # Directly use precomputed feature vector
            verbose=verbose
        )
        if prediction == y_val[i]:
            correct += 1
    return correct / len(y_val)


def make_inference(class_priors, class_probs, vocab, text, verbose=False):
    """
    Makes inference on raw text using the unified predict_class function.
    """
    text_vector = transform_vectorizer(
        [text], vocab)[0]  # Convert text to features
    return predict_class(
        class_priors,
        class_probs,
        text_vector,
        verbose=verbose
    )


def grid_search(X_train, y_train, vocab, X_val, y_val, a_values, b_values):
    best_accuracy = 0
    best_params = (None, None)

    for a, b in product(a_values, b_values):
        class_priors, class_probs = train_naive_bayes(
            X_train, y_train, vocab, a, b)

        accuracy = evaluate_accuracy(
            class_priors, class_probs, vocab, X_val, y_val)

        print(f'a={a}, b={b}, Accuracy={accuracy:.2f}')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = (a, b)

    print(f'Best (a, b): {best_params} with Accuracy={best_accuracy:.2f}')
    return best_params
