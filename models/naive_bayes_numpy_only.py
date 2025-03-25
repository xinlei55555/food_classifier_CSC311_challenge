import numpy as np
import csv
import re
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

def vocab_train_test_split(data, test_size=0.2):
    # Combine all feature columns into text
    texts = [','.join(row[:-1]) for row in data]
    labels = [row[-1] for row in data]  # Extract labels

    vocab = fit_vectorizer(texts)
    X = transform_vectorizer(texts, vocab)
    y = np.array(labels)

    # Shuffle and split data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int((1-test_size) * len(X))
    X_train, X_test = X[indices[:split]], X[indices[split:]]
    y_train, y_test = y[indices[:split]], y[indices[split:]]
    return X_train, X_test, y_train, y_test, vocab

def train_naive_bayes(X_train, y_train, vocab, a=1, b=1):
    '''Added laplacian smoothing to avoid zero probabilities'''
    classes = np.unique(y_train)
    class_counts = np.array([np.sum(y_train == c) for c in classes])
    total_samples = len(y_train)
    
    # Class priors (with Dirichlet smoothing)
    class_priors = (class_counts + a - 1) / (total_samples + (a + b - 2) * len(classes))
    
    # Likelihoods (with Laplace smoothing)
    class_probs = {}
    epsilon = 1e-10  # Avoid log(0)
    
    for c in classes:
        X_class = X_train[y_train == c]
        theta = (np.sum(X_class, axis=0) + 1) / (len(X_class) + 2)  # Laplace smoothing
        theta = np.clip(theta, epsilon, 1 - epsilon)  # Avoid 0 or 1
        class_probs[c] = theta # probability of each word to appear
    
    return dict(zip(classes, class_priors)), class_probs

def evaluate_accuracy(class_priors, class_probs, vocab, X_test, y_test):
    correct = 0
    for i in range(len(X_test)):
        # Directly use X_test[i] instead of reconstructing text
        log_probs = {}
        for c in class_priors:
            theta = class_probs[c]
            log_prob = np.log(class_priors[c])
            log_prob += np.sum(X_test[i] * np.log(theta) + (1 - X_test[i]) * np.log(1 - theta))
            log_probs[c] = log_prob
        prediction = max(log_probs, key=log_probs.get)
        if prediction == y_test[i]:
            correct += 1
    return correct / len(y_test)

# # Naive Bayes training function with adjustable MAP estimation
# def train_naive_bayes(X_train, y_train, vocab, a=2, b=2):
#     classes = np.unique(y_train)
#     class_counts = np.array([np.sum(y_train == c) for c in classes])
#     total_samples = len(y_train)
    
#     # Compute class priors using MAP estimate
#     class_priors = (class_counts + a - 1) / (total_samples + (a + b - 2) * len(classes))
    
#     # Compute likelihoods for each class
#     class_probs = {}
#     for i, c in enumerate(classes):
#         X_class = X_train[y_train == c]
#         # Vectorized calculation of theta for each class
#         theta = (np.sum(X_class, axis=0) + a - 1) / (len(X_class) + a + b - 2)
#         class_probs[c] = theta
    
#     return dict(zip(classes, class_priors)), class_probs

# Naive Bayes inference function
# def make_inference(class_priors, class_probs, vocab, text, verbose=False):
#     text_vector = transform_vectorizer([text], vocab)[0]  # Get single vector
    
#     max_log_prob = -np.inf
#     best_class = None
    
#     for class_label in class_priors:
#         theta = class_probs[class_label]
#         # Vectorized log probability calculation
#         log_prob = np.log(class_priors[class_label])
#         log_prob += np.sum(text_vector * np.log(theta) + (1 - text_vector) * np.log(1 - theta))
        
#         if verbose:
#             print(f'Class: {class_label}, Log Prob: {log_prob:.2f}')

#         if log_prob > max_log_prob:
#             max_log_prob = log_prob
#             best_class = class_label
            
#     return best_class
def make_inference(class_priors, class_probs, vocab, text, verbose=False):
    # Convert text to binary feature vector
    text_vector = transform_vectorizer([text], vocab)[0]  # Shape: (vocab_size,)
    
    max_log_prob = -np.inf
    best_class = None
    
    for class_label, prior in class_priors.items():
        theta = class_probs[class_label]  # Shape: (vocab_size,)
        
        # Numerical stability: clip probabilities to avoid log(0)
        theta_clipped = np.clip(theta, 1e-10, 1 - 1e-10)
        
        # Vectorized log probability calculation
        log_prob = np.log(prior)
        log_prob += np.sum(
            text_vector * np.log(theta_clipped) + 
            (1 - text_vector) * np.log(1 - theta_clipped)
        )
        
        if verbose:
            print(f'Class: {class_label}, Log Prob: {log_prob:.2f}, Theta_min: {np.min(theta):.3f}, Theta_max: {np.max(theta):.3f}')
        
        if log_prob > max_log_prob:
            max_log_prob = log_prob
            best_class = class_label
            
    return best_class
# # Evaluate accuracy of model
# def evaluate_accuracy(class_priors, class_probs, vocab, X_test, y_test):
#     correct = 0
#     for i in range(len(X_test)):
#         # Reconstruct the original text for prediction (this might need adjustment)
#         prediction = make_inference(
#             class_priors, class_probs, vocab, ' '.join(map(str, X_test[i])))
#         if prediction == y_test[i]:
#             correct += 1
#     return correct / len(y_test)

# Grid search for best (a, b) values
def grid_search(X_train, y_train, vocab, X_test, y_test, a_values, b_values):
    best_accuracy = 0
    best_params = (None, None)

    for a, b in product(a_values, b_values):
        class_priors, class_probs = train_naive_bayes(
            X_train, y_train, vocab, a, b)

        accuracy = evaluate_accuracy(
            class_priors, class_probs, vocab, X_test, y_test)

        print(f'a={a}, b={b}, Accuracy={accuracy:.2f}')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = (a, b)

    print(f'Best (a, b): {best_params} with Accuracy={best_accuracy:.2f}')
    return best_params