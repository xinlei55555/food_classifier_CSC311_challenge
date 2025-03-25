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
    return vectors

# Naive Bayes training function with adjustable MAP estimation
def train_naive_bayes(data_path, a=2, b=2):
    with open(data_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        data = [row for row in reader]
    
    texts = [','.join(row[:-1]) for row in data]  # Combine all feature columns into text
    labels = [row[-1] for row in data]  # Extract labels
    
    vocab = fit_vectorizer(texts)
    X = transform_vectorizer(texts, vocab)
    y = np.array(labels)
    
    # Shuffle and split data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(0.8 * len(X))
    X_train, X_test = X[indices[:split]], X[indices[split:]]
    y_train, y_test = y[indices[:split]], y[indices[split:]]
    
    # Compute class priors using adjustable MAP estimate (Beta prior with a, b)
    class_counts = defaultdict(int)
    for label in y_train:
        class_counts[label] += 1
    total_samples = len(y_train)
    class_priors = {label: (class_counts[label] + a) / (total_samples + len(class_counts) * (a + b)) for label in class_counts}
    
    # Compute likelihoods with adjustable Beta(a,b) prior
    class_probs = defaultdict(lambda: np.zeros(len(vocab)))
    for x, label in zip(X_train, y_train):
        class_probs[label] += x
    
    for label in class_probs:
        class_probs[label] = (class_probs[label] + a) / (class_counts[label] + a + b)
    
    return vocab, class_priors, class_probs, X_test, y_test

# Naive Bayes inference function
def make_inference(model, text):
    vocab, class_priors, class_probs, _, _ = model
    text_vector = transform_vectorizer([text], vocab)
    log_probs = {}
    
    for label in class_priors:
        log_prob = np.log(class_priors[label])  # Use MAP prior
        log_prob += np.sum(np.log(class_probs[label]) * text_vector)
        log_probs[label] = log_prob
    
    return max(log_probs, key=log_probs.get)

# Grid search for best (a, b) values
def grid_search(data_path, a_values, b_values):
    best_accuracy = 0
    best_params = (None, None)
    
    for a, b in product(a_values, b_values):
        model = train_naive_bayes(data_path, a, b)
        _, _, _, X_test, y_test = model
        
        correct = 0
        for i in range(len(X_test)):
            prediction = make_inference(model, ' '.join(map(str, X_test[i])))
            if prediction == y_test[i]:
                correct += 1
        
        accuracy = correct / len(y_test)
        print(f'a={a}, b={b}, Accuracy={accuracy:.2f}')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = (a, b)
    
    print(f'Best (a, b): {best_params} with Accuracy={best_accuracy:.2f}')
    return best_params