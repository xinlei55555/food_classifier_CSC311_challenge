import numpy as np
import csv
import re
from collections import defaultdict
from random import shuffle

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

# Naive Bayes training function with MAP estimation
def train_naive_bayes(data_path):
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
    
    # Compute class probabilities using MAP estimate (Beta prior with a=2, b=2)
    class_probs = defaultdict(lambda: np.zeros(len(vocab)))
    class_counts = defaultdict(int)
    
    for x, label in zip(X_train, y_train):
        class_probs[label] += x
        class_counts[label] += 1
    
    for label in class_probs:
        class_probs[label] = (class_probs[label] + 2) / (class_counts[label] + len(vocab) + 4)
    
    return vocab, class_probs, class_counts, X_test, y_test

# Naive Bayes inference function
def make_inference(model, vectorizer, text):
    vocab, class_probs, class_counts, _, _ = model
    text_vector = transform_vectorizer([text], vocab)
    total_samples = sum(class_counts.values())
    log_probs = {}
    
    for label in class_probs:
        log_prob = np.log((class_counts[label] + 2) / (total_samples + len(class_counts) * 4))  # MAP estimate for prior
        log_prob += np.sum(np.log(class_probs[label]) * text_vector)
        log_probs[label] = log_prob
    
    return max(log_probs, key=log_probs.get)
