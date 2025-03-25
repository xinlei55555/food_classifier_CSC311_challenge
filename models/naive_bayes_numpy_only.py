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


# Naive Bayes training function with adjustable MAP estimation
def train_naive_bayes(X_train, y_train, vocab, a=2, b=2):
    # Compute class priors using adjustable MAP estimate (Beta prior with a, b)
    class_counts = defaultdict(int)
    for label in y_train: # TODO WAIT WAIT WAIT -- this seems to be wrong.
        class_counts[label] += 1
    total_samples = len(y_train)
    class_priors = {label: (class_counts[label] + a - 1) / (
        total_samples + (a + b - 2)) for label in class_counts}

    # Compute likelihoods with adjustable Beta(a,b) prior
    class_probs = defaultdict(lambda: np.zeros(len(vocab)))
    for x, label in zip(X_train, y_train):
        class_probs[label] += x

    for label in class_probs:
        class_probs[label] = (class_probs[label] + a - 1) / \
            (class_counts[label] + a + b - 2)

    return class_priors, class_probs

# Naive Bayes inference function


def make_inference(class_priors, class_probs, vocab, text, verbose=False):
    text_vector = transform_vectorizer([text], vocab)
    log_probs = {}
    print('text_vector', (text_vector.shape))
    for label in class_priors:
        log_prob = np.log(class_priors[label])  # Use MAP prior
        log_prob += (np.sum(np.log(class_probs[label]) * text_vector) + (
            np.sum(np.log(1 - class_probs[label]) * (1 - text_vector))))
        log_probs[label] = log_prob

    if verbose:
        print('log_probs:', log_probs)

    return max(log_probs, key=log_probs.get)

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

# Evaluate accuracy of model


def evaluate_accuracy(class_priors, class_probs, vocab, X_test, y_test):
    correct = 0
    for i in range(len(X_test)):
        prediction = make_inference(
            class_priors, class_probs, vocab, ' '.join(map(str, X_test[i])))
        if prediction == y_test[i]:
            correct += 1
    return correct / len(y_test)
