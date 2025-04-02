'''Final inference script for Naive Bayes.'''
import os
import pandas as pd
import numpy as np
import csv
import re
from collections import OrderedDict, defaultdict
import json


def load_data(data_path):
    with open(data_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    return data


def load_model():
    """Load trained model components from files"""
    # Load class priors
    with open(f'class_priors.json', 'r') as f:
        class_priors = json.load(f)

    # Load class probabilities
    class_probs_loaded = np.load(f'class_probs.npz')
    class_probs = {class_name: class_probs_loaded[class_name]
                   for class_name in class_probs_loaded.files}

    # Load vocabulary
    with open(f'vocab.json', 'r') as f:
        vocab = json.load(f)

    print(f"Model loaded from {os.getcwd()}")
    return class_priors, class_probs, vocab


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


def preprocess_data(data):
    def parse_common_drinks(input_simple_file: str = "common_drinks.simple") -> OrderedDict[str, str]:
        """
        Parses the common_drinks.simple file and returns an ordered dict of mappings from synonyms
        to their respective drinks (so reverse the common_drinks.simple relation)
        """
        drink_dict = OrderedDict()
        with open(input_simple_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.split("#", 1)[0].strip() == "":
                    continue
                drink, synonyms_str = line.split("=", 1)
                synonyms = [syn.strip() for syn in synonyms_str.split(",")]
                for synonym in synonyms:
                    drink_dict[synonym] = drink
        return drink_dict

    def process_drink(input_str: str, common_drinks: OrderedDict, default: str = "none") -> str:
        """
        1. Replace all forward slashes with a space.
        2. Remove all special non-alphabetic characters (including digits) except dashes.
        3. Lowercase the string.
        4. Reduce consecutive whitespace to a single space.
        5. Check if the entire normalized string matches any key in the ordered dictionary;
        if found, return that dictionary value.
        6. If no direct match, split on whitespace; for each word, see if there's a match.
        If found, return the dictionary value of that match.
        7. If still no match, compute a simple similarity score between each word in the
        splitted string and every dictionary key; pick the best match. Print a warning
        and return that best match's dictionary value.
        """

        text = input_str.replace("/", " ")
        text = re.sub(r"[^a-zA-Z\- ]+", "", text)
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        # Direct match
        for k in common_drinks:
            if k == text:
                return common_drinks[k]
        # split on word
        words = text.split()
        for w in words:
            for k in common_drinks:
                if k == w:
                    return common_drinks[k]

        # If still no match, compute a basic similarity score
        # Let's define a simple character-based Jaccard similarity:
        #   similarity = |set(a) ∩ set(b)| / |set(a) ∪ set(b)|
        # We'll search for the word/key pair that yields the highest similarity
        def jaccard_similarity(a: str, b: str) -> float:
            set_a = set(a)
            set_b = set(b)
            intersection = set_a.intersection(set_b)
            union = set_a.union(set_b)
            return len(intersection) / len(union) if union else 0.0

        best_score = -1.0
        best_key = None
        # For each word in the splitted string, check all dictionary keys
        for w in words:
            for k in common_drinks:
                score = jaccard_similarity(w, k)
                if score > best_score:
                    best_score = score
                    best_key = k

        # If we found something with best similarity, return that
        if best_key is not None and best_score > 0.5:
            # print(f"WARNING: failed to find good match for '{text}', returning '{best_key}' instead by similarity")
            return common_drinks[best_key]

        # If there's absolutely nothing (e.g., empty dictionary?), return some fallback
        # or just the original text
        # print(f"WARNING: no possible matches at all for '{text}', returning default '{default}'")
        return default

    # drinks cleaning
    column_6 = data[:, 6]
    common_drinks = parse_common_drinks('common_drinks.simple')
    processed_column_6 = np.array(
        [process_drink(x, common_drinks) for x in column_6])
    data[:, 6] = processed_column_6

    # joining each row
    texts = [','.join(row) for row in data]

    # Remove labels from data -- but for us, it's useful in the final test set.
    # texts = [texts[idx].replace(labels[idx], '[LABEL]') for idx in range(len(texts))]
    return texts


def transform_vectorizer(data, vocab):
    def tokenize(text):
        '''Find the text and remove the space and punctuation'''
        return re.findall(r'\b\w+\b', text.lower())

    vectors = np.zeros((len(data), len(vocab)), dtype=int)
    for i, row in enumerate(data):
        words = tokenize(row)
        for word in words:
            if word in vocab:
                vectors[i, vocab[word]] += 1
    # Convert to binary features (presence/absence)
    vectors = (vectors > 0).astype(int)
    return vectors


def predict_all(csv_name):
    data = load_data(csv_name)
    data = np.array(data)  # (1644, 10) data.shape

    unwanted_indexes = []  # [0, 1, 2, 4]
    data = np.delete(data, unwanted_indexes, axis=0)

    data = preprocess_data(data)

    class_priors, class_probs, vocab = load_model()

    # Transform the data using the vocabulary
    X_bow_vectors = transform_vectorizer(data, vocab)

    # Make predictions
    results = []
    for i in range(len(X_bow_vectors)):
        prediction, _ = predict_class(
            class_priors, class_probs, X_bow_vectors[i], verbose=False)
        results.append(prediction)
    return results


if __name__ == '__main__':
    # Example usage
    csv_name = os.path.join('test_data', 'data_without_labels.csv')
    predictions = predict_all(csv_name)

    labels_df = pd.read_csv(os.path.join('test_data', "labels.csv"), header=None)

    # Display the loaded labels
    print(labels_df)
    labels = labels_df[0].tolist()
    print(labels[0:10])
    print(labels[-1])
    print(predictions[0:10])
    print(predictions[-1])
    print(len(labels))
    print(len(predictions))
    correct = 0
    total = 0
    for idx in range(len(labels)):
        if labels[idx] == predictions[idx]:
            correct += 1
        total += 1
    print(f"Sample Accuracy for the CSV: {correct / total * 100:.2f}%")
    print('labels', labels[0:10])
    print('labels', predictions[0:10])

    # pass