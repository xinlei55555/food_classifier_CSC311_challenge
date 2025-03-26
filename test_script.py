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
from models.naive_bayes_numpy_only import grid_search, train_naive_bayes, make_inference, load_data, train_val_test_split, evaluate_accuracy


def set_random_seed(seed):
    """Sets random seed for training reproducibility

    Args:
        seed (int)"""
    random.seed(seed)
    np.random.seed(seed)

def main():
    # Load and train model
    data_path = 'data/cleaned_data_combined_modified.csv'

    # Read and preprocess data
    data = load_data(data_path)

    # remove certain columns from the data:
    unwanted_indexes = [0, 1, 2, 4] # [index, how complex (1-5), how many ingredients, cost]
    data = np.delete(data, unwanted_indexes, axis=1)

    (X_train, y_train), (X_val, y_val), (X_test, y_test), vocab = train_val_test_split(data, split_file = 'datasets/train_test_split.csv')
    print('data[:5]:', data[:5])
    print('X_train[:5]', X_train[:5])
    
    # Grid search for best (a, b) values
    a_range = b_range = range(0, 20, 2)
    a, b = grid_search(X_train, y_train, vocab, X_val, y_val, a_range, b_range)
    print(f'Best (a, b) values: ({a}, {b})')

    class_priors, class_probs = train_naive_bayes(
        X_train, y_train, vocab, a, b)

    print('len(class_priors), len(class_probs), len(vocab)')
    print(len(class_priors), len(class_probs), len(vocab))
    print('len(class_probs["Pizza"]): ', len(class_probs['Pizza']))

    accuracy = evaluate_accuracy(class_priors, class_probs, vocab, X_val, y_val)
    print(f'Accuracy: {accuracy:.2f}')

    # Read and preprocess data
    with open(data_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        # Combine all feature columns into text
        data = [','.join(row[:-1]) for row in reader]

    # vectorizer = fit_vectorizer(data)

    # Run inference
    inference = make_inference(
        class_priors, class_probs, vocab, 'I eat this with friends, on weekends, with little hot sauce, and at a party', verbose=True)
    print(f'Inference: {inference}')
    # -------------------- New code with Numpy only ---------------


if __name__ == '__main__':
    set_random_seed(42)
    main()
