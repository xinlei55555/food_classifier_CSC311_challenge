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
from models.naive_bayes_numpy_only import train_naive_bayes, make_inference, fit_vectorizer


# Load and train model
data_path = 'data/cleaned_data_combined_modified.csv'
model = train_naive_bayes(data_path)
_, _, _, X_test, y_test = model

# Evaluate accuracy
def evaluate_accuracy(model, X_test, y_test):
    correct = 0
    for i in range(len(X_test)):
        prediction = make_inference(model, None, ' '.join(map(str, X_test[i])))
        if prediction == y_test[i]:
            correct += 1
    return correct / len(y_test)

accuracy = evaluate_accuracy(model, X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')

# Read and preprocess data
with open(data_path, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)  # Skip header
    data = [','.join(row[:-1]) for row in reader]  # Combine all feature columns into text

vectorizer = fit_vectorizer(data)

# Run inference
inference = make_inference(model, vectorizer, 'I love eating this product on Saturday evening with 3-4 ingredients, and I have little hot sauce')
print(f'Inference: {inference}')
# -------------------- New code with Numpy only ---------------
