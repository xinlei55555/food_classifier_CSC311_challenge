# ------------------- Original Code with sklearn ----------------
from models.naive_bayes import train_naive_bayes, make_inference
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

data_path = 'data/cleaned_data_combined_modified.csv'
model = train_naive_bayes(data_path)
# accuracy 0.86

# Create a CountVectorizer instance
vectorizer = CountVectorizer()

# Fit the vectorizer on the training data
data = pd.read_csv(data_path, delimiter=',', quotechar='"')
X = data.iloc[:, :-1]
X = X.fillna('')
vectorizer.fit(X.apply(lambda row: ' '.join(row.values.astype(str)), axis=1))

inference = make_inference(
    model, vectorizer, 'I love eating this product on Saturday evening with 3-4 ingredients, and I have little hot sauce')
print(f'Inference: {inference}')
# ------------------- Original Code with sklearn ----------------
