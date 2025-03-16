import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score


def train_naive_bayes(data_path):
    # Load the data
    data = pd.read_csv(data_path, delimiter=',', quotechar='"')

    # Print the names of the columns
    print(data.columns)
    print('[INFO] Data shape', data.shape)

    # Preprocess the labels column of the document.
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    print('[INFO] Sample data', X.head())

    # Ensure the text data is not empty
    X = X.fillna('')

    # Convert the text data to numerical data
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(
        X.apply(lambda row: ' '.join(row.values.astype(str)), axis=1))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42)

    # Create and train the Naive Bayes classifier
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Predict the labels for the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    return model


def make_inference(model, vectorizer, text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return prediction[0]
