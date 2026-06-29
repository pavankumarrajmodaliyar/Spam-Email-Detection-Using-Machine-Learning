import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

print("Loading dataset...")

try:
    # Load dataset
    df = pd.read_csv("spam.csv")

    # Convert labels into numbers
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.2,
        random_state=42
    )

    # Convert text into numerical features
    vectorizer = CountVectorizer()

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Test model
    prediction = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, prediction)

    print("Accuracy:", accuracy)

    # Save model and vectorizer
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("Model Saved Successfully!")

except FileNotFoundError:
    print("Error: spam.csv file not found.")
except Exception as e:
    print("An error occurred:", e)