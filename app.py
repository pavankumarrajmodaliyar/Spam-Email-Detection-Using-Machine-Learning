import pickle

print("Loading model...")

try:
    # Load model and vectorizer
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    print("====== Spam Email Detection ======")

    while True:
        message = input("\nEnter Email Message: ")

        data = vectorizer.transform([message])

        prediction = model.predict(data)

        if prediction[0] == 1:
            print("Result: Spam Email")
        else:
            print("Result: Not Spam")

        choice = input("\nCheck another email? (y/n): ")

        if choice.lower() != "y":
            print("Thank you!")
            break

except FileNotFoundError:
    print("Error: model.pkl or vectorizer.pkl not found.")
    print("Run train_model.py first.")
except Exception as e:
    print("An error occurred:", e)