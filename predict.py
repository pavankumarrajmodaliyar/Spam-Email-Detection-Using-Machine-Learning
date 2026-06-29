import pickle

# Load model
model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

while True:

    msg=input("Enter Email : ")

    data=vectorizer.transform([msg])

    result=model.predict(data)

    if result[0]==1:
        print("Spam Email")
    else:
        print("Not Spam")

    choice=input("Continue(y/n): ")

    if choice.lower()!="y":
        break