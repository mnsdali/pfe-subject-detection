from flask import Flask, render_template, request
import re
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from pickle import load

app = Flask(__name__)


def process(description):
    # Assuming new_features is a list of new text data
    new_features = [description]

    tidy_new_features = []
    for i in range(len(new_features)):
        tmp = re.sub(r"[^a-zA-ZÀ-ÿ]", " ", new_features[i])
        tmp = re.sub(r"\s[^a-zA-ZÀ-ÿ]", " ", tmp)
        tmp = re.sub(r"\s[a-zA-ZÀ-ÿ]\s", " ", tmp)
        tmp = re.sub(r"\s+", " ", tmp)
        tmp = tmp.lower()
        tidy_new_features.append(tmp)

    nltk.download("stopwords")
    final_stopwords_list = stopwords.words("english") + stopwords.words("french")

    # Load the same vectorizer used during training
    vectorizer = load(open("models/vectorizer.pkl", "rb"))
    new_X = vectorizer.transform(tidy_new_features).toarray()

    # Load the models with correct filenames
    loaded_gnb = load(open("models/gnb_model.pkl", "rb"))
    loaded_linear_svm = load(open("models/linear_svm.pkl", "rb"))
    loaded_rbf_svm = load(open("models/rbf_svm.pkl", "rb"))
    loaded_sigmoid_svm = load(open("models/sigmoid_svm.pkl", "rb"))
    loaded_poly_svm = load(open("models/poly_svm.pkl", "rb"))
    loaded_neural = load(open("models/neural.pkl", "rb"))

    # Assuming loaded_gnb, loaded_linear_svm, loaded_rbf_svm, etc. are your loaded models
    gnb_prediction = loaded_gnb.predict(new_X)
    linear_svm_prediction = loaded_linear_svm.predict(new_X)
    rbf_svm_prediction = loaded_rbf_svm.predict(new_X)
    sigmoid_svm_prediction = loaded_sigmoid_svm.predict(new_X)
    poly_svm_prediction = loaded_poly_svm.predict(new_X)
    neural_prediction = loaded_neural.predict(new_X)

    return [
        gnb_prediction[0],
        linear_svm_prediction[0],
        rbf_svm_prediction[0],
        sigmoid_svm_prediction[0],
        poly_svm_prediction[0],
        neural_prediction[0],
    ]


# Your NLP model function (replace with your actual implementation)
def predict_subject(subject):
    # Call your NLP model here and return the result
    # For demonstration, let's assume it returns a placeholder result
    return "Prediction for: " + subject


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    subject_description = request.form["subject_description"]

    [
        gnb_prediction,
        linear_svm_prediction,
        rbf_svm_prediction,
        sigmoid_svm_prediction,
        poly_svm_prediction,
        neural_prediction,
    ] = process(subject_description)
    return render_template(
        "index.html",
        gnb_prediction=gnb_prediction,
        linear_svm_prediction=linear_svm_prediction,
        rbf_svm_prediction=rbf_svm_prediction,
        poly_svm_prediction=poly_svm_prediction,
        sigmoid_svm_prediction=sigmoid_svm_prediction,
        neural_prediction=neural_prediction,
    )


if __name__ == "__main__":
    app.run(debug=True)