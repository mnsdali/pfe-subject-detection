import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from joblib import load


new_features = ["""
Low-code/No-code
Dashboard Platform

Keywords : Angular, SpringBoot,
Python FastAPI
Nbr of opportunities : 3
Profiles: 2 Fullstack, 1 Data
scientist
Project DSP-002
The goal of the platform is to enable non-technical users to
automate data visualization. We will provide the user with a
ready-made canvas and a workspace that he could design as he
wishes.""", """Customer Experience
Platform

Keywords : SpringBoot, Angular,
Flutter/React Native
Nbr of opportunities : 1
Profiles: Fullstack
Project CXP-003
The goal is to provide a User friendly Platform that allows
associations to create donation collection campaigns and
mobilize the world around their solidarity/humanitarian causes,
projects or events using gamification.""","""Automate the Provisioning and
Configuring of Secure Environments

on AWS

Automated the partners onboard. Automate operations as much as
possible and one key part is the environment and database setup.
Explore different cloud automation tools like AWS OpsWorks, Chef
Automate, and others to help build a POC of a fully automated
environment setup. The workflow automation should allow
different inputs and parameters to customize the environment to
the need of the client account manager.

Tools & Frameworks
DevSecOps, Scripting (bash, shell, PowerShell), Git, Jira, Slack,
Linux Servers, Networking, AWS Cloud, Scrum / SAFe
"""]

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

print(gnb_prediction, linear_svm_prediction, rbf_svm_prediction, sigmoid_svm_prediction,
      poly_svm_prediction, neural_prediction)
