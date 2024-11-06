import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, jsonify, render_template
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')

# Load the dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# Preprocess the dataset
df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)
df.rename(columns={"v1": "Category", "v2": "Message"}, inplace=True)
df.loc[df["Category"] == "spam", "Category"] = 0
df.loc[df["Category"] == "ham", "Category"] = 1

# Ensure Y contains integer values
Y = df["Category"].astype(int)

# Prepare the feature set
X = df["Message"]
feature_extraction = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
X_features = feature_extraction.fit_transform(X)

# Create and fit the logistic regression model
model = LogisticRegression()
model.fit(X_features, Y)

# Create a Flask app
app = Flask(__name__)

def is_gibberish(text):
    # Detect text with high symbol-to-alphanumeric ratio or lack of common words
    if len(text) == 0:
        return False
    symbol_ratio = len(re.findall(r'\W', text)) / len(text)
    # Flag as spam if more than 70% symbols or less than 2 recognizable words
    if symbol_ratio > 0.7 or len(re.findall(r'\b\w+\b', text)) < 2:
        return True
    return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        message = request.form['message']
        if not message:
                raise ValueError("No message provided")
        # new_data_features = feature_extraction.transform([message])
        # prediction = model.predict(new_data_features)
        
        # result = "Ham Mail" if prediction[0] == 1 else "Spam Mail"
        if is_gibberish(message):
            result = "Spam Mail"
        else:
            new_data_features = feature_extraction.transform([message])
            prediction = model.predict(new_data_features)
            result = "Ham Mail" if prediction[0] == 1 else "Spam Mail"
        return render_template('index.html', prediction=result, message=message)
    except Exception as e:
        return render_template('index.html', error=str(e), message=message)
if __name__ == "__main__":
    app.run(debug=True)
