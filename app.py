from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

app = Flask(__name__)

# Load dataset
data = pd.read_csv("emails.csv")

# Clean data
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

# Features and labels
X = data['text'].astype(str)
y = data['spam'].astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------- Naïve Bayes ---------
count_vectorizer = CountVectorizer()
X_train_nb = count_vectorizer.fit_transform(X_train)
X_test_nb = count_vectorizer.transform(X_test)

nb_model = MultinomialNB()
nb_model.fit(X_train_nb, y_train)

# --------- Logistic Regression ---------
tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_lr = tfidf_vectorizer.fit_transform(X_train)
X_test_lr = tfidf_vectorizer.transform(X_test)

lr_model = LogisticRegression()
lr_model.fit(X_train_lr, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get email text from request
    data = request.get_json()
    email_text = data['email_text']

    # Vectorize the input text
    nb_vec = count_vectorizer.transform([email_text])
    lr_vec = tfidf_vectorizer.transform([email_text])

    # Get probabilities from both models
    nb_p = nb_model.predict_proba(nb_vec)
    lr_p = lr_model.predict_proba(lr_vec)

    # Average the probabilities (Soft Voting)
    avg_p = (nb_p + lr_p) / 2
    prediction = "Spam Email!" if avg_p[0, 1] > 0.5 else "Ham Email!"

    return jsonify(result=prediction)

if __name__ == '__main__':
    app.run(debug=True)
