from flask import Flask, jsonify, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
import cPickle as pickle
import logging
app = Flask(__name__)

@app.route('/student')
def get_student_name():
    return jsonify({'student': 'Fedorenko'})

@app.route('/sentiment', methods=['POST'])
def get_text_score():
    try:
        text = request.json.get('text', '')
        L = pickle.load(open("logreg.pkl", "rb"))
        tfidf = pickle.load(open("tf-idf.pkl", "rb"))
        score = L.predict_proba(tfidf.transform([text]))[0][1]
#        print score
    except Exception as e:
        logging.error(str(e))
        score = 0.0
    return jsonify({'score': score})


if __name__ == '__main__':
    app.run()
