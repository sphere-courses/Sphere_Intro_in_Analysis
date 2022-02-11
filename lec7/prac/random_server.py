import pickle
import logging


from flask import Flask, jsonify, request


app = Flask(__name__)


@app.route('/student')
def get_student_name():
    return jsonify({'student': 'Your Name'})


@app.route('/sentiment', methods=['GET', 'POST'])
def get_text_score():
    if request.method == 'GET':
        text = request.args.get('text')
    else:
        text = request.json.get('text', '')
    try:
        print(text)
        L = pickle.load(open("./artefacts/logreg.pkl", "rb"))
        tfidf = pickle.load(open("./artefacts/tf-idf.pkl", "rb"))
        params = pickle.load(open("./artefacts/params.pkl", "rb"))
        score = L.predict_proba(tfidf.transform([text]))[0][1]
        sentiment = 'positive' if score > params['threshold'] else 'negative'
    except Exception as e:
        logging.error(str(e))
        score, sentiment = 0.0, 'unknown'
    return jsonify({'score': score, 'sentiment': sentiment})


if __name__ == '__main__':
    app.run()
