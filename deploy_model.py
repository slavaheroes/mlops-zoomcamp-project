import pandas as pd
from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
import re
import os
from utils import config, load_artifact
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab


app = Flask(__name__, template_folder='./templates')

stemmer = load_artifact("artifacts/PortStemmer.pickle")
vectorizer = load_artifact("artifacts/vectorizer.pickle")
model = load_artifact("artifacts/model.pickle")

def stemming(content, port_stem):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


def make_prediction(data):
    dataset = pd.DataFrame([data])
    dataset = dataset.fillna('')
    dataset['content'] = dataset['author']+' '+ dataset['title']
    dataset['content'] = dataset['content'].apply(stemming, args=(stemmer, ))
    X_data = dataset['content'].values
    X_data = vectorizer.transform(X_data)
    prediction = model.predict(X_data)
    probability = model.predict_proba(X_data)
    response = {"prediction": float(prediction[0]), "probability": float(probability[0][1])}
    return response


def update_monitor(dataframe_path):
    df = pd.read_csv(dataframe_path).drop(columns=['title', 'author', 'text'])
    data_drift_report = Dashboard(tabs = [DataDriftTab])
    data_drift_report.calculate(df[:100], df[100:])
    data_drift_report.save("templates/my_report.html")


def write_prediction(req, res):
    os.makedirs(config['OUTPUT_PATH'], exist_ok=True)
    path = os.path.join(config['OUTPUT_PATH'], "outputs.csv")
    try:
        items = pd.read_csv(path).to_dict('records')
    except:
        items = []

    items.append({
        "title": req['title'],
        "author": req['author'],
        "text": req['text'],
        "pred_label": res['prediction'],
        "pred_prob": res['probability']
    })
    pd.DataFrame(items).to_csv(path, index = None)
    if len(items) > 150:
        update_monitor(path)


@app.route('/', methods=['GET','POST'])
def form():
    if request.method == "GET":
        return render_template("index.html")
    req = request.form
    response = make_prediction(req)
    write_prediction(req, response)
    return response

@app.route('/monitor', methods = ['GET'])
def monitor():
    if os.path.exists("templates/my_report.html"):
        return render_template("my_report.html")
    return "Currently there are not enough data to analyze"


if __name__ == "__main__":
    app.run(host = "0.0.0.0", port=9000, debug=True)