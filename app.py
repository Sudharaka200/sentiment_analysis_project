from flask import Flask, render_template, request, redirect
from helper import preprocessing, vectorizer, get_prediction
from logger import logging

app = Flask(__name__)

logging.info('Flask server started')

data = dict()
reviews = ['Good Product', 'Bad Product', 'I like it']
positive = 2
negative = 1


@app.route("/")
def index():
    data['reviews'] =  reviews
    data['positive'] = positive
    data['negative'] = negative

    logging.info('------Open Home Page-----------')

    return render_template('index.html', data=data)


@app.route("/", methods = ['post'])
def my_post():
    text = request.form['text']
    logging.info(f'Text : {text}')

    preprocessed_txt = preprocessing(text)
    logging.info(f'Text : {preprocessed_txt}')

    vectorized_txt = vectorizer(preprocessed_txt)
    logging.info(f'Text : {vectorized_txt}')

    predection =  get_prediction(vectorized_txt)
    logging.info(f'Text : {predection}')

    if predection == 'negative':
        global negative
        negative += 1
    else:
        global positive
        positive += 1

    reviews.insert(0, text)
    return redirect(request.url)


if __name__ == "__main__":
    app.run()