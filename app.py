from flask import Flask, render_template, request, redirect
from helper import preprocessing, vectorizer, get_prediction

app = Flask(__name__)

data = dict()
reviews = ['Good Product', 'Bad Product', 'I like it']
positive = 2
negative = 1


@app.route("/")
def index():
    data['reviews'] =  reviews
    data['positive'] = positive
    data['negative'] = negative
    return render_template('index.html', data=data)


@app.route("/", methods = ['post'])
def my_post():
    text = request.form['text']
    preprocessed_txt = preprocessing(text)
    vectorized_txt = vectorizer(preprocessed_txt)
    predection =  get_prediction(vectorized_txt)

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