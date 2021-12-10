import tensorflow as tf
import tensorflow_text as text

from flask import Flask, request, render_template, redirect, url_for
from scorer import get_score_glove, get_score_bert
from textblob import TextBlob

app = Flask(__name__)

model = tf.saved_model.load('./models/trained_model_files/model_bert_small')

tf.config.experimental.set_visible_devices([], 'GPU')


@app.route('/')
def home():
    return render_template('webpage.html', score=-1, essay=[], lang=1)


@app.route('/', methods=['POST'])
def send_text():
    if request.method == 'POST':
        essay = request.form['text']

        # find word count
        word_count = len(essay.split())

        # remove new lines for displaying text
        a_text = essay.split('\n')

        # checks for minimum word count
        if word_count < 10:
            grade = -1
        else:
            # format text for language detection
            b_text = TextBlob(essay)

            # check essay language
            if b_text.detect_language() != 'en':
                return render_template('webpage.html', score=-1, essay=a_text, lang=0)

            # Automatically give a 1.0 score for extremely short answers
            if word_count < 15:
                grade = 1.0
            else:
                # grade = int(get_score(essay, tokenizer, model)[0][0]) / 2.0
                grade = int(get_score_bert(essay, model)) / 2.0
                if grade < 1.0:
                    grade = 1.0

        return render_template('webpage.html', score=grade, essay=a_text, lang=1)


if __name__ == '__main__':
    app.run()
