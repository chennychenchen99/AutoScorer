import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import argparse
import tensorflow_text as text

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


def make_prediction_bert(config):
    """
    Prints the prediction score of the string inputted using the BERT model
    :param config: config.text will be the string inputted
    :return: None
    """
    input_text = [config.text]

    # Load the saved model
    model = tf.saved_model.load('./trained_model_files/model_bert_small')

    # Make the prediction and round to make it an integer
    pred = tf.nn.relu(model(tf.constant(input_text)))
    pred = pred[0][0]
    pred = np.around(pred)

    print(pred)


def make_prediction_glove(config):
    """
    Prints the prediction score of the string inputted using the GloVe model
    :param config: config.text will be the string inputted
    :return: None
    """
    input_text = [config.text]

    df = pd.DataFrame(input_text, columns=['essay'])
    df['essay'] = df['essay'].str.lower()

    with open('trained_model_files/model_glove/tokenizer_glove.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # The max length that was set when training the model
    max_len = 275

    sequences = tokenizer.texts_to_sequences(df['essay'])
    padded_seq = pad_sequences(sequences, maxlen=max_len, padding='post')

    model = load_model('trained_model_files/model_glove/model_glove.h5')

    pred = np.around(model.predict(padded_seq))

    print(pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--text', type=str, help='text to make prediction on', required=True)
    parser.add_argument('-b', '--bert', action='store_true')
    parser.add_argument('-g', '--glove', action='store_true')
    config = parser.parse_args()

    if not (config.bert or config.glove):
        parser.error('No model type requested, add -b/--bert or -g/--glove')

    if config.bert:
        make_prediction_bert(config)
    else:
        make_prediction_glove(config)
