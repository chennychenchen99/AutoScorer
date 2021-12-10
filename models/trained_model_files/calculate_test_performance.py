import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_text as text
import pickle
import argparse

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.metrics import cohen_kappa_score


# Load the test data and split data from labels
test_data_file = '../../data/dataset/test_data.xlsx'
test_df = pd.read_excel(test_data_file)

y_test = test_df['domain1_score']


def calc_test_performance_glove(test_df, y_test):
    """
    Calculates and prints out the Quadratic Weighted Kappa Score for the model using GloVe
    :param test_df: The test data read into a DataFrame
    :param y_test: All the essay targets
    :return: None
    """
    max_len = 275

    test_df['essay'] = test_df['essay'].str.lower()

    with open('model_glove/tokenizer_glove.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    sequences = tokenizer.texts_to_sequences(test_df['essay'])
    padded_seq = pad_sequences(sequences, maxlen=max_len, padding='post')

    model = load_model('model_glove/model_glove.h5')

    preds = np.around(model.predict(padded_seq))

    kappa_score = cohen_kappa_score(preds, y_test, weights='quadratic')
    print(f"Quadratic Kappa Score on Test Data with GloVe: {kappa_score}\n")


def calc_test_performance_bert(test_df, y_test, small=True):
    """
    Calculates and prints out the Quadratic Weighted Kappa Score for the model using BERT or small BERT
    :param test_df: The test data read into a DataFrame
    :param y_test: All the essay targets
    :param small: A Boolean to calculate kappa score for either model using BERT or small BERT
    :return: None
    """
    if small:
        model = tf.saved_model.load('model_bert_small')
    else:
        model = tf.saved_model.load('model_bert')

    test_prediction_tensors = tf.nn.relu(model(tf.constant(test_df['essay'])))

    preds = []
    for values in test_prediction_tensors:
        preds.append(values.numpy()[0])

    preds = np.asarray(preds)
    preds = np.around(preds)

    kappa_score = cohen_kappa_score(preds, y_test, weights='quadratic')
    print(f"Quadratic Kappa Score on Test Data with BERT: {kappa_score}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--glove', action='store_true')
    parser.add_argument('-b', '--bert', action='store_true')
    parser.add_argument('-s', '--small', action='store_true')
    config = parser.parse_args()

    if not (config.glove or config.bert):
        parser.error('No model type requested for getting test performance, add -b/--bert or -g/--glove')

    if config.glove:
        calc_test_performance_glove(test_df, y_test)

    if config.bert:
        calc_test_performance_bert(test_df, y_test, config.small)
