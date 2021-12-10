import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_text as text

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


tf.config.experimental.set_visible_devices([], 'GPU')


def get_score_glove(input_text, tokenizer, model):
    """
    Gets score from pretrained model using GloVe.
    :param input_text: string
    :param tokenizer: tokenizer model
    :param model: prediction model using GloVe
    :return: predicted score
    """

    df = pd.DataFrame([input_text], columns=['essay'])
    df['essay'] = df['essay'].str.lower()

    # The max length that was set when training the model
    max_len = 275

    sequences = tokenizer.texts_to_sequences(df['essay'])
    padded_seq = pad_sequences(sequences, maxlen=max_len, padding='post')

    pred = np.around(model.predict(padded_seq))

    return pred


def get_score_bert(input_text, model):
    """
    Gets score from pretrained model using BERT.
    :param input_text: string
    :param model: prediction model using BERT
    :return: predicted score
    """

    pred = tf.nn.relu(model(tf.constant([input_text])))
    pred = pred[0][0]
    pred = np.around(pred)

    return pred

