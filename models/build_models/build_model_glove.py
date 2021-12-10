import pandas as pd
import numpy as np
import tensorflow as tf
import os
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score


# Hyperparameters for training
max_len = 275
max_words = 57500
LSTM1_units = 200
LSTM2_units = 64
learning_rate = 1e-3
optimizer = optimizers.Adam(learning_rate)

epochs = 12
batch_size = 64

# Load in the training data
train_data_file = '../../data/dataset/train_data.xlsx'
train_df = pd.read_excel(train_data_file)

# ---------- Data Preprocessing ----------
train_df['essay'] = train_df['essay'].str.lower()

# Create a validation set using 10% of training data
train_df, val_df = train_test_split(train_df, test_size=0.1, shuffle=True)

# Create tokenizer using the training data
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_df['essay'])

# Save the tokenizer
with open('../trained_model_files/model_glove/tokenizer_glove_2.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Tokenize the two datasets
train_sequences = tokenizer.texts_to_sequences(train_df['essay'])
val_sequences = tokenizer.texts_to_sequences(val_df['essay'])

word_index = tokenizer.word_index

# Add 0's to the end if the sequence is shorter, truncate if it is longer
X_train = pad_sequences(train_sequences, maxlen=max_len, padding='post')
X_val = pad_sequences(val_sequences, maxlen=max_len, padding='post')

# Separate the labels
y_train = train_df['domain1_score']
y_val = val_df['domain1_score']

# ---------- Load the GloVe embeddings ----------
glove_dir = '../glove'

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6b.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# ---------- Build the model ----------
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len, weights=[embedding_matrix], trainable=True))
model.add(Bidirectional(LSTM(LSTM1_units, recurrent_dropout=0.30, return_sequences=True)))
model.add(Bidirectional(LSTM(LSTM2_units, recurrent_dropout=0.15)))
model.add(Dense(1))

model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
model.summary()

callbacks_list = [
    # Save the model when it achieves the lowest loss on validation
    callbacks.ModelCheckpoint(
        filepath='../trained_model_files/model_glove/model_glove_2.h5',
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )
]

# Train the model
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks_list)

# Load the best model that saved during training and calculate quadratic weighted kappa score
model = load_model('../trained_model_files/model_glove/model_glove_2.h5')
preds = model.predict(X_val)
preds = np.around(preds)

kappa_score = cohen_kappa_score(preds, y_val, weights='quadratic')
print(f"Kappa Score: {kappa_score}\n")
