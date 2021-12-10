import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


def get_length_distribution(dataframe):
    """
    Prints out the essay length distribution for the 'essay' portion of the DataFrame
    :param dataframe: Pandas DataFrame
    :return: None
    """
    seq_lengths = np.asarray([len(s.split()) for s in dataframe['essay']])
    print([(p, np.percentile(seq_lengths, p)) for p in [75, 80, 90, 95, 99, 100]])
    print()


def get_vocab_size(dataframe):
    """
    Counts the number of unique words in the 'essay' portion of the DataFrame
    :param dataframe: Pandas DataFrame
    :return: Vocab size
    """
    vocab = {None}
    for text in dataframe['essay'].tolist():
        split = text.split()
        for word in split:
            if word not in vocab:
                vocab.add(word)

    return len(vocab)


df = pd.read_excel('train_data.xlsx')
df_test = pd.read_excel('test_data.xlsx')

df['essay'] = df['essay'].str.lower()
df, val_df = train_test_split(df, test_size=0.1, shuffle=True)

print('Essay length distribution in training data:')
get_length_distribution(df)

vocab_size = get_vocab_size(df)
print(f'Vocab size in training data: {vocab_size}\n')

print('Training DataFrame:')
print(df.describe())
print()

print('Validation DataFrame:')
print(val_df.describe())
print()

print('Test DataFrame:')
print(df_test.describe())
