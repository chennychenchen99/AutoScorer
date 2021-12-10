import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def scale(col, set_min, set_max, final_min, final_max):
    """
    Normalizes the scores with different scoring metrics and scales them to desired range
    :param col: The score column in the Pandas DataFrame
    :param set_min: The min possible score of the essay set
    :param set_max: The max possible score of the essay set
    :param final_min: The min score of the new range
    :param final_max: The max score of the new range
    :return: New Pandas DataFrame score column with all values scaled appropriately
    """

    normalized = (col - set_min) / (set_max - set_min)
    range = final_max - final_min

    return normalized * range + final_min


data_fp = '../dataset/training_set_rel3.xlsx'

data = pd.read_excel(data_fp)
data = data[['essay_id', 'essay_set', 'essay', 'domain1_score']]  # Keep these columns only

# Drop duplicate essay copies and drop rows containing null values
data.drop_duplicates(subset=['essay'], inplace=True)
data.dropna(inplace=True)

essay_col = data['essay_set']

# The final score for sets 1, 7, and 8 are added up by the two graders
# Divide the score by 2 for these sets
data.loc[essay_col == 1, 'domain1_score'] = data.loc[essay_col == 1, 'domain1_score'] / 2
data.loc[essay_col == 7, 'domain1_score'] = data.loc[essay_col == 7, 'domain1_score'] / 2
data.loc[essay_col == 8, 'domain1_score'] = data.loc[essay_col == 8, 'domain1_score'] / 2

# Scale the scores to [1, 12] and round them
data.loc[essay_col == 1, 'domain1_score'] = np.around(scale(data.loc[essay_col == 1, 'domain1_score'], 1, 6, 1, 12))
data.loc[essay_col == 2, 'domain1_score'] = np.around(scale(data.loc[essay_col == 2, 'domain1_score'], 1, 6, 1, 12))
data.loc[essay_col == 3, 'domain1_score'] = np.around(scale(data.loc[essay_col == 3, 'domain1_score'], 0, 3, 1, 12))
data.loc[essay_col == 4, 'domain1_score'] = np.around(scale(data.loc[essay_col == 4, 'domain1_score'], 0, 3, 1, 12))
data.loc[essay_col == 5, 'domain1_score'] = np.around(scale(data.loc[essay_col == 5, 'domain1_score'], 0, 4, 1, 12))
data.loc[essay_col == 6, 'domain1_score'] = np.around(scale(data.loc[essay_col == 6, 'domain1_score'], 0, 4, 1, 12))
data.loc[essay_col == 7, 'domain1_score'] = np.around(scale(data.loc[essay_col == 7, 'domain1_score'], 0, 12, 1, 12))
data.loc[essay_col == 8, 'domain1_score'] = np.around(scale(data.loc[essay_col == 8, 'domain1_score'], 0, 30, 1, 12))

data['essay'] = data['essay'].str.replace('\'', '')
data['essay'] = data['essay'].str.replace('\"', '')
data['essay'] = data['essay'].str.strip()

# Creating one .xlsx file for training data and another for test data
# 90/10 split on the original dataset
train_data_fp = '../dataset/train_data.xlsx'
test_data_fp = '../dataset/test_data.xlsx'

train_df, test_df = train_test_split(data, test_size=0.1, shuffle=True)

train_df.to_excel(train_data_fp, index=False)
test_df.to_excel(test_data_fp, index=False)
