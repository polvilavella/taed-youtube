import pandas as pd
import torch
import emoji
import re
from sklearn.model_selection import train_test_split


def prepare_target(sentiments):
    """Given the list with the classes for each comment, returns the output with the desired format.

    :return: binary tensor of size 3 representing the class of the comment
    :rtype: torch.tensor
    """
    target = []
    for sent in sentiments:
        if sent == 0:
            target.append([1.0,0.0,0.0])
        elif sent == 1:
            target.append([0.0,1.0,0.0])
        else:
            target.append([0.0,0.0,1.0])
    return torch.tensor(target)


def clean_data(df_raw):
    """TODO: finish documentation"""
    df_clean = df_raw.dropna().reset_index(drop=True)

    for idx, row in df_clean.iterrows():
        text = row['Comment']
        text_clean = emoji.demojize(text, delimiters=(" ", " "))
        text_clean = re.sub(' +', ' ', text_clean)
        df_clean.loc[idx,'Comment'] = text_clean

    return df_clean


def preprocess(data_file='../../data/raw/comments.csv', text_col='Comments', target_col='Sentiment'):
    """TODO: finish documentation"""
    df_raw = pd.read_csv(data_file)
    df_clean = clean_data(df_raw)
    comments = df_clean[text_col].tolist()  # The tokenizer recieves a list as input
    target = prepare_target(df_clean[target_col])

    return comments, target
