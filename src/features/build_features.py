
""" Module to prepare the input of the model for the prediction from the cleaned data. """

import pandas as pd
import torch


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


def preprocess(data_clean='../../data/processed/comments_clean.csv',
               text_col='Comment', target_col='Sentiment'):
    """ Preprocess clean data for the prdiction. """
    df_clean = pd.read_csv(data_clean, index_col=0, sep=',')
    comments = df_clean[text_col].tolist()  # The tokenizer recieves a list as input
    target = prepare_target(df_clean[target_col])

    return comments, target
