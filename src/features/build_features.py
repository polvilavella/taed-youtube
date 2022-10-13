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


def preprocess(data_file='../../data/processed/comments.csv', text_col='Comments', target_col='Sentiment'):
    """TODO: finish documentation"""
    df_clean = pd.read_csv(data_file)
    comments = df_clean[text_col].tolist()  # The tokenizer recieves a list as input
    target = prepare_target(df_clean[target_col])

    return comments, target
