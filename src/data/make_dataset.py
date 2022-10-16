# -*- coding: utf-8 -*-
import logging
import pandas as pd
import emoji
import re
import string


def clean_data(df_raw, text_col='Comment'):
    """Given a dataframe and a text column, """
    df_clean = df_raw.dropna().reset_index(drop=True)

    for idx, row in df_clean.iterrows():
        text = row[text_col]
        text_clean = emoji.demojize(text, delimiters=(" ", " "))
        text_clean = text_clean.translate(str.maketrans('', '', string.punctuation))
        text_clean = text_clean.replace("\n", " ")
        text_clean = text_clean.replace("\r", " ")
        text_clean = re.sub(' +', ' ', text_clean)
        df_clean.loc[idx,text_col] = text_clean

    return df_clean


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    
    logger.info('making final data set from raw data...')
    
    data_raw = "../../data/raw/comments.csv"
    df_raw = pd.read_csv(data_raw, index_col=0, sep=',')
    
    df_clean = clean_data(df_raw, text_col='Comment')

    logger.info('final data set created')

    data_processed = "../../data/processed/comments_clean.csv"
    df_clean.to_csv(data_processed)
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
