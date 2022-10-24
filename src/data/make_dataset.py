# -*- coding: utf-8 -*-

""" Generate clean dataset from raw data """

import os
import logging
import re
import pandas as pd
import emoji
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException


logger = logging.getLogger(__name__)


def clean_data(df_raw, text_col='Comment'):
    """ Given a dataframe and a text column, treats the text column of the dataframe and keeps
        only English text with positive or negative sentiment, but not neutral.
    """
    df_0 = df_raw[df_raw['Sentiment'] == 0]
    # df_1 = df_raw[df_raw['Sentiment'] == 1]
    df_2 = df_raw[df_raw['Sentiment'] == 2]
    length = min(len(df_0), len(df_2))

    df_clean = pd.concat([df_0.sample(n=length),df_2.sample(n=length)])
    df_clean = df_clean.dropna().reset_index(drop=True)
    ids_en = []
    # Clean the text
    for idx, row in df_clean.iterrows():
        text = row[text_col]

        # Remove special characters
        text_clean = text.translate(str.maketrans('', '', '#$%&()*+<=>?@[\\]^_`{|}~'))
        text_clean = text_clean.replace("\"", " ")
        # Remove endlines and tabs
        text_clean = text_clean.replace("\n", " ")
        text_clean = text_clean.replace("\r", " ")
        text_clean = text_clean.replace("\xa0", " ")
        # Reduce multiple spces to one
        text_clean = re.sub(' +', ' ', text_clean)

        try:
            language = detect(text_clean)
        except LangDetectException:
            language = 'error'
        # If the text is in English, then keep the row
        if language == 'en':
            ids_en.append(idx)

        # Translate emojis to text
        text_clean = emoji.demojize(text_clean, delimiters=(" ", " "))

        df_clean.loc[idx,text_col] = text_clean

    df_clean = df_clean[df_clean.index.isin(ids_en)].reset_index(drop=True)

    return df_clean


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info('making final data set from raw data...')

    # Path of the script
    dirname = os.path.dirname(__file__)
    # Path to the raw data
    data_raw = os.path.join(dirname, "../../data/raw/comments.csv")

    df_raw = pd.read_csv(data_raw, index_col=0, sep=',')
    df_clean = clean_data(df_raw, text_col='Comment')

    logger.info('final data set created')

    # Path to the processed data
    data_processed = os.path.join(dirname, "../../data/processed/comments_clean.csv")
    df_clean.to_csv(data_processed)


if __name__ == '__main__':
    LOG_FMT = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    main()
