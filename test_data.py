
""" Script that uses great expectations to test if the clean data is correct. """

import great_expectations as ge
import pandas as pd

df = pd.read_csv("data/processed/comments_clean.csv", index_col=0, sep=',')
df = ge.from_pandas(df)
print(df)
df.expect_column_to_exist("Sentiment")
df.expect_column_to_exist("Comment")
df.expect_column_values_to_not_be_null("Sentiment")
df.expect_column_values_to_not_be_null("Comment")
df.expect_column_distinct_values_to_be_in_set("Sentiment", [0,2])
df.expect_column_values_to_be_of_type("Comment", 'str')


config = df.get_expectation_suite()
print(df.validate(config))
