import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')

def TextAnalysis(df):

    # Read file and tokenize DHC Survey comments
    # df = pd.read_csv("DHCPython.csv")
    df["Words"] = df['Comments'].str.lower().str.split(" ", n=-1, expand=False)

    # Clean up token data (Drop Blanks, etc.)
    df['Words'].replace('', np.nan, inplace=True)
    df['Words'].replace('[^\w\s]','',regex=True)
    df.dropna(subset=['Words'], inplace=True)

    # Pivot Words
    df = df.explode("Words")

    # Remove NLTK stop words
    pat = r'\b(?:{})\b'.format('|'.join(stop))
    df['FWords'] = df["Words"].str.replace(pat, '',regex=True)
    df['FWords'] = df['FWords'].str.replace(r'\s+', ' ',regex=True)

    # Clean up stop-word filtered token data (Drop Blanks)
    df['FWords'].replace('', np.nan, inplace=True)
    df['FWords'].replace('[^\w\s]','',regex=True)
    df.dropna(subset=['FWords'], inplace=True)
    df.drop('Words', inplace=True, axis=1)

    #df.to_csv("DHCtokens.csv")
    return df

def get_output_schema():
    return pd.DataFrame({
        'SurveyId': prep_int(),
        'Score': prep_int(),
        'Comments' : prep_string(),
        'FWords' : prep_string(),
        'ResponseDate' : prep_datetime()
    })


#data types for the get_output_schema:
    # prep_string() --> String
    # prep_decimal() --> Decimal
    # prep_int() --> Integer
    # prep_bool() --> Boolean
    # prep_date() --> Date
    # prep_datetime() --> DateTime
