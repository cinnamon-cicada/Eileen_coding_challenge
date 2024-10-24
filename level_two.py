import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords


# PART ONE
tk = WhitespaceTokenizer()
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
pd.set_option('display.max_columns', 100) 

FILE_PATHS = ['export_harrypotter.csv']
DATAFRAMES = []
for path in FILE_PATHS:
    DATAFRAMES.append(pd.read_csv(path,  sep=','))

analyze_df = pd.DataFrame(columns=DATAFRAMES[0].columns.tolist() + ['month'])

# *0. Clean data
# Helper method to extract keywords
def get_keywords(text):
    if type(text) == str:
        arr = set(tk.tokenize(text))
        arr = [x for x in arr if not x.lower() in stop_words]
        return ' '.join(arr)
    return ''

def get_month(text):
    return text.split('-')[1]

for df in DATAFRAMES:
    # **A. Get keywords (from 'text')
    df['text'] = df['text'].apply(lambda x: get_keywords(x))

    # **B. Month (from 'date')
    df['month'] = df['date'].apply(lambda x: get_month(x))  

    # Append df to analyze_df
    analyze_df = pd.concat([analyze_df, df])

analyze_df = analyze_df[['ratings', 'month', 'text']]
#print(analyze_df.head(5))

# Check for NaN values
print(np.unique(analyze_df['ratings']))
unique, counts = np.unique(analyze_df['month'], return_counts=True)

print("Month Data: ")
for a, b in zip(unique, counts):
    print(a, ": ", b)

# PART TWO