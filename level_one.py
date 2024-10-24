# https://github.com/VandyDataScience/coding-challenge-FA24/tree/main/Level_1
# Dataset: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/UUB774
# RQ: What predicts a product's average rating? (Variables: Review keywords, time of year)
# Type: Regression
# Cleaning rating: 

# Resources:
# https://medium.com/machine-learning-for-humans/why-machine-learning-matters-6164faf1df12

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords

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

# BELOW: CODE SHELVED FOR FUTURE DEVELOPMENT
# # *1. Visualize data for outliers
# # A. Keywords 
# subset = analyze_df[['text', 'ratings']]
# stars_to_words = {1: [], 2: [], 3: [], 4: [], 5: []}
# def gather_keywords(stars, words):
#     stars_to_words[int(stars)] += words.split(' ')
# subset.apply(lambda x: gather_keywords(x.ratings, x.text), axis=1)
# for i in range(1,6):
#     unique, counts = np.unique(stars_to_words[i], return_counts=True)
#     plt.bar(unique, counts)
#     plt.xlabel('Values')
#     plt.ylabel('Counts')
#     plt.title('Word Counts ' + str(i))
#     plt.show()

