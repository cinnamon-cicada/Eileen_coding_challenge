import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
import string
import pymc as pm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score

# TEXT CLASSIFICATION RESULTS
# (linear regression results after)

# *** PART ONE ***
# Overhead
tk = WhitespaceTokenizer()
nltk.download('stopwords')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))
pd.set_option('display.max_columns', 100) 
vectorizer = TfidfVectorizer()

FILE_PATHS = ['export_harrypotter.csv']
DATAFRAMES = []
for path in FILE_PATHS:
    DATAFRAMES.append(pd.read_csv(path,  sep=','))

analyze_df = pd.DataFrame(columns=DATAFRAMES[0].columns.tolist() + ['month'])

# CLEAN DATA
# Extract keywords for basic visualization purposes
def get_keywords(text):
    if type(text) == str:
        arr = nltk.word_tokenize(text) # get keywords
        arr = [word.lower() for word in arr if word.isalpha()] 
        arr = [x for x in arr if not x.lower() in stop_words] # remove stopwords
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

# *** PART TWO ***
# VISUALIZE DATA
# Ratings distribution
unique, counts = np.unique(analyze_df['ratings'], return_counts=True)
print("Ratings Data: ")
for a, b in zip(unique, counts):
    print(a, ": ", b)

# Result: Highly skewed left
print('\n')

unique, counts = np.unique(analyze_df['month'], return_counts=True)
print("Month Data: ")
for a, b in zip(unique, counts):
    print(a, ": ", b)

# Result: Peak in winter, low in summer/fall
print('\n')

subset = analyze_df[['text', 'ratings']]
stars_to_words = {1: [], 2: [], 3: [], 4: [], 5: []}
def gather_keywords(stars, words):
    stars_to_words[int(stars)] += words.split(' ')
subset.apply(lambda x: gather_keywords(x.ratings, x.text), axis=1)
for i in range(1,6):
    unique, counts = np.unique(stars_to_words[i], return_counts=True)
    tmp = pd.DataFrame({'word': unique, 'count': counts})
    tmp = tmp.sort_values(by='count', ascending=False)
    print("Top 20 for", str(i), "- STAR")
    print(tmp.head(20))

print('\n')

# *** PREDICT RATINGS BY KEYWORD ***
print('*****')
print('PREDICT BY KEYWORD')

# Create relevant variables
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from torch.utils.data import DataLoader

BASE_MODEL = "bert-base-uncased"
LEARNING_RATE = 2e-5
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 20

# Overhead to compute metrics later on
from evaluate import load
metric = load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Create train, validate, and test data
X = analyze_df['text'] 
y = analyze_df['ratings']
train, validate, test = np.split(analyze_df.sample(frac=1, random_state=42), 
                                 [int(.6*len(analyze_df)), int(.8*len(analyze_df))])

# Classes 1-5 correspond to star numbers
id2label = {k:k for k in range(1, 6)}
label2id = {k:k for k in range(1, 6)}

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, id2label=id2label, label2id=label2id)

ds = {"train": train, "validation": validate, "test": test}

# Pre-process text
def preprocess_function(text):
    label = text["ratings"]
    text = tokenizer(text["text"], truncation=True, padding="max_length", max_length=256)
    text["label"] = label
    return text

for split in ds:
    ds[split] = ds[split].apply(preprocess_function)
    ds[split] = ds[split].remove_columns(["text", "ratings"])

# Train model
training_args = TrainingArguments(
    output_dir="models/hp_reviews",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="accuracy",
    load_best_model_at_end=True,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    compute_metrics=compute_metrics
)

trainer.train()

# *** MODEL EVALUATION ***
print('*****')
print('TEST DATA')
trainer.eval_dataset=ds["test"]
trainer.evaluate()




# LINEAR REGRESSION RESULTS

# *** PART ONE ***
tk = WhitespaceTokenizer()
nltk.download('stopwords')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))
pd.set_option('display.max_columns', 100) 
vectorizer = TfidfVectorizer()


FILE_PATHS = ['export_harrypotter.csv']
DATAFRAMES = []
for path in FILE_PATHS:
    DATAFRAMES.append(pd.read_csv(path,  sep=','))

analyze_df = pd.DataFrame(columns=DATAFRAMES[0].columns.tolist() + ['month'])

# CLEAN DATA
# Helper method to extract keywords
def get_keywords(text):
    if type(text) == str:
        arr = nltk.word_tokenize(text) # get keywords
        arr = [word.lower() for word in arr if word.isalpha()] 
        arr = [x for x in arr if not x.lower() in stop_words] # remove stopwords
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

# *** PART TWO ***
# goal: predict star rating distribution using bayesian regression
# DETERMINE PRIORS
# Ratings distribution
unique, counts = np.unique(analyze_df['ratings'], return_counts=True)
print("Ratings Data: ")
for a, b in zip(unique, counts):
    print(a, ": ", b)

# Result: Highly skewed left
print('\n')

unique, counts = np.unique(analyze_df['month'], return_counts=True)
print("Month Data: ")
for a, b in zip(unique, counts):
    print(a, ": ", b)

# Result: Peak in winter, low in summer/fall
print('\n')

subset = analyze_df[['text', 'ratings']]
stars_to_words = {1: [], 2: [], 3: [], 4: [], 5: []}
def gather_keywords(stars, words):
    stars_to_words[int(stars)] += words.split(' ')
subset.apply(lambda x: gather_keywords(x.ratings, x.text), axis=1)
for i in range(1,6):
    unique, counts = np.unique(stars_to_words[i], return_counts=True)
    tmp = pd.DataFrame({'word': unique, 'count': counts})
    tmp = tmp.sort_values(by='count', ascending=False)
    print("Top 20 for", str(i), "- STAR")
    print(tmp.head(20))

print('\n')

# CHECK ASSUMPTIONS

# *** PREDICT RATINGS BY MONTH ***
X = analyze_df['text'] # Extract features using TF-IDF
y = analyze_df['ratings']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# *** PREDICT RATINGS BY KEYWORD ***
print('*****')
print('PREDICT BY KEYWORD')

X = vectorizer.fit_transform(analyze_df['text']) # Extract features using TF-IDF
y = analyze_df['ratings']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the demand on the test set
y_preds = model.predict(X_test)

# *** MODEL EVALUATION ***
print('*****')
print('MODEL ACCURACY')
rmse = root_mean_squared_error(y_test, y_preds)
print('RMSE:', rmse)

r2 = r2_score(y_test,y_preds)
print('R^2:', r2)