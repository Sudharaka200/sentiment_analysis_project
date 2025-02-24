import numpy as np
import pandas as pd
import re
import string
import pickle

from nltk.stem import PorterStemmer
ps = PorterStemmer()

# Load Model
with open('static/model/model.pickle', 'rb') as f:
    model = pickle.load(f)

#Load stopwords
with open('static/model/corpora/stopwords/english', 'r') as file:
    sw = file.read().splitlines()

#Load vocabulary
vocab =  pd.read_csv('static/model/vocabulary.txt', header=None)
tokens = vocab[0].tolist()

def remove_punctuations(text):
    for punctuation in string.punctuation:  # Fixed typo here
        text = text.replace(punctuation, '')
    return text

def preprocessing(text):
    data = pd.DataFrame([text], columns=['tweet'])
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(x.lower() for x in x.split()))
    data["tweet"] = data['tweet'].apply(lambda x: " ".join(re.sub(r'^https?:\/\/.*[\r\n]*', '', word, flags=re.MULTILINE) for word in x.split()))
    data["tweet"] = data['tweet'].apply(remove_punctuations)
    data["tweet"] = data['tweet'].str.replace(r'\d+', '', regex=True)
    data["tweet"] =  data["tweet"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
    data["tweet"] =  data["tweet"].apply(lambda x: " ".join(ps.stem(x) for x in x.split()))
    return data["tweet"]

def vectorizer(ds):
    vectorized_1st = []  # Initialize the list before using it
    for sentence in ds:
        sentence_1st = [0] * len(tokens)
        for i in range(len(tokens)):
            if tokens[i] in sentence.split():
                sentence_1st[i] = 1  # Use 'sentence_1st' instead of 'sentence_list'     
        vectorized_1st.append(sentence_1st)  # Append sentence_1st, not sentence_list
    vectorized_1st_new = np.asarray(vectorized_1st, dtype=np.float32)
    return vectorized_1st_new

def get_prediction(vectorized_text):
    predection = model.predict(vectorized_text)
    if predection == 1:
        return 'negative'
    else:
        return 'positive'