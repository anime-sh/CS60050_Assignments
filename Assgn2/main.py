import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
df=pd.read_csv("train.csv")
print(df.columns)
vec=CountVectorizer(stop_words='english')  # IMPLEMENT THIS KHUDSE
M=vec.fit_transform(df['text'].to_numpy()).toarray()
print(M.shape)
print(df.author.shape)

X_train, X_test, y_train, y_test = train_test_split(M, df.author, test_size=0.3, random_state=42) # stratify ???
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

