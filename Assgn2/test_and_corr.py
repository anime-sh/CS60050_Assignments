from tqdm import tqdm
from nltk.util import pr
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("train.csv")
vec=CountVectorizer(stop_words='english',binary=True) 
M=vec.fit_transform(df['text'].to_numpy()).toarray()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df.author)
print(le.classes_)
df.author=le.transform(df.author)
print(type(M))
print(df.author.shape)

X_train, X_test, y_train, y_test = train_test_split(M, df.author.to_numpy(),stratify=df.author.to_numpy(), test_size=0.30, random_state=42) # stratify ???
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Base 
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report

# clf = MultinomialNB()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))


feature_count=np.zeros_like(M[0])
for x in M:
    feature_count+=x
print(feature_count)
ind = np.argpartition(feature_count, -20)[-20:]
# print(ind,feature_count[ind])
M_reduced=M[:,ind]
print(M_reduced.shape)
corr=np.corrcoef(M_reduced)
print(corr)