from nltk.util import pr
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("train.csv")
vec=CountVectorizer(stop_words='english')  # IMPLEMENT THIS KHUDSE
M=vec.fit_transform(df['text'].to_numpy()).toarray()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df.author)
print(le.classes_)
df.author=le.transform(df.author)
print(M.shape)
print(df.author.shape)

X_train, X_test, y_train, y_test = train_test_split(M, df.author.to_numpy(),stratify=df.author.to_numpy(), test_size=0.3, random_state=42) # stratify ???
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


def prior_calc():
    prior_prob=[]
    for clas in range(len(le.classes_)):
        prior_prob.append(len(df[df.author==clas])/len(df))
    print(prior_prob,np.sum(prior_prob))
    return prior_prob

def likelihood_calc(X_n, col_num, val, label):
    X_n=X_n[X_n[-1]==label]
    p_x_conditioned_y = len(X_n[X_n[col_num]==val]) / len(X_n)
    return p_x_conditioned_y


def naive_bayes(X_train,y_train, X_test):
    y_train = y_train.reshape(-1, 1)
    X_n = np.hstack((X_train, y_train))
    
    prior = prior_calc()
    Y_pred = []
    for x in X_test:
        likelihood = [1]*len(le.classes_)
        for j in range(len(le.classes_)):
            for i in range(X_test.shape[1]):
                likelihood[j] *= likelihood_calc(X_n, i, x[i],j)
        post_prob = [1]*len(le.classes_)
        for j in range(len(le.classes_)):
            post_prob[j] = likelihood[j] * prior[j]
        Y_pred.append(np.argmax(post_prob))
    return np.array(Y_pred)


Y_pred = naive_bayes(X_train,y_train,X_test)
from sklearn.metrics import confusion_matrix, f1_score
print(confusion_matrix(y_test, Y_pred))
print(f1_score(y_test, Y_pred))