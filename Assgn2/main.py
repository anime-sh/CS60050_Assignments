from tqdm import tqdm
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from scipy import sparse as sp
import numpy as np
df=pd.read_csv("train.csv")
vec=CountVectorizer(stop_words='english',binary=True)  # IMPLEMENT THIS KHUDSE
M=vec.fit_transform(df['text'].to_numpy())
M=M.toarray()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df.author)
print(le.classes_)
df.author=le.transform(df.author)
print(M.shape)
print(df.author.shape)

X_train, X_test, y_train, y_test = train_test_split(M, df.author.to_numpy(),stratify=df.author.to_numpy(), test_size=0.99, random_state=42) # stratify ???
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


def prior_calc():
    prior_prob=[]
    for clas in range(len(le.classes_)):
        prior_prob.append(len(df[df.author==clas])/len(df))
    
    return prior_prob

def likelihood_calc(X_n, col_num, val, label):
    X_n=X_n[X_n[:,-1]==label]
    p_x_conditioned_y = len(X_n[X_n[:,col_num]==val]) / len(X_n)
    return p_x_conditioned_y

def naive_bayes(X_train,y_train, X_test,alpha=1):
    y_train = y_train.reshape(-1, 1)
    X_n = np.hstack((X_train, y_train))
    print(X_n.shape)
    print(X_test.shape)
    p_y = prior_calc()
    prior=[np.log(lo+alpha) for lo in p_y]
    Y_pred = []
    lookup_table_likelihood=np.zeros((X_test.shape[1],len(le.classes_),2))

    for i in tqdm(range(X_train.shape[1])):
        for j in range(len(le.classes_)):
            lookup_table_likelihood[i][j][1] = np.log(likelihood_calc(X_n, i,1,j)+alpha)
            lookup_table_likelihood[i][j][0] = np.log(likelihood_calc(X_n, i,0,j)+alpha)
    
    for x in tqdm(X_test):
        likelihood = [0]*len(le.classes_)
        post_prob = [0]*len(le.classes_)
        for j in range(len(le.classes_)):
            for i in range(X_train.shape[1]):
                likelihood[j] += lookup_table_likelihood[i][j][x[i]]
        for j in range(len(le.classes_)):
            post_prob[j] = likelihood[j] + prior[j]
        Y_pred.append(np.argmax(post_prob))
    return np.array(Y_pred)

Y_pred = naive_bayes(X_train,y_train,X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, Y_pred))
print(classification_report(y_test, Y_pred))