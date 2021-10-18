from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import preprocessing
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import re
from nltk.corpus import stopwords
from model import NaiveBayes

def read_data():
    df = pd.read_csv("train.csv")
    return df


def get_M_matrix():
    stops = set(stopwords.words('english'))

    vocab = {}
    text_arr = df['text'].to_numpy()
    for i in range(len(text_arr)):
        for word in re.findall("[a-z0-9]+", text_arr[i].casefold()):
            if (word not in stops) and (len(word) > 2):
                vocab[word] = 1

    print(f"Number of distinct words in the dataset = {len(vocab)}")
    M = np.zeros((len(text_arr), len(vocab)))

    idx = 0
    for word in vocab:
        vocab[word] = idx
        idx += 1

    for i in tqdm(range(len(text_arr))):
        for word in re.findall("[a-z0-9]+", text_arr[i].casefold()):
            if (word not in stops) and (len(word) > 2):
                M[i][vocab[word]] = 1
    return M

def train_test_split_data(M, y):
    X_train, X_test, y_train, y_test = train_test_split(M, y, stratify=y, test_size=0.30, random_state=42) 
    print(f" Shape X train {X_train.shape}")
    print(f" Shape y train {y_train.shape}")
    print(f" Shape X test {X_test.shape}")
    print(f" Shape y test {y_test.shape}")
    
    return X_train, X_test, y_train, y_test




def run_experiment(X_train,X_test,y_train,y_test,le,alpha=1):
    model = NaiveBayes(alpha=alpha,n_classes=len(le.classes_))
    model.fit(X_train, y_train)
    y_pred,y_probs = model.predict(X_test)
    print(f"Full Classification report:")
    print(classification_report(y_test, y_pred))
    print(F"Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(F"Accuracy on test split = {accuracy_score(y_test,y_pred)}")


if __name__ == "__main__":
    np.seterr(all="ignore")

    df = read_data()
    M = get_M_matrix()
    le = preprocessing.LabelEncoder()
    le.fit(df.author)
    print(le.classes_)
    df.author = le.transform(df.author)
    print(M.shape)
    print(df.author.shape)
    X_train, X_test, y_train, y_test = train_test_split_data(M, df.author.to_numpy())
    print(F"Experiment with no laplace correction:")
    run_experiment(X_train,X_test,y_train,y_test,le,alpha=0)
    print(F"Experiment with laplace correction, alpha=1:")
    run_experiment(X_train,X_test,y_train,y_test,le,alpha=1)
    print(F"Experiment with laplace correction, alpha=10:")
    run_experiment(X_train,X_test,y_train,y_test,le,alpha=10)

