from sklearn import preprocessing
from tqdm import tqdm
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from scipy import sparse as sp
import numpy as np
df = pd.read_csv("train.csv")
# IMPLEMENT THIS KHUDSE
vec = CountVectorizer(stop_words='english', binary=True)
M = vec.fit_transform(df['text'].to_numpy())
M = M.toarray()
le = preprocessing.LabelEncoder()
le.fit(df.author)
print(le.classes_)
df.author = le.transform(df.author)
print(M.shape)
print(df.author.shape)

X_train, X_test, y_train, y_test = train_test_split(M, df.author.to_numpy(
), stratify=df.author.to_numpy(), test_size=0.99, random_state=42)  # stratify ???
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


class NaiveBayes(object):
    def __init__(self, alpha=1, n_classes=3):
        """
        Initialize the Naive Bayes model with alpha (correction) and n_classes
        """
        self.alpha = alpha
        self.n_classes = n_classes
        self.X_train = None
        self.Y_train = None
        self.label_total_text_counts = {}
        self.label_total_word_counts = {}
        self.label_word_counts = {}
        for i in range(n_classes):
            self.label_total_text_counts[i] = 0.0
            self.label_total_word_counts[i] = 0.0
            self.label_word_counts[i] = []

    def fit(self, X_train, y_train):
        """
        Fit the model to X_train, y_train
        """
        # Count how many words per label, the frequency of the word for a label
        self.Y_train = y_train
        self.X_train = X_train
        i = 0

        for j in range(self.n_classes):
            self.label_word_counts[j] = np.zeros(self.X_train.shape[1])

        for x in self.X_train:
            self.label_total_text_counts[y_train[i]] += 1
            # self.label_word_counts[y_train[i]] = np.sum([self.label_word_counts[y_train[i]],x])
            self.label_total_word_counts[y_train[i]] += x
            self.label_total_word_counts[y_train[i]] += np.sum(x)
            i += 1

    def p_doc(self, x, y):
        s = 0
        # Calculate conditional probability P(word+alpha|label+vocab*alpha) (with smoothening)
        # Multiplying frequency here
        for index in range(len(x)):
            if x[index]:
                s += ((self.label_word_counts[y][index]+self.alpha))


        # mask=x 
        # np.sum(self.label_word_counts[y]*mask)+np.sum(mask)*self.alpha  can replace for loop with this
        s/=(self.label_total_word_counts[y]+self.X_train.shape[1]*self.alpha)
        return s

    def prior(self, y):
        # Calculate probability of the label
        # total = 0
        # for l in self.label_total_text_counts:
        #     total+=self.label_total_text_counts[l]
        # return self.label_total_text_counts[y]/total
        return self.label_total_text_counts[y]/self.X_train.shape[0]

    def predict(self, X_test):
        """
        Predict the X_test with the fitted model
        """
        pred = []
        probs = []
        priors = []
        for i in range(self.n_classes):
            priors.append(self.prior(i))
        
        for x in tqdm(X_test):
            denom = 0
            local_preds = []
            for j in range(self.n_classes):
                lolol = self.p_doc(x, j)
                numerator = np.log(priors[j])+np.log(lolol)
                denom += priors[j]*lolol
                local_preds.append(numerator)
            denom = np.log(denom)
            local_preds = np.array(local_preds)-denom  # broadcasting
            pred.append(np.argmax(local_preds))
            probs.append(np.exp(local_preds))
            
            
        return pred, probs


nb=NaiveBayes()
from sklearn.metrics import confusion_matrix, classification_report

nb.fit(X_train,y_train)
Y_pred,y_probs=nb.predict(X_test)
print(confusion_matrix(y_test,Y_pred))
p