import numpy as np
import random
from tqdm import tqdm

def safe_log(x):
    """If x > 0 return log(x) else return a very large negative number"""
    if x > 0:
        return np.log(x)
    else:
        return -10000000

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
        self.vectorized_safe_log = np.vectorize(safe_log)
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

        for j in range(self.n_classes):
            self.label_word_counts[j] = np.zeros(self.X_train.shape[1])

        for i in range(self.X_train.shape[0]):
            self.label_total_text_counts[y_train[i]] += 1
            self.label_word_counts[y_train[i]] += self.X_train[i]
            self.label_total_word_counts[y_train[i]] += np.sum(self.X_train[i])

    def log_p_doc(self, x, y):
        # # Calculate conditional probability P(word+alpha|label+vocab*alpha) (with smoothening)

        # x acts a mask, it is a vector of 0s and 1s, x represents the document
        # broadcasting   num is of size vocab
        num = self.vectorized_safe_log(self.label_word_counts[y]+self.alpha)
        # denom is a scalar
        denom = np.log(
            self.label_total_word_counts[y]+self.X_train.shape[0]*self.alpha)
        num = num-denom  # broadcasting
        log_prob = np.sum(x*num)  # masked sum
        return log_prob

    def prior(self, y):
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
            if np.sum(x) == 0:
                pred.append(random.randint(0, self.n_classes))
                probs.append(0)
                continue
            for j in range(self.n_classes):
                lolol = self.log_p_doc(x, j)
                numerator = np.log(priors[j])+lolol
                denom += priors[j]*np.exp(lolol)
                local_preds.append(numerator)
            denom = self.vectorized_safe_log(denom+self.alpha)
            local_preds = np.array(local_preds)-denom  # broadcasting
            pred.append(np.argmax(local_preds))
            probs.append(np.exp(local_preds))

        return pred, probs

