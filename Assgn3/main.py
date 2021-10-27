import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def read_data():
    df_list = []
    df_list.append(pd.read_csv('occupancy_data/datatest.txt'))
    df_list.append(pd.read_csv('occupancy_data/datatraining.txt'))
    df_list.append(pd.read_csv('occupancy_data/datatest2.txt'))
    return pd.concat(df_list)


def train_val_test_split(X, y, train_size, val_size, shuffle=True, seed=42):
    '''
    Splits the x into training, validation and test sets.
    :param X: the x
    :param y: the target values
    :param train_size: the size of the training set
    :param val_size: the size of the validation set
    :param shuffle: whether to shuffle the x
    :param seed: the seed for the random generator
    :return: X_train, X_val, X_test, y_train, y_val, y_test
    '''

    test_size = 1-train_size-val_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, stratify=y, test_size=(1.0 - train_size), random_state=seed)
    relative_test_size = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, stratify=y_temp, test_size=relative_test_size, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


df = read_data()
df = df.drop(['date'], axis=1)  # remove date
print(f"Final Attributed {df.columns}")
X = df.drop(df.columns[-1], axis=1).to_numpy()
y = df[df.columns[-1]].to_numpy()
X = StandardScaler().fit_transform(X)
print(np.unique(y))
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
    X, y, 0.7, 0.1)
print(f"X_train shape {X_train.shape}")
print(f"y_train shape {y_train.shape}")
print(f"X_val shape {X_val.shape}")
print(f"y_val shape {y_val.shape}")
print(f"X_test shape {X_test.shape}")
print(f"y_test shape {y_test.shape}")

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principaldf = pd.DataFrame(data=principalComponents, columns=[
                           'pc1', 'pc2'])
principaldf['y']=y
sns.scatterplot(data=principaldf,x="pc1",y="pc2",hue="y",cmap='virdis')
plt.show()
