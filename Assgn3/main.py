import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import copy
from itertools import product


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


def pca_runs(X_train, X_val, X_test, y_train, y_val, y_test):
    pca = PCA(n_components=2)
    pca.fit(X)
    principalComponents_train = pca.transform(X_train)
    principaldf = pd.DataFrame(data=principalComponents_train, columns=[
        'pc1', 'pc2'])
    principaldf['y'] = y_train
    sns.scatterplot(data=principaldf, x="pc1", y="pc2", hue="y", cmap='virdis')
    plt.title('Principal Component Analysis Scatter Plot')
    plt.show()

    pc_X_Train = pca.transform(X_train)
    pc_X_Val = pca.transform(X_val)
    pc_X_Test = pca.transform(X_test)
    best_model = None
    best_Acc = 0

    '''
       Hyperparameter tuning for SVM
       Used:
            gamma = ['auto','scale'] # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
            C = [0.01,1]        # Regularization parameter. The strength of the regularization is inversely proportional to C
            kernel=['rbf','poly','sigmoid'] # the kernel type to be used in the algorithm
            class_weight = ['balanced',None]       # class_weight[i]*C for SVC. None, C=1, else C is proportional to inverse of class frequencies
    '''
    parameters= {
        'C':[0.01,1],
        'gamma':['auto','scale'],
        'kernel':['rbf','poly','sigmoid'],
        'class_weight':['balanced',None]
    }
    keys,values = parameters.keys(), parameters.values()
    full_hyperparameter_list = [dict(zip(keys,items)) for items in product(*values)]
    best_params=None
    run=0
    acc_array=[]
    for params in full_hyperparameter_list:
        model = SVC(**params)
        model.fit(pc_X_Train, y_train)
        y_val_pred = model.predict(pc_X_Val)
        acc = accuracy_score(y_val, y_val_pred)
        # print(f"Run: {run} \t Model Params: {params}\t Accuracy: {acc}")
        print(f"Run: {run}")
        acc_array.append(acc)
        run+=1
        if acc > best_Acc:
            best_Acc = acc
            best_model = copy.deepcopy(model)
            best_params = copy.deepcopy(params)

    tuning_df=pd.DataFrame(full_hyperparameter_list)
    tuning_df['val_accuracy']=acc_array
    tuning_df.to_csv('pca_results.csv')
    print(tuning_df)
    print(F"Best params on validation set: {best_params}")
    y_test_pred = best_model.predict(pc_X_Test)
    print(f"Classification Report on Test Set for the best model on validation set")
    print(classification_report(y_test, y_test_pred))
    cm = confusion_matrix(y_test, y_test_pred)
    df_cm = pd.DataFrame(cm, range(2), range(2))
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 15}, fmt='g')
    plt.title('PCA: Confusion Matrix (Test Set)')
    plt.show()
    # sns.heatmap(tuning_df.corr(), annot=True, fmt='g')
    # plt.title('PCA: Hyperparameter Correlation Heatmap')
    # plt.show()


def lda_runs(X_train, X_val, X_test, y_train, y_val, y_test):
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    lda_train = lda.transform(X_train)
    lda_train_df = pd.DataFrame(data=lda_train, columns=['lda'])
    lda_train_df['y'] = y_train
    lda_train_df['lol'] = np.zeros_like(y_train)
    sns.scatterplot(data=lda_train_df, x='lda',
                    y='lol', hue='y', cmap='viridis')
    plt.title('Linear Discriminant Analysis Scatter Plot')
    plt.show()

    lda_X_Train = lda.transform(X_train)
    lda_X_Val = lda.transform(X_val)
    lda_X_Test = lda.transform(X_test)
    best_model = None
    best_Acc = 0
    
    '''
       Hyperparameter tuning for SVM
       Used:
            gamma = ['auto','scale'] # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
            C = [0.01,1]        # Regularization parameter. The strength of the regularization is inversely proportional to C
            kernel=['rbf','poly','sigmoid'] # the kernel type to be used in the algorithm
            class_weight = ['balanced',None]       # class_weight[i]*C for SVC. None, C=1, else C is proportional to inverse of class frequencies
    '''
    parameters= {
        'C':[0.01,1],
        'gamma':['auto','scale'],
        'kernel':['rbf','poly','sigmoid'],
        'class_weight':['balanced',None]
    }
    keys,values = parameters.keys(), parameters.values()
    full_hyperparameter_list = [dict(zip(keys,items)) for items in product(*values)]
    best_params=None
    run=0
    acc_array=[]
    for params in full_hyperparameter_list:
        model = SVC(**params)
        model.fit(lda_X_Train, y_train)
        y_val_pred = model.predict(lda_X_Val)
        acc = accuracy_score(y_val, y_val_pred)
        # print(f"Run: {run} \t Model Params: {params}\t Accuracy: {acc}")
        print(f"Run: {run}")
        acc_array.append(acc)
        run+=1
        if acc > best_Acc:
            best_Acc = acc
            best_model = copy.deepcopy(model)
            best_params = copy.deepcopy(params)

    tuning_df=pd.DataFrame(full_hyperparameter_list)
    tuning_df['val_accuracy']=acc_array
    tuning_df.to_csv('lda_results.csv')
    print(tuning_df)
    print(F"Best params on validation set: {best_params}")
    
    y_test_pred = best_model.predict(lda_X_Test)
    print(f"Classification Report on Test Set for the best model on validation set")
    print(classification_report(y_test, y_test_pred))
    cm = confusion_matrix(y_test, y_test_pred)
    df_cm = pd.DataFrame(cm, range(2), range(2))
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 15}, fmt='g')
    plt.title('LDA: Confusion Matrix (Test Set)')
    plt.show()
    # sns.heatmap(tuning_df.corr(), annot=True, fmt='g')
    # plt.title('LDA: Hyperparamter Correlation Matrix')
    # plt.show()


if __name__ == "__main__":
    df = read_data()
    df = df.drop(['date'], axis=1)  # remove date
    print(f"Final Attributes {df.columns}")
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
    pca_runs(X_train, X_val, X_test, y_train, y_val, y_test)
    lda_runs(X_train, X_val, X_test, y_train, y_val, y_test)
