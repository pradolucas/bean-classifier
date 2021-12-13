import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools 
import random
import os

import sklearn
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

def download_dataset(url, path):
    import requests, zipfile, io
    r = requests.get(url, stream=True)
    print(f'\n%%%%%%Dry Bean Dataset downloaded%%%%%')
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path)
    
def load_dataset(path):
    df = pd.read_excel(path, sheet_name=None)
    print(f'\n%%%%%% Dataset Loaded %%%%%%%')
    return df


def Over_sampling(X, Y, multiply_dataset = 2, modo = "SMOTE"):   
    """
    X: data matrix
    Y: labels
    multiply_dataset: number of new samples
    """

    random.seed(10)
    if(multiply_dataset == 0):
        return X,Y
    unqs_orig, unqs_cont_orig = np.unique(Y, return_counts = True)

    if(modo == "SMOTE"):
        if(np.min(unqs_cont_orig)<=5):
            smt = SMOTE(sampling_strategy='auto', k_neighbors = int(np.min(unqs_cont_orig))-1)
        else:
            smt = SMOTE(sampling_strategy='auto')
    elif(modo == "ADA"):
        ada = ADASYN(sampling_strategy='minority') 
    elif(modo == "SMOTE-TOMEK"):
        smt = SMOTETomek(sampling_strategy='auto') 
    elif(modo == "SMOTE-ENN"):
        smt = SMOTEENN(sampling_strategy='auto')
    else:
        smt = SMOTE(sampling_strategy='auto')
    #print(f'orig: {unqs_orig, unqs_cont_orig}, {modo} ')        
    
    new_data, new_data_labels = [], [] 
    X_smote, Y_smote = smt.fit_resample(X,Y)
    F_all, labels = np.asarray(X_smote), np.asarray(Y_smote).reshape(-1,1) 
    _, unqs_cont_smote = np.unique(Y_smote, return_counts = True)

    # gera n amostras para a primeira classe e posteriormente balaceia as demais a partir disso
    for i in range(0, multiply_dataset):
        X, Y = np.asarray(X_smote), np.asarray(Y_smote).reshape(-1,1)
        samples = random.sample(list(np.where(Y == 0)[0]), 1)
        X, Y = np.delete(X, samples, 0), np.delete(Y, samples, 0)
        X_smote, Y_smote = smt.fit_resample(X,Y)
        new_data.append(X_smote[-1]); new_data_labels.append(Y_smote[-1])  
        
    new_data, new_data_labels = np.array(new_data).reshape(-1,X.shape[1]), np.array(new_data_labels).reshape(-1,1)  #transforma lst em nparray, antes (12,2)->(24,1)
    X, Y = np.append(F_all, new_data, axis = 0), np.append(labels, new_data_labels, axis = 0)  #acrescentando os valores à mtx original 
    X_smote, Y_smote = smt.fit_resample(X,Y)
    #print(f'smote: {np.unique(Y_smote, return_counts = True)}')

    return X_smote, Y_smote

def LDA_reduction(X, y, X_test, n_components):
    lda = LDA(n_components=n_components) #n-dimensional LDA
    F_all_train = lda.fit_transform(X, np.ravel(y))
    F_all_test = lda.transform(X_test)
    return X, X_test

def PCA_reduction(X, y, X_test, n_components):
    pca = PCA(n_components=n_components) #n-dimensional PCA
    F_all_train = pca.fit_transform(X, np.ravel(y))
    F_all_test = pca.transform(X_test)
    return X, X_test

def Plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues,
                        save_file = (False, None)):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] ##np.round
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(np.round(cm,3))

    im = plt.imshow(cm, interpolation='nearest', 
                    cmap=cmap, vmin = 0, vmax = np.max(np.sum(cm, axis = 1)))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j],3),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.title(title)
    plt.ylabel('Classes Reais')
    plt.xlabel('Classes Previstas')
    plt.colorbar(im)
    if(save_file[0]):
        if(not os.path.exists(f'{os.getcwd()}/../images')):
            os.mkdir(f'../images')
            print (f'Successfully created the directory Saved_subjects/')
        plt.savefig(f"../images/mtx_conf_{save_file[1]}.png", dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format=None, transparent=False, bbox_inches='tight', pad_inches=0.1)
        plt.close()