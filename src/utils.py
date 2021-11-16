import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import itertools 
import random
import os
from collections import deque
from six.moves import cPickle as pickle

import sklearn
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

def load_dataset(sujeito):
    data = loadmat(f'{os.getcwd()}\data\{sujeito}') #os.path.dirname(os.getcwd())
    print(f'\n%%%%%% LOADING {sujeito} %%%%%%%')
    return data['stimuliData'], data['fs'], np.ravel(data['label']-1)


def train_test_split_balanceado(F_all, labels, split):
    ''' 
    Pressupoe dataset ordenado e separado
    '''
    if(split > 0):
        classes, nsamples_classes = np.unique(labels, return_counts=True)## (0,12),(1,12),(2,12),(3,12)
        nsamples_tests = [int(i*split) for i in nsamples_classes] ## [0.2*12,0.2*12,0.2*12,0.2*12]

        idx_test = np.array([np.array( random.sample(range(label*nsamples, (label+1)*nsamples), nsamples_test) )
                             for label, nsamples, nsamples_test in list(zip(classes, nsamples_classes, nsamples_tests))]).ravel()
        idx_train = np.array((range(0,F_all.shape[0])))
        idx_train = np.delete(idx_train, [np.where(idx_train == v) for v in idx_test]) 
        #poderia trocar por np.delete(idx_train[idx_test]) 

        F_all_train, labels_train = F_all[idx_train,:], labels[idx_train]
        F_all_test, labels_test = F_all[idx_test,:], labels[idx_test]

        idx = np.array((range(0,F_all_train.shape[0]))) # n precisa?
        #     random.shuffle(idx)
        return F_all_train[idx,:], labels_train[idx], F_all_test, labels_test
    else:
        return F_all, labels, _, _ 

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

def LDA_reduction(X, Y, X_test, n_components):
    lda = LDA(n_components=n_components) #n-dimensional LDA
    F_all_train = lda.fit_transform(X, np.ravel(Y))
    F_all_test = lda.transform(X_test)
    return F_all_train, F_all_test

def PCA_reduction(X, Y, X_test, n_components):
    pca = PCA(n_components=n_components) #n-dimensional PCA
    F_all_train = pca.fit_transform(X, np.ravel(Y))
    F_all_test = pca.transform(X_test)
    return F_all_train, F_all_test

def db_manual(F_all_train, labels_train):
    DB_score_F_all_train = []
    for k in np.arange(F_all_train.shape[1]):
        F_all_train_SingleCollum = F_all_train[:,k]
        
        # Disperções
        s = [np.std(F_all_train_SingleCollum[np.where(labels_train == i)]) 
            for i in np.arange(np.unique(labels_train).shape[0])] 
        centroide = [np.mean(F_all_train_SingleCollum[np.where(labels_train == i)]) 
             for i in np.arange(np.unique(labels_train).shape[0])]
        
        s_std = np.zeros((4,4))
        for pair in itertools.combinations(np.unique(labels_train), r =2):
            s_std[pair[0],pair[1]] = (s[pair[0]]+s[pair[1]])/np.abs(centroide[pair[0]]-centroide[pair[1]]) # (s_i+s_j)/distCentroide
            
        s_std = s_std + s_std.T - np.diag(s_std.diagonal())
        DB = np.mean(np.max(s_std, axis = 0))   
        DB_score_F_all_train.append(DB)
    return np.argsort(DB_score_F_all_train)    

def db_indexs(F_all_train, labels_train):
    ''' 
        Not finished
    '''
    DB_score_F_all_train = []
    for k in np.arange(F_all_train.shape[1]): # passa por todos as features(4) de todos canais(16) 4*16
        DB_score2a2 = []
        F_all_train_SingleCollum = F_all_train[:,k]
        for i in np.arange(np.unique(labels_train).shape[0]):
            for j in np.arange(np.unique(labels_train).shape[0])[-1:i:-1]:
                #print(f'i,j: {i,j}')
                F_all_train2a2 = np.concatenate((F_all_train_SingleCollum[np.where(labels_train == i)],
                                                 F_all_train_SingleCollum[np.where(labels_train == j)]))
                labels_train2a2 = np.concatenate((labels_train[np.where(labels_train == i)],
                                                  labels_train[np.where(labels_train == j)]))
                
                DB_score_atual = davies_bouldin_score(F_all_train2a2.reshape(-1,1), labels_train2a2)
                DB_score2a2.append(DB_score_atual)
                #print(DB_score2a2)
                
        DB_score_F_all_train.append(np.max(DB_score2a2)/len(np.unique(labels_train)))
    #print(DB_score_F_all_train)
    #print(np.argsort(DB_score_F_all_train))
    #print(len(DB_score_F_all_train))
    #F_all_train_DB_ORDER = F_all_train[:, np.argsort(DB_score_F_all_train)]
    return np.argsort(DB_score_F_all_train)

def best_atributtes(F_all_train, labels_train, F_all_test, labels_test, nfirst_features):
    # Filtro
    idx_feat = db_manual(F_all_train, labels_train) 
    F_all_train =  F_all_train[:,idx_feat]
    F_all_test = F_all_test[:,idx_feat]
        
    #Selecao de Features 
    F_all_train = F_all_train[:,0:nfirst_features]
    F_all_test = F_all_test[:,0:nfirst_features]

    return F_all_train, F_all_test

def wrappers(F_all, labels, F_all_test, labels_test, quedas_conseq):
    ''' 
        Not finished
    '''
    #print(f'\n WRAPPERS')
    # Split Train-Validation e DB ordenation
    F_all_train, labels_train, F_all_val, labels_val = train_test_split_balanceado(F_all, labels, 0)
    idx_feat = db_manual(F_all, labels) 

    n=0
    bst_ChanFeature = [idx_feat[0]]
    bst_ChanFeature_acc = 0
    bst_clf_scaler = deque([])
    
    add_ChanFeature_temp = deque([])
    add_ChanFeature_temp_clf_scaler = deque([])
    add_ChanFeature_temp_acc = deque([])

    while(n<quedas_conseq and len(bst_ChanFeature)<=30):

        for ChFeature in range(0, F_all.shape[1]):
            if(ChFeature not in bst_ChanFeature and ChFeature not in add_ChanFeature_temp):
                
                # Separacao Features
                add_ChanFeature_temp.append(ChFeature)
                valid_ChanFeature = np.concatenate((bst_ChanFeature, add_ChanFeature_temp), axis = 0).astype("int8") #Features escolhidas
                #print(f'valid chans: {valid_ChanFeature}')
                
                # Treinamento
                clf, scaler = train(F_all_train[:,valid_ChanFeature], labels_train)
                val_acc = inferencia(F_all_val[:,valid_ChanFeature], labels_val, clf, scaler)

                add_ChanFeature_temp_acc.append((val_acc, ChFeature))
                add_ChanFeature_temp_clf_scaler.append((clf, scaler))
                add_ChanFeature_temp.pop()

        argmax = np.argmax(list(zip(*add_ChanFeature_temp_acc))[0])
        add_ChanFeature_temp.append(add_ChanFeature_temp_acc[argmax][1])   
        #print(f'\n argmax, ch: {add_ChanFeature_temp_acc[argmax][0], add_ChanFeature_temp_acc[argmax][1]} \n')
       
        if(add_ChanFeature_temp_acc[argmax][0] < bst_ChanFeature_acc):
            n+=1
        elif(add_ChanFeature_temp_acc[argmax][0] >= bst_ChanFeature_acc):
            n=0
            bst_ChanFeature = np.concatenate((bst_ChanFeature, add_ChanFeature_temp), axis = 0).astype("int8") #canais escolhidos
            bst_ChanFeature_acc = add_ChanFeature_temp_acc[argmax][0]
            bst_clf_scaler = add_ChanFeature_temp_clf_scaler[argmax]
            add_ChanFeature_temp = []
            #print(f'bst_ChanFeature: {bst_ChanFeature}, acc: {bst_ChanFeature_acc}')
            
            # Split Validation 
            F_all_train, labels_train, F_all_val, labels_val = train_test_split_balanceado(F_all, labels, 0)
        
        #print(f'n = {n}, len(best): {len(bst_ChanFeature)}, len(temp): {len(add_ChanFeature_temp)}')
        #print(f'add_ChanFeature_temp: {add_ChanFeature_temp} \nbst_ChanFeature: {bst_ChanFeature} \n ')
        add_ChanFeature_temp_acc = []
        add_ChanFeature_temp_clf_scaler = []
        
    test_acc, test_cm = inferencia(F_all_test[:,bst_ChanFeature], labels_test, *bst_clf_scaler)
    train_acc, _ = inferencia(F_all[:,bst_ChanFeature], labels, *bst_clf_scaler)
    
    #print(f'Best features: {bst_ChanFeature}')
    #print(f'Best features val acc: {bst_ChanFeature_acc}')  
    #print(f'Train acc: {train_acc}')  
    return bst_clf_scaler[0], bst_clf_scaler[1] , train_acc, test_acc

def Plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(np.round(cm,3))

    im = plt.imshow(cm, interpolation='nearest', 
                    cmap=cmap, vmin = 0, vmax = np.max(np.sum(cm, axis = 1)))
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j],3),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Classes Reais')
    plt.xlabel('Classes Previstas')
    plt.colorbar(im)

    #plt.savefig(f"../imagens/mtx_conf_MLP.png", dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format=None, transparent=False, bbox_inches='tight', pad_inches=0.1)


def show_all_acc(all_acc):
    n_round = 3

    print(f'\nAcc train: ')
    for train_subject in all_acc.keys(): 
            print(f'{train_subject}: {str(np.round(np.mean(all_acc[train_subject][0]), n_round)).replace(".",",")} +/- {str(np.round(np.std(all_acc[train_subject][0], ddof = 1), n_round)).replace(".",",")}')
    
    print(f'\nAcc test: ')
    for test_subject in all_acc.keys(): 
            print(f'{test_subject}: {str(np.round(np.mean(all_acc[test_subject][1]), n_round)).replace(".",",")} +/- {str(np.round(np.std(all_acc[test_subject][1], ddof = 1), n_round)).replace(".",",")}')

def save_dict(di_, filename_):
  print('SAVING DICT')
  with open(filename_, 'wb') as f:
      pickle.dump(di_, f)

def load_dict(filename_):
  #print('LOADING DICT')
  with open(filename_, 'rb') as f:
      ret_di = pickle.load(f)
  return ret_di
