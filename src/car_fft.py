import numpy as np
from scipy.fft import fft


def CAR(X, labels):
    N = X.shape
    N_classes = len(np.unique(labels))
    
    data10 = np.zeros((N[0], N[1], 1))
    data11 = np.zeros((N[0], N[1], 1))
    data12 = np.zeros((N[0], N[1], 1))
    data13 = np.zeros((N[0], N[1], 1))
    
    for trial in range(N[2]): ## Média de cada um os trials de todos canais
        data = X[:,:,trial]
        X_med = np.mean(data, axis = 1).reshape(data.shape[0])
        data_car = data - X_med.reshape((X.shape[0],1))
        
        if (labels[trial] == 0):
            data10 = np.append(data10, data_car.reshape((data_car.shape[0], data_car.shape[1], 1)), axis = 2)#append na terceira dimensão, dos trials
        elif (labels[trial]  == 1):
            data11 = np.append(data11, data_car.reshape((data_car.shape[0], data_car.shape[1], 1)), axis = 2)
        elif (labels[trial]  == 2):
            data12 = np.append(data12, data_car.reshape((data_car.shape[0], data_car.shape[1], 1)), axis = 2)
        elif (labels[trial]  == 3):
            data13 = np.append(data13, data_car.reshape((data_car.shape[0], data_car.shape[1], 1)), axis = 2)
      
    data10 = np.delete(data10, 0, axis=2)
    data11 = np.delete(data11, 0, axis=2)
    data12 = np.delete(data12, 0, axis=2)
    data13 = np.delete(data13, 0, axis=2)
    
    return data10,data11,data12,data13
    
def Ext_fft (N, fs, data10, data11, data12, data13, out_chans):
    """
    args: 
        N ->  x.shape
        fs -> sampling frequency
        dataX -> matrix (1536,16, 12)
        out_chan -> canais a serem excluidos
    """
    
    N_class = 4; N_trials = 12; n_harmonicas = 2
    N_pos = ((N[0]/fs)*np.array([np.array([10,11,12,13])*i for i in range(1,n_harmonicas+1)])).ravel().astype(int)
    val_chans =  np.array(range(1,17))
    val_chans = np.delete(val_chans, [np.where(val_chans == c) for c in out_chans]) #Cria array(1:16) dps exclui os valores a partir de out_chan
    N_chans = val_chans.shape[0]
    n_features = N_pos.shape[0]
    
    F_dez=np.zeros((N_trials,N_chans*N_class*n_harmonicas)) #vetor de trials X (canais*classes)
    F_onze=np.zeros((N_trials,N_chans*N_class*n_harmonicas))
    F_doze=np.zeros((N_trials,N_chans*N_class*n_harmonicas))
    F_treze=np.zeros((N_trials,N_chans*N_class*n_harmonicas))
    
    for trial in range(0,N_trials):
        Chans_XY=0
        for chans in val_chans-1:
            a = abs(fft(data10[:,chans,trial])) # roda pela posição de N_pos 10,11,12,13
            b = abs(fft(data11[:,chans,trial]))
            c = abs(fft(data12[:,chans,trial]))
            d = abs(fft(data13[:,chans,trial]))
            
            F_dez[trial,Chans_XY+np.array(range(0,n_features))] = a[N_pos[range(0,n_features)]]; # roda pela posição de N_pos 10,11,12,13
            F_onze[trial,Chans_XY+np.array(range(0,n_features))] = b[N_pos[range(0,n_features)]]; # roda pela posição de N_pos 10,11,12,13
            F_doze[trial,Chans_XY+np.array(range(0,n_features))] = c[N_pos[range(0,n_features)]]; # roda pela posição de N_pos 10,11,12,13
            F_treze[trial,Chans_XY+np.array(range(0,n_features))] = d[N_pos[range(0,n_features)]]; # roda pela posição de N_pos 10,11,12,13
                     
            Chans_XY += n_features
            
    return F_dez, F_onze, F_doze, F_treze

def CAR_FFT(X,labels, fs):
    # FILTRO CAR
    d10, d11, d12, d13 = CAR(X,labels)

    # EXTRAÇÃO FFT
    out_chans = []
    #out_chans = [1, 2, 3, 4, 10, 14, 15,16]
    F_dez, F_onze, F_doze, F_treze = Ext_fft (*(X.shape, fs, d10, d11, d12, d13), out_chans = out_chans)
    F_all = np.vstack([F_dez, F_onze, F_doze, F_treze])
    return F_all