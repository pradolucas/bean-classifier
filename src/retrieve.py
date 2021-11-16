#%%
from _typeshed import SupportsNoArgReadline
from utils import *
import argparse
# parser = argparse.ArgumentParser(description="MLP")
# parser.add_argument('-ov', '--oversam', type=int, default = 0, help="Oversample train samples")
# parser.add_argument('-dr','--dimred', choices=("lda", "pca"), default = None, help="Dimension reduction")
# parser.add_argument('-f','--filter', choices=("best", "wrappers"), default = None, help="")
# parser.add_argument('-l', type = int, nargs='+', default = [7], help="layers layout")
# parser.add_argument('-d','--dropout', action = 'store_false', help="Enable Dropout in last layer")
# parser.add_argument('-n','--ntest', type=int, default = 100, help="Number of tests")
# args = parser.parse_args()

# path = os.path.join(os.getcwd(),"saved_subjects - backup")
# print(os.listdir(path)[0])
# for file in os.listdir(path)[0::]:
#     subject = load_dict(os.path.join(path, file))
#     subject = (args, subject)
#     save_dict(subject, f'saved_subjects/{file.split(".")[0]}_TESTE.pkl')


#%% 

from utils import *
import argparse

path = os.path.join(os.getcwd(),"..","saved_subjects","MLP")
accs = []
cms = np.zeros((4,4))
round = 2
for file in os.listdir(path)[0::]:
    subject = load_dict(os.path.join(path, file))
    acc = np.trace(subject[1][3])/np.sum(subject[1][3])*100
    print(f'{str(np.round(acc,round)).replace(".",",")}')
    accs.append(acc)
    cms += subject[1][3]

print(f'{np.round(np.mean(accs),round)} ± {np.round(np.std(accs, ddof = 1), round)} \n') 
cm_plot_labels = ['10Hz','11Hz', '12Hz', '13Hz']
Plot_confusion_matrix(cm=cms, classes=cm_plot_labels, title='Matriz de Confusão MLP', normalize=True)

precisions = (np.diagonal(cms)/ np.sum(cms, axis = 0))*100 
recalls = (np.diagonal(cms)/ np.sum(cms, axis = 1))*100
f1_score = ((2*precisions*recalls)/ (precisions+ recalls) )

print("Precisão: ", np.round(precisions, round))#.reshape(-1,1))
print("Recall: ", np.round(recalls, round))#.reshape(-1,1))
print("F1 Score: ",np.round(f1_score, round))#.reshape(-1,1))

# %%
from utils import *
import argparse

path = os.path.join(os.getcwd(),"..","saved_subjects", "MMQ_SMOTE")
accs = []
cms = np.zeros((4,4))
round = 2
print(os.listdir(path))
for file in os.listdir(path)[0::]:
    subject = load_dict(os.path.join(path, file))
    acc = np.trace(subject[1][3])/np.sum(subject[1][3])*100
    print(f'{str(np.round(acc,round)).replace(".",",")}')
    accs.append(acc)
    cms += subject[1][3]

print(f'{np.round(np.mean(accs),round)} ± {np.round(np.std(accs, ddof = 1), round)} \n') 
cm_plot_labels = ['10Hz','11Hz', '12Hz', '13Hz']
#Plot_confusion_matrix(cm=cms, classes=cm_plot_labels, title='Matriz de Confusão MMQ', normalize=True)

precisions = (np.diagonal(cms)/ np.sum(cms, axis = 0))*100 
recalls = (np.diagonal(cms)/ np.sum(cms, axis = 1))*100
f1_score = ((2*precisions*recalls)/ (precisions+ recalls) )

for x,y,z in list(zip(precisions,recalls, f1_score)):
    print(str(np.round(x,round)).replace(".",","),
          str(np.round(y,round)).replace(".",","),
          str(np.round(z,round)).replace(".",","))

print("Precisão: ", np.round(precisions, round))#.reshape(-1,1))
print("Recall: ", np.round(recalls, round))#.reshape(-1,1))
print("F1 Score: ",np.round(f1_score, round))#.reshape(-1,1))

# %%

## RENAME ARCHIVES
from utils import *
import argparse
path = os.path.join(os.getcwd(),"..","saved_subjects - backup","MLP")
print(os.listdir(path))
for file in os.listdir(path)[0::]:
    sujeito = load_dict(os.path.join(path, file))
    save_dict(sujeito, f'{os.path.join(path, "..", file.split(".")[0][0:-6])}.pkl')
    
#%%
from utils import *
import argparse

path = os.path.join(os.getcwd(),"..","saved_subjects", "MMQ_SMOTE_12")

accs = []
cms = np.zeros((4,4))
round = 2
print(os.listdir(path))
for file in os.listdir(path)[0::]:
    subject = load_dict(os.path.join(path, file))
    acc = np.trace(subject[1][3])/np.sum(subject[1][3])*100
    print(f'{str(np.round(acc,round)).replace(".",",")} ±{np.round(np.std(subject[1][2], ddof=1), round)}')
    accs.append(acc)
    cms += subject[1][3]
 

print(f'{np.round(np.mean(accs),round)} ± {np.round(np.std(accs, ddof = 1), round)} \n') 
cm_plot_labels = ['10Hz','11Hz', '12Hz', '13Hz']
#Plot_confusion_matrix(cm=cms, classes=cm_plot_labels, title='Matriz de Confusão MMQ', normalize=True)

precisions = (np.diagonal(cms)/ np.sum(cms, axis = 0))*100 
recalls = (np.diagonal(cms)/ np.sum(cms, axis = 1))*100
f1_score = ((2*precisions*recalls)/ (precisions+ recalls) )

for x,y,z in list(zip(precisions,recalls, f1_score)):
    print(str(np.round(x,round)).replace(".",","),
          str(np.round(y,round)).replace(".",","),
          str(np.round(z,round)).replace(".",","))

print("Precisão: ", np.round(precisions, round))#.reshape(-1,1))
print("Recall: ", np.round(recalls, round))#.reshape(-1,1))
print("F1 Score: ",np.round(f1_score, round))#.reshape(-1,1))


# %%

from utils import *
import csv

path1 = os.path.join(os.getcwd(),"..","saved_subjects", "MMQ")
path2 = os.path.join(os.getcwd(),"..","saved_subjects", "MLP")
path3 = os.path.join(os.getcwd(),"..","saved_subjects", "MMQ_SMOTE_12")
path4 = os.path.join(os.getcwd(),"..","saved_subjects", "MLP_SMOTE_12")

paths = [path1, path2, path3, path4]

for i in range(1,16):
    print(f'{i} \n')
    save_mtx = {}
    for path in paths:
        for file in os.listdir(path)[i-1:i]:
            method = path.split("\\")[-1]
            subject = load_dict(os.path.join(path, file))
            #print(f'{np.mean(subject[1][2])} \n')
            save_mtx[method] = subject[1][2]
    print(f'{save_mtx}\n\n')

    with open(f'S0{i}.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in save_mtx.items():
            writer.writerow(key, value)


       
# %%

from utils import *
import csv

path = os.path.join(os.getcwd(),"..","etc", "data csv")

sujeito = 5
for file in os.listdir(path)[:]:
    print(file)
    with open(f'{os.path.join(path, file)}') as csv_file:
        reader = csv.reader(csv_file)
        mydict = dict(list(filter(None, reader)))

    for key, value in mydict.items():
        a = value.strip('[]').split(",")
        mydict[key] = np.array(a, dtype = "float").T
    
    values = list(mydict.values())
    for i in range(values[0].shape[0]):
        [print(f'{(values[j][i])} ', end ='') for j in range(len(mydict.keys()))]
        print()
    print(f'\n\n\n\n')
# %%
