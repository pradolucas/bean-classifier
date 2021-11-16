from utils import *
from car_fft import *
import argparse

def train(F_all_train, labels_train, layers, dropout):
    
  # Normalize data
  scaler = MinMaxScaler(feature_range=(0,1))  
  Kscaled_train_samples, Ktrain_labels = scaler.fit_transform(F_all_train), labels_train
  
  model = Sequential()
  model.add(Dense(units=layers[0], input_shape=(F_all_train.shape[1],), activation='tanh'))
  for layer in layers[1::]:
      model.add(Dense(units=layer, activation='tanh'))
  if dropout:
    model.add(Dropout(.2))
  model.add(Dense(units=4, activation='softmax'))
  #model.summary()
  
  # Train
  model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  model.fit(
      x=Kscaled_train_samples
      , y=Ktrain_labels
      , validation_split=0
      , batch_size=16
      , epochs=1000
      , verbose=0)

  return model, scaler

def inferencia(F_all_test, labels_test, classifier, scaler):

  Kscaled_test_samples, Ktest_labels = scaler.transform(F_all_test), labels_test
  predictions = classifier(Kscaled_test_samples)
  class_predictions = np.argmax(predictions, axis = 1).reshape(-1,1) 
  cm = confusion_matrix(y_true = Ktest_labels, y_pred = class_predictions)
  val_acc = np.trace(cm)/np.sum(cm)*100
  
  return val_acc, cm

def run(args, sujeito):
  accs_sujeito_test = []
  accs_sujeito_train = []
  all_cm_test = np.zeros((4,4))
  X, fs, labels = load_dataset(sujeito)
  F_all = CAR_FFT(X,labels, fs) # ordenado
  labels = np.sort(labels)
  
  for i in range(0,args.ntest):
    # Split dataset
    F_all_train, labels_train, F_all_test, labels_test = train_test_split_balanceado(F_all, labels, 0.2)

    # Oversampling
    if(args.oversam > 0):
      F_all_train, labels_train = Over_sampling(F_all_train, labels_train, multiply_dataset = args.oversam , modo = "SMOTE")
    
    # Dim reduction
    if(args.dimred == "lda"):
      F_all_train, F_all_test = LDA_reduction(F_all_train, labels_train, F_all_test, 3)
    elif(args.dimred == "pca"):
      F_all_train, F_all_test = PCA_reduction(F_all_train, labels_train, F_all_test, 30)

    # Filter
    if(args.filter[0] == "best"):
      F_all_train, F_all_test = best_atributtes(F_all_train, labels_train, F_all_test, labels_test, int(args.filter[1])); 
    
    # Classification & inference
    ## clf, scaler, val_acc = wrappers(F_all_train, labels_train, F_all_test, labels_test, 5)
    clf, scaler = train(F_all_train, labels_train, args.l, args.dropout); test_acc, cm_test = inferencia(F_all_test, labels_test, clf, scaler); train_acc, _ = inferencia(F_all_train, labels_train, clf, scaler)
    
    all_cm_test+=cm_test
    accs_sujeito_train.append(train_acc)
    accs_sujeito_test.append(test_acc)
    print(f'test acc: {test_acc}')

  mean_train_acc = np.mean(accs_sujeito_train)
  mean_test_acc = np.mean(accs_sujeito_test)
  print("Mean test acc: ", mean_test_acc)
  
  return sujeito, accs_sujeito_train, accs_sujeito_test, all_cm_test

def main(args):
  try:
    os.mkdir(f'saved_subjects')
  except OSError:
    print (f'Creation of the directory Saved_subjects/ failed')
  else:
    print (f'Successfully created the directory Saved_subjects/')
  
  all_acc = {}
  for sujeito in os.listdir(os.getcwd() + '\data')[0::]:     #os.path.dirname(os.getcwd())
    save_dict((args, run(args, sujeito)), f'saved_subjects/MLP_SMOTE_20/MLP_{sujeito.split(".")[0]}_result.pkl')
    #all_acc[sujeito.split(".")[0]] = run(args, sujeito)
    # all_acc.append(run(args, sujeito))
  
  #show_all_acc(all_acc)
  #save_dict(all_acc, 'saved_subjects/result.pkl')
  # cm_plot_labels = ['10Hz','11Hz', '12Hz', '13Hz']
  # Plot_confusion_matrix(cm=all_acc[0][3], classes=cm_plot_labels, title='Matriz de Confusão', normalize=False)

if __name__=='__main__':
  parser = argparse.ArgumentParser(description="MLP")
  parser.add_argument('-ov', '--oversam', type=int, default = 0, help="Oversample train samples")
  parser.add_argument('-dr','--dimred', choices=("lda", "pca"), default = None, help="Dimension reduction")
  parser.add_argument('-f','--filter', nargs= 2, default = [False], help="Choose best/wrappers and the number of features/consecutive drop in accuracy")
  parser.add_argument('-l', type = int, nargs='+', default = [7], help="layers layout")
  parser.add_argument('-d','--dropout', action = 'store_true', help="Enable Dropout in last layer")
  parser.add_argument('-n','--ntest', type=int, default = 2, help="Number of tests")
  args = parser.parse_args()

  main(args)
  