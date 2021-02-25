# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 10:26:58 2020
@author: Thiru Nadesan
"""
#############################LOADING DEPENDENCIES##############################
'''tensorflow and keras dependencies'''
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping

'''sklearn dependencies'''
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

'''math dependencies'''
import numpy as np

'''plot dependencies'''
import matplotlib.pyplot as plt

#############################MAIN CODE#########################################

#training data
data = np.genfromtxt('result.txt', delimiter=' ', skip_header=True) 
input_features = 3 #first four columns
np.seterr(divide='ignore', invalid='ignore')
output_data = (data[:,-1]/data[:,-2]).reshape(-1,1)
ind = np.where(np.isinf(output_data))
data = np.delete(data, ind, 0)

'''input-output features'''
input_data = data[:,0:3]
output_data= data[:,-2]/data[:,-1]

'''feature scaling - before choosing standardization or binning or normalization,
it is necessary to look into the distribution of the input features'''

titles=['a', 'b', 'c', 'alpha', 'cl', 'cd']
fig, ax = plt.subplots(3,2)
ax = ax.ravel()

for idx,ax in enumerate(ax):
    ax.hist(input_data[idx])
    ax.set_title(titles[idx])

fig.tight_layout()
fig.savefig('./figures/input_features.png', dpi=600)

#train test split
input_train, input_test, output_train, output_test = train_test_split(input_data,
                                                                      output_data,
                                                                      test_size=0.20,
                                                                      random_state=42,
                                                                      shuffle=True)

np.savetxt("./train_data/input_train_cl_by_cd.txt", np.c_[input_train])
np.savetxt("./train_data/output_train_cl_by_cd.txt", np.c_[output_train])
np.savetxt("./test_data/cl_by_cd_input_test.txt", np.c_[input_test])
np.savetxt("./test_data/cl_by_cd_output_test.txt", np.c_[output_test])

scaler_mm = MinMaxScaler()
scaler_ss = StandardScaler()
input_train = scaler_mm.fit_transform(input_train)
output_train = scaler_mm.fit_transform(output_train.reshape(-1,1))

'''optimizers'''
adam = optimizers.Adam(learning_rate = 1e-03, 
                       beta_1 = 0.9, #moving weighted average of dW and db
                       beta_2 = 0.999, # moving weighted average of dW^2 and db^2
                       amsgrad = False)

adadelta = optimizers.Adadelta(learning_rate=0.1,
                               rho=0.5,
                               epsilon = 1e-08)

'''hyperparameters for tuning the model'''
batch_size=2**6
hid_layer_neurons = 3
num_hidden_layers = 10 #excludes input and output layer
num_epochs=3000
activation='relu'

count=0
def get_loss(input_train, output_train, input_val, output_val):
    global count
    global batch_size
    global hid_layer_neurons
    global num_hidden_layers
    global num_epochs
    global adadelta
    global activation
    '''create keras neural network model with hidden layers. Rectified linear unit 
    activation function is used so that the training is faster'''

    model = Sequential()

    for i in range(num_hidden_layers+1):
        if i ==0:
            model.add(Dense(hid_layer_neurons, activation=activation, input_dim=input_features))
        elif i!=0 and i!=num_hidden_layers:
            model.add(Dense(hid_layer_neurons, activation=activation))
        else:
            model.add(Dense(1))
    
    model.summary()

    model.compile(loss='mae', 
                  optimizer=adadelta, 
                  metrics=['mae'])

    monitor=EarlyStopping(monitor='val_loss', min_delta=1e-04, patience=100, verbose=1,
                     restore_best_weights=True)
    history = model.fit(input_train,
                        output_train, 
                        epochs=num_epochs,
                        batch_size=batch_size,
                        validation_data =(input_val, output_val),
                        callbacks=[monitor]
                        )
    #plot training history 
    fig2=plt.figure(figsize=(8, 8))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.xlabel('Epochs')
    plt.ylabel('Mean absolute Error')
    plt.legend()
    fig2.savefig("./figures/initial_training_loss_cl_by_cd_hln10_nhl6_noalpha"+str(batch_size)+"_adadelta"+"_earlystopping_kfold"+str(count)+".png", 
                 bbox_inches='tight', dpi = 600)
    plt.show()
    
    model.save("./saved_model/xfoil_ann_cl_by_cd_hln10_nhl6_kfold_noalpha"+str(count)+".h5")
    count+=1
    return history.history['val_loss'][-1]

n_splits = 3 #10 folds
kf = KFold(n_splits=n_splits)
splits = kf.split(input_train)
loss=[]
train_count=1

for train_index, val_index in splits:
    train_input, val_input, train_output, val_output = input_train[train_index], input_train[val_index], \
                                                       output_train[train_index], output_train[val_index]
    loss.append(get_loss(train_input, train_output, val_input, val_output))
    train_count+=1
                                                           
loss = np.array(loss)
np.savetxt("./loss/loss_hln10_nhl6_cd_by_cl_kfold_noalpha", np.c_[loss])
print(loss)
print(np.mean(loss))                                   
