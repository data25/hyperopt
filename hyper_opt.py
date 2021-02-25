# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 12:55:33 2020

@author: Thiru Nadesan
"""
import numpy as np
import timeit
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler 
from hyperopt import fmin, tpe, hp, STATUS_OK, space_eval
import matplotlib.pyplot as plt

def objective(x):
    
    global model, input_train, output_train, output_test_tr, output_test, count, plot_prediction
    
    scaler = MinMaxScaler()

    a, b, c, alpha = x['a'], x['b'], x['c'], x['alpha']
    
    inputs = np.array([a, b, c, int(alpha)]).reshape(-1,4)
    
    input_scaler = scaler.fit(input_train)
    
    inputs = input_scaler.transform(inputs)

    prediction = model.predict(inputs)
    
    output_scaler = scaler.fit(output_train.reshape(-1,1))
    
    output_test = output_scaler.transform(output_test_tr.reshape(-1,1))
    
    prediction = output_scaler.inverse_transform(prediction)
    
    prediction = 1*prediction
    
    l_b = np.full((50,1), 40)

    u_b = np.full((50,1), 60)
    
    plot_l_b = np.linspace(20,40,50).reshape(-1,1)
    
    plot_u_b = np.linspace(40,60,50).reshape(-1,1)

    fig = plt.figure(figsize=(8,8))
    plt.plot(output_test_tr, output_test_tr, 'gx', label='Test Data')
    plt.plot(prediction, prediction, 'kx', mew=5, label = 'predicted minimum')
    plt.plot(l_b, plot_l_b, 'r--', label = 'lower bound')
    plt.plot(u_b, plot_u_b, 'b--', label = 'upper bound')
    plt.legend(loc = 'best')
    plt.xlabel('$C_L/C_D$')
    plt.ylabel('$C_L/C_D$')
    plt.xlim([-85, 85])
    plt.ylim([-85, 85])
    plt.text(-80,40, "Iteration = "+str(count), {'color':'k', 'fontsize':20})
    if prediction < plot_prediction:
        plot_prediction=prediction
    plt.text(-80,20, "Best minimum = "+str(plot_prediction[0][0]), {'color':'k', 'fontsize':20})
    fig.savefig("../Gaussian_process/ann_predicted_minimum_oc"+str(count)+".png", dpi=600)
    plt.show()
    
    count+=1
         
    return {'loss': prediction, 'status': STATUS_OK}

if __name__=="__main__":
    
    input_train = np.genfromtxt('./train_data/input_train_cl_by_cd.txt', delimiter=' ')
    output_train = np.genfromtxt('./train_data/output_train_cl_by_cd.txt')
    input_test = np.genfromtxt('./test_data/cl_by_cd_input_test.txt', delimiter=' ')
    output_test_tr = np.genfromtxt('./test_data/cl_by_cd_output_test.txt')
    
    ind= np.where((output_test_tr>=40) & (output_test_tr<=60))
    a_s = input_test[ind[0][0]:ind[0][-1], 0].reshape(-1,1)
    b_s = input_test[ind[0][0]:ind[0][-1], 1].reshape(-1,1)
    c_s = input_test[ind[0][0]:ind[0][-1], 2].reshape(-1,1)
    alpha_s = input_test[ind[0][0]:ind[0][-1], 3].reshape(-1,1)
     
    a_min, a_max = np.min(a_s[:,0]), np.max(a_s[:,0])
    b_min, b_max = np.min(b_s[:,0]), np.max(b_s[:,0])
    c_min, c_max = np.min(c_s[:,0]), np.max(c_s[:,0])
    alpha_min, alpha_max = np.min(alpha_s[:,0]), np.max(alpha_s[:,0])
    
    space = {
        'a': hp.uniform('a', a_min, a_max),
        'b': hp.uniform('b', b_min, b_max),
        'c': hp.uniform('c', c_min, c_max),
        'alpha': hp.uniform('alpha', alpha_min, alpha_max)
        }
    
    '''        
    space = {
        'a': hp.uniform('a', -1*a_max, -1*a_min),
        'b': hp.uniform('b', -1*b_max, -1*b_min),
        'c': hp.uniform('c', -1*c_max, -1*c_min),
        'alpha': hp.uniform('alpha', -1*alpha_max, -1*alpha_min)
        }'''
    
    max_evals = 2000
    
    model = load_model("./saved_model/xfoil_ann_cl_by_cd_hln10_nhl6_kfold2"+".h5")
    
    start = timeit.default_timer()
    
    count=1
    
    plot_prediction=[[0]]
    
    best = fmin(objective,
                space= space,
                algo=tpe.suggest,
                max_evals=max_evals
                )

    f=objective(space_eval(space, best))
    
    print("evals = {0}, maximum f = {1}, best = {2}".format(max_evals, f["loss"][0], best))
    print("Total time: {0}".format(timeit.default_timer() - start))     
    
    
    
    
    
    
    