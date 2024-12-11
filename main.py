# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 09:08:59 2022

@author: Raneen_new
"""
import sys, getopt
#import numpy as np
from ReadData import ReadData
from FlySmote import FlySmote
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import math
from NNModel import SimpleMLP
import numpy as np
import matplotlib.pyplot as plt

#from imblearn.over_sampling import SMOTE
import time
def read_data(name,location):
    data_load = ReadData(name)
    Xtr,Ytr,Xte,Yte = data_load.load_data(location)
    return Xtr,Ytr,Xte,Yte

def check_imbalance(y_data):
    #print(y_data)
    cont=np.bincount(y_data)
    num_zeros = cont[0]
    num_ones = cont[1]
    threshould = 0
    if(num_zeros < num_ones):
        minority_lable = 0
        threshould = num_zeros/num_ones
    else:
        minority_lable = 1 
        threshould = num_ones/num_zeros
    #print(num_zeros)
    #print(num_ones)
    return minority_lable, threshould

def run(argv):
    data_name = ''
    dir_name = ''
    k_value = 4 #bank = 5, compass=5, Adult = 5
    r_value = 0.4 #bank=0.2,0.3 , compass =0.3, Adult = 0.2
    thre = 0.30
    num_clients = 3
    try:
      opts, args = getopt.getopt(argv,"hf:d:k:r:",["file_name=","directory_name=","k_value=","r_value="])
    except getopt.GetoptError:
      print ('Train_example_dataset.py -f <file name> -d <directory name> -k <samples from miniority> -r <ratio of new samples >')
      sys.exit(2)
    print (opts)
    for opt, arg in opts:
      if opt == '-h':
         print ('main.py -f <dataname> -d <directory name>')
         sys.exit()
      elif opt in ("-f", "--file"):
         data_name = arg
      elif opt in ("-d", "--directory"):
         dir_name = arg
      elif opt in ("-k", "--kvalue"):
         k_value = arg
      elif opt in ("-r", "--rvalue"):
         r_value = arg
     
    Xtr,Ytr,Xte,Yte = read_data(data_name,dir_name)
    minority_lable = 0
    minority_lable,threshould = check_imbalance(Ytr)
    fly_smote = FlySmote(Xtr,Ytr,Xte,Yte)
    clients = fly_smote.create_clients(Xtr, Ytr, num_clients, initial='client')
    clients_batched = dict()
    for (client_name, data) in clients.items():
        clients_batched[client_name] = fly_smote.batch_data(data)
    
    #process and batch the test set  
    test_batched = tf.data.Dataset.from_tensor_slices((Xte, Yte)).batch(len(Yte))
    #print('Number of client datasets: {l}'.format(l=len(test_batched)))
    client_names = list(clients_batched.keys())
    bs = list(clients_batched[client_name])[0][0].shape[0]
    #first calculate the total training data points across clinets
    global_count = sum([tf.data.experimental.cardinality(clients_batched[client_name]).numpy() for client_name in client_names])*bs
    local_count = tf.data.experimental.cardinality(clients_batched['client_1']).numpy()*bs
    print('Number of client datasets: {l}'.format(l=len(clients_batched)))
    print('First dataset: {d}'.format(d=clients_batched['client_1']))
    lr = 0.01 
    comms_round = 50
    loss='binary_crossentropy'
    metrics = ['accuracy']
    optimizer = SGD(lr=lr, 
                    decay=lr / comms_round, 
                    momentum=0.9
                   ) 
    smlp_global = SimpleMLP()
    num_layers_mult=1
    n=num_layers_mult
    global_model = smlp_global.build(Xtr,n)
    sensitivity_= []
    specificity_= []
    BalanceACC_= []
    G_mean_= []
    FP_rate_= []
    FN_rate_= []
    accuracy_= []
    loss_= []
    mcc_=[]
    #commence global training loop
    start = time.time()
    end_time = []
    e=EarlyStopping(patience=5,restore_best_weights=True)   
    for comm_round in range(comms_round):
        #print('Communication round: {}'.format(comm_round))
        # get the global model's weights - will serve as the initial weights for all local models
        global_weights = global_model.get_weights()
        
        #initial list to collect local model weights after scalling
        scaled_local_weight_list = list()
    
        #randomize client data - using keys
        client_names= list(clients_batched.keys())
        #random.shuffle(client_names)
        
        #loop through each client and create new local model
        for client in client_names:
            smlp_local = SimpleMLP()
            local_model = smlp_local.build(Xtr, n)
            local_model.compile(loss=loss, 
                          optimizer=optimizer, 
                          metrics=metrics)
            
            #set local model weight to the weight of the global model
            local_model.set_weights(global_weights)
            
            #here we should get the traning data from clients and try to do ksmote and see changes
            #if the data is imbalanced do ksmote, else normal traning       
            xte = []
            yte = []
            for(X_test, Y_test) in clients_batched[client]:
                i = 0
                while (i <len(X_test)):       
                    xte.append(X_test[i].numpy())
                    yte.append(Y_test[i].numpy())
                    i +=1
            minority_lable,threshould = check_imbalance(yte)
            #print(threshould)
            if(threshould<=thre):
                # print("imbalance")
                #here we change k and r to see how affect our result
                Xtr_new,Ytr_new = fly_smote.create_synth_data(xte,yte, minority_lable,k_value,r_value)
                added_points = len(Xtr_new) - len(xte)
                #split data for validation set
                Xtr_new,x_val,Ytr_new,y_val=train_test_split(Xtr_new,Ytr_new,test_size=0.10,shuffle=True)
                data = list(zip(Xtr_new, Ytr_new))
                btach_data = fly_smote.batch_data(data, bs=4)
                #fit local model with client's data
                local_model.fit(btach_data,validation_data=(x_val,y_val),callbacks=[e], epochs=1, verbose=0)
            else:
                Xtr_new,x_val,Ytr_new,y_val=train_test_split(xte,yte,test_size=0.10,shuffle=True)
                data = list(zip(Xtr_new, Ytr_new))
                btach_data = fly_smote.batch_data(data, bs=4)
                #fit local model with client's data
                added_points = 0
                local_model.fit(clients_batched[client], epochs=1, verbose=0)
            
            #scale the model weights and add to list       
            client_names = list(clients_batched.keys())
            bs = list(clients_batched[client_name])[0][0].shape[0]
            #first calculate the total training data points across clinets
            global_count = sum([tf.data.experimental.cardinality(clients_batched[client_name]).numpy() for client_name in client_names])*bs
            global_count = global_count + (added_points)
            # get the total number of data points held by a client
            local_count = len(Xtr_new)
            scaling_factor = local_count/global_count
                    
            scaled_weights = fly_smote.scale_model_weights(local_model.get_weights(), scaling_factor)
            scaled_local_weight_list.append(scaled_weights)
            
            #clear session to free memory after each communication round
            K.clear_session()
            
        #to get the average over all the local model, we simply take the sum of the scaled weights
        average_weights = fly_smote.sum_scaled_weights(scaled_local_weight_list)
        
        #update global model 
        global_model.set_weights(average_weights)
    
        #test global model and print out metrics after each communications round
        for(X_test, Y_test) in test_batched:
            global_acc, global_loss,conf = fly_smote.test_model(X_test, Y_test, global_model, comm_round)
            TN = conf[0][0]
            FP = conf[0][1]
            FN = conf[1][0]
            TP = conf[1][1]
            sensitivity = TP/(TP+FN) 
            specificity = TN/(FP+TN)
            BalanceACC = (sensitivity+specificity)/2
            G_mean = math.sqrt(sensitivity*specificity)
            FN_rate= FN/(FN+TP) 
            FP_rate = FP/(FP+TN) 
            mcc = ((TP*TN)-(FP*FN))/(math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
            #add the data to arrays
            sensitivity_.append(sensitivity)
            specificity_.append(specificity)
            BalanceACC_.append(BalanceACC)
            G_mean_.append(G_mean)
            FP_rate_.append(FP_rate)
            FN_rate_.append(FN_rate)
            accuracy_.append(global_acc)
            loss_.append(global_loss)
            mcc_.append(mcc)
            
        end_time.append(((time.time()) - start))

    print('Accuracy: {l}'.format(l=accuracy_[comms_round-1]))
    print('BalanceAccuracy: {l}'.format(l=BalanceACC_[comms_round-1]))
    plt.plot(BalanceACC_)
if __name__ == '__main__':
    run(sys.argv[1:])