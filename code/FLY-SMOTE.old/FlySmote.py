# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 09:52:47 2022

@author: Raneen_new
"""
import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization,Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import warnings

# Custom script 
#%matplotlibe inline
class FlySmote:
    def __init__(self, Xtr,Ytr,Xte,Yte):
        self.Xtr = Xtr
        self.Ytr = Ytr
        self.Xte = Xte
        self.Yte = Yte
    
    def create_clients(self,image_list, label_list, num_clients, initial='clients'):
        ''' return: a dictionary with keys clients' names and value as 
                    data shards - tuple of images and label lists.
            args: 
                image_list: a list of numpy arrays of training images
                label_list:a list of binarized labels for each image
                num_client: number of fedrated members (clients)
                initials: the clients'name prefix, e.g, clients_1 
                
        '''
    
        #create a list of client names
        client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]
    
        #randomize the data
        data = list(zip(image_list, label_list))
        random.shuffle(data)
    
        #shard data and place at each client
        size = len(data)//num_clients
        shards = [data[i:i + size] for i in range(0, size*num_clients, size)]
    
        #number of clients must equal number of shards
        assert(len(shards) == len(client_names))
    
        return {client_names[i] : shards[i] for i in range(len(client_names))} 
    
    def batch_data(self,data_shard, bs=4):
        '''Takes in a clients data shard and create a tfds object off it
        args:
            shard: a data, label constituting a client's data shard
            bs:batch size
        return:
            tfds object'''
        #seperate shard into data and labels lists
        data, label = zip(*data_shard)
        dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
        return dataset.shuffle(len(label)).batch(bs)
    
    
    def weight_scalling_factor(self,clients_trn_data, client_name):
        client_names = list(clients_trn_data.keys())
        #get the bs
        bs = list(clients_trn_data[client_name])[0][0].shape[0]
        #first calculate the total training data points across clinets
        global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
        # get the total number of data points held by a client
        local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
        return local_count/global_count


    def scale_model_weights(self,weight, scalar):
        '''function for scaling a models weights'''
        weight_final = []
        steps = len(weight)
        for i in range(steps):
            weight_final.append(scalar * weight[i])
        return weight_final
    
    
    
    def sum_scaled_weights(self,scaled_weight_list):
        '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
        avg_grad = list()
        #get the average grad accross all client gradients
        for grad_list_tuple in zip(*scaled_weight_list):
            layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
            avg_grad.append(layer_mean)
            
        return avg_grad


    def test_model(self, X_test, Y_test,model, comm_round):
        cce = tf.keras.losses.BinaryCrossentropy()
        logits = model.predict(X_test)
        preidect = np.around(logits)
        preidect = np.nan_to_num(preidect)
        Y_test = np.nan_to_num(Y_test)
        conf = (confusion_matrix(Y_test,preidect))   
        loss = cce(Y_test, preidect)
        acc = accuracy_score(preidect,Y_test)
        print('comm_round: {} | global_acc: {} | global_loss: {}'.format(comm_round, acc, loss))
        return acc, loss,conf
    
    def k_nearest_neighbors(self,data, predict, k):
        #k=8
        if len(data) >= k:
            warnings.warn('K is set to a value less than total voting groups!')
    
        distances = []
        count = 0
        for sample in data:
            euclidean_distance = np.linalg.norm(np.array(sample)-np.array(predict))
            distances.append([euclidean_distance,count])
            count+=1
        
        votes = [i[1] for i in sorted(distances)[:k]]
        #print(votes)
        #vote_result = Counter(votes).most_common(9)[0][0]
        return votes
    
    def kSMOTE(self,dmajor,dminor,k,r):
        S = []
        Ns = int(r * (len(dmajor) - len(dminor)))
        Nks = int(Ns / k)
        rb = []
        #pick a random k sample from dmin and save them in rb
        dmin_rand = random.sample(dminor, k)
        #do algorithem (choose the nearest neighbor and linear interpolation)
        for xb in dmin_rand:
            N= self.k_nearest_neighbors(dminor,xb,k)
            #do linear interpolation
            Sxb = []
            for s in range(Nks):
                j = N[0]
                #randome k sample
                j = random.randint(0, len(N))     
                ##here nearst neghber insted of dminor
                x_new = ((dminor[j]-xb) * random.sample(range(0, 1), 1))
                #while(j < len(N)):
                #    #here on random xb
                #    ind = N[j]
                #    x_new = x_new + ((dminor[ind]-xb) * random.sample(range(0, 1), 1))
                #    j += 1
                #x_new = x_new / len(N)         
                Sxb.append(xb + x_new)
            S.append(Sxb)
        return S
    
    def splitYtrain(self,Xtr, Ytr,minority_lable):
        #print(Ytr)
        dmaj_x = []
        dmin_x = []
        for i in range(len(Ytr)):
            if((Ytr[i]) == minority_lable):
                dmin_x.append(Xtr[i])
            else:
                dmaj_x.append(Xtr[i])
    
        return dmaj_x,dmin_x
    
    def create_synth_data(self,clinet_traning_x, clinet_traning_y, minority_lable,k,r):
        #create two data set from traning data (one for maj (ex.0 class) and one for min(ex.1 class)) 
        dmaj_x,dmin_x = self.splitYtrain(clinet_traning_x,clinet_traning_y,minority_lable)
        x_syn = self.kSMOTE(dmaj_x,dmin_x,k,r)
        # add the created synthatic data to the traning data
        Xtr_new = []
        Ytr_new = []
        # here merrge old traning data with the new synthatic data
        new_label =[] 
        for k in clinet_traning_y:
            if(k == minority_lable):
                new_label = k
                break
        
        for j in x_syn:
            for s in j:
                Xtr_new.append(s)
                Ytr_new.append(new_label)
                
        for k in clinet_traning_x:
            Xtr_new.append(k)
            
        for k in clinet_traning_y:
            Ytr_new.append(k)
        
        Xtr_new = np.array(Xtr_new)
        Ytr_new = np.array(Ytr_new)
    
        return Xtr_new,Ytr_new