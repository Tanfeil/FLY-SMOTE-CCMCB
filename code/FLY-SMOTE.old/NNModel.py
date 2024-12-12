# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 10:15:51 2022

@author: Raneen_new
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization,Input
from tensorflow.keras.callbacks import EarlyStopping
class SimpleMLP:
    @staticmethod
    def build(x_train,n):
        
        model = Sequential()
        model.add(Input(shape=(x_train.shape[1],)))
        model.add(Dense(x_train.shape[1],activation='relu'))#,input_shape=(x_train.shape[1],)))
        model.add(Dropout(0.25))
        model.add(Dense(n*x_train.shape[1],activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n*x_train.shape[1],activation='relu'))

        model.add(Dense(1,activation='sigmoid'))
        #print(model.summary())
        return model