# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 09:14:05 2022

@author: Raneen_new
"""
import random
import warnings
from collections import Counter
import os, sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn import feature_extraction
from sklearn import preprocessing
from random import seed, shuffle
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler,StandardScaler

class ReadData:
    def __init__(self, name):
        self.data_name = name
    
    def load_data(self,location):
        Xtr = 0
        Ytr=0
        Xte=0
        Yte = 0
        if(self.data_name == "Bank"):
            Xtr,Ytr,Xte,Yte = self.load_bank(location)
        elif(self.data_name == "Adult"):
            Xtr,Ytr,Xte,Yte = self.load_adult(location)
        elif( self.data_name == "Compass"):
            Xtr,Ytr,Xte,Yte = self.load_compass(location)
        else:
            print("Please choose a correct dataset name")
        return Xtr,Ytr,Xte,Yte

    def load_bank(self,location):
        FEATURES_CLASSIFICATION = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact",
                                   "day", "month", "duration", "campaign", "pdays", "previous",
                                   "poutcome"]  # features to be used for classification
        CONT_VARIABLES = ["age", "balance", "day", "duration", "campaign", "pdays",
                          "previous"]  # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
        CLASS_FEATURE = "y"  # the decision variable
        SENSITIVE_ATTRS = ["marital"]
    
        # COMPAS_INPUT_FILE = "bank-full.csv"
        #COMPAS_INPUT_FILE = "bank-full.csv"
        COMPAS_INPUT_FILE = "%s.csv"%(location)
        df = pd.read_csv(COMPAS_INPUT_FILE)
    
        # convert to np array
        data = df.to_dict('list')
        for k in data.keys():
            data[k] = np.array(data[k])
    
        """ Feature normalization and one hot encoding """
    
        # convert class label 0 to -1
        y = data[CLASS_FEATURE]
        y[y == "yes"] = 1
        y[y == 'no'] = -1
        y = np.array([int(k) for k in y])
    
        X = np.array([]).reshape(len(y), 0)  # empty array with num rows same as num examples, will hstack the features to it
        x_control = defaultdict(list)
    
        feature_names = []
        for attr in FEATURES_CLASSIFICATION:
            vals = data[attr]
            if attr in CONT_VARIABLES:
                vals = [float(v) for v in vals]
                vals = preprocessing.scale(vals)  # 0 mean and 1 variance
                vals = np.reshape(vals, (len(y), -1))  # convert from 1-d arr to a 2-d arr with one col
    
            else:  # for binary categorical variables, the label binarizer uses just one var instead of two
                lb = preprocessing.LabelBinarizer()
                lb.fit(vals)
                vals = lb.transform(vals)
    
            # add to sensitive features dict
            if attr in SENSITIVE_ATTRS:
                x_control[attr] = vals
    
            # add to learnable features
            X = np.hstack((X, vals))
    
            if attr in CONT_VARIABLES:  # continuous feature, just append the name
                feature_names.append(attr)
            else:  # categorical features
                if vals.shape[1] == 1:  # binary features that passed through lib binarizer
                    feature_names.append(attr)
                else:
                    for k in lb.classes_:  # non-binary categorical features, need to add the names for each cat
                        feature_names.append(attr + "_" + str(k))
    
        # convert the sensitive feature to 1-d array
        x_control = dict(x_control)
        for k in x_control.keys():
            assert (x_control[k].shape[1] == 1)  # make sure that the sensitive feature is binary after one hot encoding
            x_control[k] = np.array(x_control[k]).flatten()
    
        feature_names.append('target')
        Y = []
        for i in y:
            if (i == -1):
                Y.append(0)
            else:
                Y.append(1)
        Y = np.array(Y)
        x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
        Xtr = x_train
        Xte = x_test
        Ytr = y_train
        Yte = y_test
        return Xtr,Ytr,Xte,Yte
    
    def load_adult(self,location):
        #df = pd.read_csv('adult.csv',na_values=['?'])
        name = "%s.csv"%(location)
        #print(name)
        df = pd.read_csv(name,na_values=['?'])
        df=df.dropna()
        X=df.drop(['income'],axis=1)
        y=df[['income']]
        X.head()
        x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.10,shuffle=True)
        (x_train.shape,x_test.shape)
        df.columns[df.dtypes=='object']
        oh=ColumnTransformer([
            ('encoder',OneHotEncoder(drop='first',sparse=False),
             ['workclass', 'education', 'marital.status', 'occupation',
               'relationship', 'race', 'sex', 'native.country'])
        ],remainder='passthrough')
        
        pipeline=Pipeline([
            ('encoder',oh),
            ('scaler',StandardScaler())
        ])
        x_train.iloc[0,:]
        x_train=pipeline.fit_transform(x_train)
        x_test=pipeline.transform(x_test)
        pipeline[0].transformers_[0][1].categories_
        o1=OneHotEncoder(sparse=False)
        y_train=o1.fit_transform(y_train)
        y_test=o1.transform(y_test)
        o1.categories_
        Xtr = x_train
        Ytr = y_train
        Xte = x_test
        Yte = y_test
        Yte = np.argmax(Yte, axis=1)
        Ytr = np.argmax(Ytr, axis=1)
        return Xtr,Ytr,Xte,Yte

    def load_compass(self,location):

        # import pandas as pd
        compas_scores_raw = pd.read_csv("compas-scores-raw.csv")
        cox_violent_parsed = pd.read_csv("cox-violent-parsed.csv")
        cox_violent_parsed_filt = pd.read_csv("cox-violent-parsed_filt.csv")
        
        TARGET_COL = "Two_yr_Recidivism"
        df = pd.read_csv("propublica_data_for_fairml.csv")
        
        print(df.shape)
        #display(df.columns)
        df.head()
        df = df[df['Female']>0]
        
        # Here is a way to select these columns using the column names
            
        #feature_columns = ['Number_of_Priors', 'score_factor','Age_Above_FourtyFive', 'Age_Below_TwentyFive', 'African_American','Asian', 'Hispanic', 'Native_American', 'Other', 'Female',       'Misdemeanor']
        feature_columns = ['Number_of_Priors', 'score_factor','Age_Above_FourtyFive', 'Age_Below_TwentyFive', 'Misdemeanor']
        
        data = df[feature_columns].values
        y = df['Two_yr_Recidivism'].values
        
        
        train_x, test_x, train_y, test_y = train_test_split(data, y, test_size=0.20, shuffle=True, stratify=y)
        Xtr = train_x
        Xte = test_x
        Ytr = train_y
        Yte = test_y
        return Xtr,Ytr,Xte,Yte