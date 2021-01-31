import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt 
from scipy.stats import mode
from sklearn.metrics import classification_report, f1_score
from sklearn import preprocessing
import time
import matplotlib.pylab as plt 
from feature_extract import make_feature_vector 
from feature_extract_cur import make_cur_feature_vector
import sklearn
from sklearn.model_selection import train_test_split

def load_file(dataset_path):
    df = pd.read_csv(dataset_path)
    df = df.drop(['Unnamed: 0'], axis = 'columns')
    df = sklearn.utils.shuffle(df)
    # y_test = df_test.values[:, -1]
    y = df.values[:, -1]
    x = df.iloc[:,:-1]
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1, shuffle =True, random_state = 1004) 

    return x_train, x_test, y_train, y_test

def load_curfile(R_dataset,T_dataset,S_dataset):
    df_R = pd.read_csv(R_dataset)
    df_R = df_R.drop(['Unnamed: 0'], axis = 'columns')
    y = df_R.values[:, -1]
    x_R = df_R.iloc[:,:-1]

    df_T = pd.read_csv(T_dataset)
    df_T = df_T.drop(['Unnamed: 0'], axis = 'columns')
    x_T = df_T.iloc[:,:-1]

    df_S = pd.read_csv(S_dataset)
    df_S = df_S.drop(['Unnamed: 0'], axis = 'columns')
    x_S = df_R.iloc[:,:-1]

    return x_R, x_T, x_S, y

if __name__=="__main__":
    # dataset_path = '../preprocessing_data/kimm.csv'
    data = 'cur'
    
    if data == 'kimm' or data == 'vibration':
        dataset_path = '../preprocessing_data/vibration_interp.csv'
        X_train_raw, X_test_raw, y_train, y_test = load_file(dataset_path) 
        X_train = make_feature_vector(X_train_raw, Te=1/50)
        X_test = make_feature_vector(X_test_raw, Te=1/50)
        
    elif data == 'cur':
        R_dataset = '../preprocessing_data/curR.csv'
        T_dataset = '../preprocessing_data/curT.csv'
        S_dataset = '../preprocessing_data/curS.csv'

        X_R, X_T, X_S, y = load_curfile(R_dataset,T_dataset, S_dataset)

        X = make_cur_feature_vector(X_R, X_T, X_S, Te=1/50)

        X = pd.DataFrame(X)
        y = pd.DataFrame(y)

        df = pd.concat([X,y], axis=1)
        df = sklearn.utils.shuffle(df)
        y = df.values[:, -1]
        x = df.iloc[:,:-1]
        
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.1, shuffle =True, random_state = 1004) 


    df_X_train = pd.DataFrame(X_train)
    df_X_test = pd.DataFrame(X_test)
    df_y_train = pd.DataFrame(y_train)
    df_y_test = pd.DataFrame(y_test)
        
    print("X_train shape : {}".format(df_X_train.shape))
    print("X_test shape: {}".format(df_X_test.shape))
    print("y_train shape : {}".format(df_y_train.shape))
    print("y_test shape : {}".format(df_y_test.shape))

    df_X_train.to_csv('X_train.csv')
    df_X_test.to_csv('X_test.csv')
    df_y_train.to_csv('y_train.csv')
    df_y_test.to_csv('y_test.csv')
