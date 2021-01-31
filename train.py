import os
import sys
import numpy as np
import pandas as pd



def create_classifier(classifier_name, X_train,X_test,y_train,y_test):
    if classifier_name == 'knn':
        from model import knn
        return knn.Classifier_KNN(X_train,X_test,y_train,y_test)
    elif classifier_name == 'svm':
        from model import svm
        return svm.Classifier_SVM(X_train,X_test,y_train,y_test)
    elif classifier_name == 'fcn':
        from model import fcn
        return fcn.Classifier_FCN(X_train,X_test,y_train,y_test)

if __name__=="__main__":

    X_train = pd.read_csv('data/vib/X_train.csv')
    X_test = pd.read_csv('data/vib/X_test.csv')
    y_train = pd.read_csv('data/vib/y_train.csv')
    y_test = pd.read_csv('data/vib/y_test.csv')

    X_train = X_train.drop(['Unnamed: 0'], axis = 'columns')
    X_test = X_test.drop(['Unnamed: 0'], axis = 'columns')
    y_train = y_train.drop(['Unnamed: 0'], axis = 'columns')
    y_test = y_test.drop(['Unnamed: 0'], axis = 'columns')

    classifier_name = sys.argv[1]

    classifier = create_classifier(classifier_name,X_train,X_test,y_train,y_test)

    # classifier.gridsearch(X_train, y_train)

    # if문으로 완성하기
    classifier.model_fit()
    # classifier.model_fit()
