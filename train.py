import os
import sys
import numpy as np
import pandas as pd

def create_classifier(classifier_name):
    if classifier_name == 'knn':
        from model import knn
        return knn.Classifier_KNN()

if __name__=="__main__":

    X_train = pd.read_csv('kimm/X_train.csv')
    X_test = pd.read_csv('kimm/X_test.csv')
    y_train = pd.read_csv('kimm/y_train.csv')
    y_test = pd.read_csv('kimm/y_test.csv')

    classifier_name = sys.argv[1]

    classifier = create_classifier(classifier_name)

    classifier.model_fit(X_train,X_test,y_train,y_test)
