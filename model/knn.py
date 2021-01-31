import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, f1_score, accuracy_score


class Classifier_KNN:

    def __init__(self,X_train,X_test,y_train,y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return

    def gridsearch(self):
        grid_params = {
            'n_neighbors': [3,5,7,11,19],
            'weights' : ['uniform', 'distance'],
            'metric' :  ['euclidean','minkowski']
        }
        gs = GridSearchCV(KNeighborsClassifier(),grid_params,verbose = 1, cv = 3, n_jobs = -1)

        gs_results = gs.fit(self.X_train, self.y_train)

        print("Best Parameters: {}".format(gs.best_params_))
        print("Best Cross-validity Score: {:.3f}".format(gs.best_score_))
        print("Test set Score: {:.3f}".format(gs.score(self.X_test, self.y_test)))

    def model_fit(self,k):

        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

        knn.fit(self.X_train, self.y_train)

        print("Test set report")

        y_pred = knn.predict(self.X_test)
        print(y_pred)
        print(self.y_test)

        print('accuacy', knn.score(self.X_test, self.y_test))