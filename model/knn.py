import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, f1_score, accuracy_score

class Classifier_KNN:

    def euclidian_distance(x1,x2):
        return np.linalg.norm(x1-x2)

    def make_distance_matrix(X_train, X_test, w=60, distance = euclidian_distance):
        """ This function returns the distance matrix between samples of X_train and X_tes according to a 
        similarity measure.
        INPUTS:
            - X_train a (n, p) numpy array with n:number of training samples and m: number of features
            - X_test a (m, p) numpy array with m: number of test samples and m as above
            - w DTW window
            - distance_type the type of distance to consider for the algorithm ['euclidian', 'DTW']
        OUTPUTS:
            - dis_m a (m,n) numpy array with dist_m[i,j] = distance(X_test[i,:], X_train[j,:])
        """
        
        # Distance matrix calculation
        n = X_train.shape[0]
        m = X_test.shape[0]  
        dist_m = np.zeros((m,n))
        for row, test_spl in enumerate(X_test):
            for col, train_spl in enumerate(X_train):
                if distance == euclidian_distance:
                    dist_row_col = distance(test_spl, train_spl)
                    dist_m[row,col] = dist_row_col
                else:
                    dist_row_col = distance(test_spl, train_spl, w)
                    dist_m[row,col] = dist_row_col                    
        return dist_m

    def find_k_closest(dist_m, y_train, k):
        """ This function returns the most represented label among the k nearest neighbors of each sample from
        X_test.
        INPUTS:
            - dist_m a (m,n) numpy array with dist_m[i,j] = distance(X_test[i,:], X_train[j,:])
            - y_train a (n,) numpy array with X_train labels
            - k number of neighbors to consider (int)
        OUPUTS:
            - y_pred a (m,) numpy array of predicted labels for X_test
        """
        knn_indexes = np.argsort(dist_m)[:,:k]
        knn_labels = y_train[knn_indexes]
        y_pred = mode(knn_labels, axis=1)[0]
        return y_pred

    def find_k_best(dist_m, y_train, y_test, k_range=np.arange(1,22)):
        k_range = np.arange(1,22) # range of k to test
        precision_score = np.empty(k_range.shape) # we are going to store f1 scores here
        # now we loop over k_range and compute f1_scores...
        for k in k_range:
            y_pred = find_k_closest(dist_m, y_train, k=k)
            precision_score[k-1] = precision_score(y_test, y_pred, average='macro')
        return k_range[np.argmax(precision_score)]

    def gridsearch(X_train, X_test, y_train, y_test):
        grid_params = {
            'n_neighbors': [3,5,7,11,19],
            'weights' : ['uniform', 'distance'],
            'metric' :  ['euclidean','minkowski']
        }
        gs = GridSearchCV(KNeighborsClassifier(),grid_params,verbose = 1, cv = 3, n_jobs = -1)

        gs_results = gs.fit(X_train, y_train)

        print("Best Parameters: {}".format(gs.best_params_))
        print("Best Cross-validity Score: {:.3f}".format(gs.best_score_))
        print("Test set Score: {:.3f}".format(gs.score(X_test, y_test)))

    def model_fit(X_train,X_test,y_train,y_test):
        dist_m = make_distance_matrix(X_train, X_test)

        k_best = find_k_best(dist_m, y_train, y_test, k_range=np.arange(1,22))
        y_pred = find_k_closest(dist_m, y_train, k=k_best)

        print("Parameters:")
        print("k = {}".format(k_best))
        print("\n")

        print("Test set report")
        print(f1_score(y_test, y_pred))

        print("accuracy score")
        print(accuracy_score(y_test, y_pred))

        print("precision score")
        print(precision_score(y_test, y_pred))