import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, f1_score, accuracy_score


class Classifier_SVM:

    def __init__(self,X_train,X_test,y_train,y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return

    def gridsearch(self, X_train, y_train):
        tuned_parameters = {
            'C': (np.arange(10,12,0.2)), 'gamma': (np.arange(0.1,10,0.1)), 'kernel': ['rbf']
                   }
        svm_model= SVC()
        model_svm = GridSearchCV(svm_model, tuned_parameters,cv=10,scoring='accuracy', verbose = 3)

        model_svm.fit(X_train, y_train)
        print("Best Parameters: {}".format(model_svm.best_params_))
        print("Best Cross-validity Score: {:.3f}".format(model_svm.best_score_))
        print("Test set Score: {:.3f}".format(model_svm.score(self.X_test, self.y_test)))

    def model_fit(self):

        ## kernel 설정
        # svc_rbf = SVC(kernel='rbf', C=10, gamma=0.001) #10, 0,001
        # svc_poly = SVC(kernel='poly', C=10)
        # svc_sigmoid = SVC(kernel='sigmoid', C=200)
        svc_linear = SVC(kernel='linear', C=80)

        svc_linear.fit(self.X_train, self.y_train) # 모델트레이닝

        y_pred=svc_linear.predict(self.X_test)

        print('Accuracy Score:')
        print(accuracy_score(self.y_test,y_pred))

        # print('Precision Score:')
        # print(precision_score(self.y_test,y_pred))