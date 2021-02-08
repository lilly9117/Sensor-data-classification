# from utils.utils import generate_results_csv
from utils.utils4cur import create_directory
from utils.utils4cur import read_datasetR
from utils.utils4cur import read_datasetT
from utils.utils4cur import read_datasetS

# from utils.utils import transform_mts_to_ucr_format
# from utils.utils import visualize_filter
# from utils.utils import viz_for_survey_paper
# from utils.utils import viz_cam
import os
import numpy as np
import sys
import sklearn
import utils
from utils.constants import CLASSIFIERS
# from utils.constants import ARCHIVE_NAMES
from utils.constants import ITERATIONS


def fit_classifier():
    Rx_train = datasets_dictR[dataset_Rname][0]
    y_train = datasets_dictR[dataset_Rname][1]
    Rx_test = datasets_dictR[dataset_Rname][2]
    y_test = datasets_dictR[dataset_Rname][3]

    Tx_train = datasets_dictT[dataset_Tname][0]
    Ty_train = datasets_dictT[dataset_Tname][1]
    Tx_test = datasets_dictT[dataset_Tname][2]
    Ty_test = datasets_dictT[dataset_Tname][3]

    Sx_train = datasets_dictS[dataset_Sname][0]
    Sy_train = datasets_dictS[dataset_Sname][1]
    Sx_test = datasets_dictS[dataset_Sname][2]
    Sy_test = datasets_dictS[dataset_Sname][3]


    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    ## y_true = np.argmax(y_test, axis=1)
    y_true = np.argmax(y_test, axis=1)
    # Ty_true = np.argmax(Ty_test, axis=1)
    # Sy_true = np.argmax(S_test, axis=1)

    if len(Rx_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        Rx_train = Rx_train.reshape((Rx_train.shape[0], Rx_train.shape[1], 1))
        Rx_test = Rx_test.reshape((Rx_test.shape[0], Rx_test.shape[1], 1))

    if len(Tx_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        Tx_train = Tx_train.reshape((Tx_train.shape[0], Tx_train.shape[1], 1))
        Tx_test = Tx_test.reshape((Tx_test.shape[0], Tx_test.shape[1], 1))

    if len(Sx_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        Sx_train = Sx_train.reshape((Sx_train.shape[0], Sx_train.shape[1], 1))
        Sx_test = Sx_test.reshape((Sx_test.shape[0], Sx_test.shape[1], 1))

    input_shape1 = Rx_train.shape[1:]
    print('inputshape 출력',input_shape1)
    input_shape2 = Tx_train.shape[1:]
    input_shape3 = Sx_train.shape[1:]

    print('input shape: ', input_shape1)


    classifier = create_classifier(classifier_name, input_shape1, input_shape2, input_shape3, nb_classes, output_directory)   
    print(type(classifier))
    # fit에 딕셔너리 형태의 인풋
    # classifier.model_fit(x_train, y_train, x_test, y_test, y_true)
    classifier.model_fit(Rx_train,Tx_train,Sx_train, y_train, Rx_test, Tx_test, Sx_test, y_test, y_true)
    #  x_train, y_train, x_val, y_val,y_true : model_fit의 인풋


def create_classifier(classifier_name, input_shape1,input_shape2,input_shape3, nb_classes, output_directory, verbose=True):
    if classifier_name == 'fcn4cur':
        from classifiers import fcn4cur
        return fcn4cur.Classifier_FCN(output_directory, input_shape1, input_shape2, input_shape3, nb_classes, verbose)
    # if classifier_name == 'resnet':
    #     from classifiers import resnet
    #     return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)


############################################### main

# change this directory for your machine
# root_dir = '../../preprocessing_data' # data
# 데이터 파일 경로
root_dir =  '../../preprocessing_data/'

# archive_name = sys.argv[1]
    
# 데이터 파일 이름
dataset_Rname = 'curR13만개'
dataset_Tname = 'curT13만개'
dataset_Sname = 'curS13만개'


classifier_name = sys.argv[1]


output_directory = 'results전류new/' + dataset_Rname + '_' + classifier_name  + '_sample_' + '/'

test_dir_df_metrics = output_directory + 'df_metrics.csv'

print('Method: ', dataset_Rname, ' ', classifier_name)

if os.path.exists(test_dir_df_metrics):
    print('Already done')
else:

    create_directory(output_directory)
    datasets_dictR = read_datasetR(root_dir, dataset_Rname)
    datasets_dictT = read_datasetT(root_dir, dataset_Tname)
    datasets_dictS = read_datasetS(root_dir, dataset_Sname)

    fit_classifier()

    print('DONE')

    # the creation of this directory means
    create_directory(output_directory + '/DONE')
