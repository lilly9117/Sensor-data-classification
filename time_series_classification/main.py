# from utils.utils import generate_results_csv
from utils.utils import create_directory
from utils.utils import read_dataset
import os
import numpy as np
import sys
import sklearn
import utils
# from utils.constants import CLASSIFIERS
# from utils.constants import ITERATIONS
# from utils.utils import read_all_datasets


def fit_classifier():
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]

    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

    classifier.model_fit(x_train, y_train, x_test, y_test, y_true)


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=False):
    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'lstm_fcn':
        from classifiers import lstm_fcn
        return inception.Classifier_LSTMFCN(output_directory, input_shape, nb_classes, verbose)



############################################### main

# change this directory for your machine
root_dir = '../../preprocessing_data' # data

# archive_name = sys.argv[1]
    
dataset_name = 'kimm_df'

classifier_name = sys.argv[1]


output_directory = 'results/' + dataset_name + '_' + classifier_name + '/'

test_dir_df_metrics = output_directory + 'df_metrics.csv'

print('Method: ', dataset_name, ' ', classifier_name)

if os.path.exists(test_dir_df_metrics):
    print('Already done')
else:

    create_directory(output_directory)
    datasets_dict = read_dataset(root_dir, dataset_name)

    fit_classifier()

    print('DONE')

    # the creation of this directory means
    create_directory(output_directory + '/DONE')
