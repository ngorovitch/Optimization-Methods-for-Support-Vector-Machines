import sys
sys.path.insert(0, '../lib')
import numpy as np
from homeworkLib import SVMClassifier
import pandas as pd

if __name__ == "__main__":
    # The hyper params where choosed by using...: hyper_params_tunning.py
    kernel_hyper_param = 0.1; C = 1
    #kernel_hyper_param = 1; C = 2**-3
    kernel = 'RBF' # 'polynomial' or 'RBF'

    SVM = SVMClassifier(kernel_h_p = kernel_hyper_param, C = C, kernel = kernel)

    # In the training set, each example consists of 256 input values
    # and 1 output value representing the coresponding digit
    data_set_train_2 = pd.read_csv("../../Data/Train_2.csv")
    data_set_train_8 = pd.read_csv("../../Data/Train_8.csv")
    data_set_train_full = data_set_train_2.append(data_set_train_8, ignore_index=True)
    data_set_train_full.loc[data_set_train_full['0'] == 2, ['0']] = 1
    data_set_train_full.loc[data_set_train_full['0'] == 8, ['0']] = -1

    # Full training set
    training_set_inputs = np.asarray(data_set_train_full.drop(['0'], axis =1))
    training_set_outputs = np.asarray(data_set_train_full[['0']])


    data_set_test_2 = pd.read_csv("../../Data/Test_2.csv")
    data_set_test_8 = pd.read_csv("../../Data/Test_8.csv")
    data_set_test_full = data_set_test_2.append(data_set_test_8, ignore_index=True)
    data_set_test_full.loc[data_set_test_full['0'] == 2, ['0']] = 1
    data_set_test_full.loc[data_set_test_full['0'] == 8, ['0']] = -1

    # Full test set
    test_set_inputs = np.asarray(data_set_test_full.drop(['0'], axis =1))
    test_set_outputs = np.asarray(data_set_test_full[['0']])

    # Train the SVM model using the training set.
    SVM.fit_MVP(training_set_inputs, training_set_outputs, verbose = False)

    # Test the neural network both on the training and the testing data
    result_output_train = SVM.predict(training_set_inputs)
    result_output_test = SVM.predict(test_set_inputs)

    print()
    print("Optimization routine chosen: cvxopt")
    print()
    print("Chosen kernel: ", kernel)
    print()
    print("Difference m(a)-M(a): ", SVM.m_a-SVM.M_a)
    print()
    print("Classifcation rate on the training set: "+str(SVM.get_error(result_output_train, training_set_outputs)*100)+"%")
    print()
    print("Classifcation rate on the test set: "+str(SVM.get_error(result_output_test, test_set_outputs)*100)+"%")
    print()
    print("Time for finding the KKT point (in seconds): ", SVM.training_time)
    print()
    print("Number of optimization iterations: ", SVM.NbrIterations)
    print()
    print("Value of the "+kernel+" kernel hyper parameter: ", kernel_hyper_param)
    print()
    print("Value of C: ", C)