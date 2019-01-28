import sys
sys.path.insert(0, '../lib')
from homeworkLib import SVMClassifier
import pandas as pd
import numpy as np
from tqdm import tqdm

# here we use a K-fold cross validation to minimize the hyperparameters    
def objective(params, train_data, test_data):
    kernel = 'RBF' # 'polynomial' or 'RBF'
    routine = 'cvxopt' # 'quadprog' or 'cvxopt'
    Cs = params[0]; kernel_params = params[1]
    best_params = {'C': 1, 'kernel_h_p': 1, 'best_test_error': 0, 'best_train_error': 0}
    errors_per_iterations = []
    
    training_set_inputs = np.asarray(train_data.drop(columns=['0']))
    training_set_outputs = np.asarray(train_data[['0']])
    test_set_inputs = np.asarray(test_data.drop(columns=['0']))
    test_set_outputs = np.asarray(test_data[['0']])
    
    for C in tqdm(Cs):
        for kernel_h_p in kernel_params:
            print()
            print("C = ", C)
            print("kernel_h_p = ", kernel_h_p)
            SVM = SVMClassifier(kernel_h_p = kernel_h_p, C = C, kernel = kernel)
            # Train the SVM model using the training set.
            SVM.fit(training_set_inputs, training_set_outputs, routine = routine, verbose = False)
            # Test the SVM model both on the training and the testing data
            result_output_train = SVM.predict(training_set_inputs)
            result_output_test = SVM.predict(test_set_inputs)
            error_test = SVM.get_error(result_output_test, test_set_outputs)*100
            error_train = SVM.get_error(result_output_train, training_set_outputs)*100
            print("Train_error: ", error_train)
            print("Test_error: ", error_test)
            errors_per_iterations.append({"C": C, "kernel_h_p": kernel_h_p, "Train_error": error_train, "Test_error": error_test})
            if (error_test > best_params['best_test_error']):
                best_params['best_test_error'] = error_test
                best_params['best_train_error'] = error_train
                best_params['C'] = C
                best_params['kernel_h_p'] = kernel_h_p                         
    return best_params, errors_per_iterations
 

#let's optimize the hyper parameters
k = 2
data_set_train_2 = pd.read_csv("../../Data/Train_2.csv")
data_set_train_8 = pd.read_csv("../../Data/Train_8.csv")
train_data = data_set_train_2.append(data_set_train_8, ignore_index=True)
train_data[train_data[['0']] == 2] = 1
train_data[train_data[['0']] == 8] = -1

data_set_test_2 = pd.read_csv("../../Data/Test_2.csv")
data_set_test_8 = pd.read_csv("../../Data/Test_8.csv")
test_data = data_set_test_2.append(data_set_test_8, ignore_index=True)
test_data[test_data[['0']] == 2] = 1
test_data[test_data[['0']] == 8] = -1

result, details = objective([[1],[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]], train_data, test_data)

best_C = result['C']; best_kernel_hyper_param = result['kernel_h_p']; best_train_error = result['best_train_error']
best_test_error = result['best_test_error']
print()
print("best_C = ", best_C)
print("best_kernel_hyper_param = ", best_kernel_hyper_param)
print("best_test_error = ", best_test_error)
print("best_train_error = ", best_train_error)

GridSearch_details = pd.DataFrame(details)
with open('../../Report/GridSearch_details.csv', 'a') as f:
    GridSearch_details.to_csv(f, sep = ',', index=False, header=False)