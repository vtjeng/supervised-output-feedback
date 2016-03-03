__author__ = 'peteflorence'

import numpy as np
import sklearn.cross_validation
import os

'''
Splits a data file into test, training, and validation sets of specified size.
'''

target_directory = '../resources/run_1'
target_filename = 'full'
target_file = target_filename + '.csv'

test_fraction = 0.2
validation_fraction = 0.2
random_seed=42

def getData(name):
    data = np.genfromtxt(name, delimiter=',')
    # Returns column matrices
    X = data[:,0:-1]
    Y = data[:,-1]
    return X, Y

X, Y = getData(os.path.join(target_directory, target_file))

print('Succesfully imported file.')

X_train, Y_train, X_2, Y_2 = \
    sklearn.cross_validation.train_test_split(X, Y, test_size=test_fraction + validation_fraction, random_state=random_seed)

X_test, Y_test, X_validation, Y_validation = \
    sklearn.cross_validation.train_test_split(X, Y, test_size=validation_fraction/(test_fraction+validation_fraction), random_state=random_seed)

np.savetxt(os.path.join(target_directory, target_filename + "_trainX.csv"), X_train, delimiter=",")
np.savetxt(os.path.join(target_directory, target_filename + "_trainY.csv"), Y_train, delimiter=",")
np.savetxt(os.path.join(target_directory, target_filename + "_testX.csv"), X_test, delimiter=",")
np.savetxt(os.path.join(target_directory, target_filename + "_testY.csv"), Y_test, delimiter=",")
np.savetxt(os.path.join(target_directory, target_filename + "_validationY.csv"), X_validation, delimiter=",")
np.savetxt(os.path.join(target_directory, target_filename + "_validationY.csv"), Y_validation, delimiter=",")
