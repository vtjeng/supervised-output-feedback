__author__ = 'peteflorence'

import numpy as np
import sklearn.cross_validation
import os

'''
Splits a data file into test, training, and validation sets of specified size.
'''

target_filename = 'small'

target_directory = '../resources/run_1'
target_file = target_filename + '.csv'

test_fraction = 0.25
validation_fraction = 0.15
random_seed = 42


def getData(name):
    data = np.genfromtxt(name, delimiter=',')
    # Returns column matrices

    # filter data to remove points which were in collision.
    # for these points, at least one sensor reading will be zero.
    filtered_data = np.array(filter(lambda x: min(x[0:-1])>0, data))

    X = filtered_data[:,0:-1]
    Y = filtered_data[:,-1]
    return X, Y

X, Y = getData(os.path.join(target_directory, target_file))

print('Succesfully imported file.')

X_train, X_2, Y_train, Y_2 = \
    sklearn.cross_validation.train_test_split(X, Y, test_size=test_fraction + validation_fraction, random_state=random_seed)

X_test, X_validation, Y_test, Y_validation = \
    sklearn.cross_validation.train_test_split(X_2, Y_2, test_size=validation_fraction/(test_fraction+validation_fraction), random_state=random_seed)

print('Generated training dataset of size {0}'.format(X_train.shape))
print('Generated test dataset of size {0}'.format(X_test.shape))
print('Generated validation dataset of size {0}'.format(X_validation.shape))

np.savetxt(os.path.join(target_directory, target_filename + "_trainX.csv"), X_train, delimiter=",")
np.savetxt(os.path.join(target_directory, target_filename + "_trainY.csv"), Y_train, delimiter=",")
np.savetxt(os.path.join(target_directory, target_filename + "_testX.csv"), X_test, delimiter=",")
np.savetxt(os.path.join(target_directory, target_filename + "_testY.csv"), Y_test, delimiter=",")
np.savetxt(os.path.join(target_directory, target_filename + "_validationX.csv"), X_validation, delimiter=",")
np.savetxt(os.path.join(target_directory, target_filename + "_validationY.csv"), Y_validation, delimiter=",")

print('Export complete.')