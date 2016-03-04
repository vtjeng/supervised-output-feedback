import tensorflow as tf
import numpy as np
import os

target_directory = './resources/run_1'
target_filename = 'small'

X_data = np.genfromtxt(os.path.join(target_directory, target_filename+"_validationX.csv"), delimiter=',')
Y_data = np.genfromtxt(os.path.join(target_directory, target_filename+"_validationY.csv"), delimiter=',')

X_subsample = X_data[0:10000]
Y_subsample = Y_data[0:10000]

print('Succesfully imported dataset with of size {0}'.format(X_data.shape))

num_features = X_data.shape[1]

x = tf.placeholder(tf.float32, [None, num_features])
y_ = tf.placeholder(tf.float32)

W = tf.Variable(tf.zeros([num_features, 1]))

y_pred = tf.matmul(x, W)

loss = tf.reduce_mean(tf.square(y_pred - y_)) # using least squares
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(loss)


init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

# Fit the line.
for step in xrange(1000):
    
    sess.run(train_step, feed_dict = {x: X_subsample, y_: Y_subsample}) # training on the full dataset
    if step % 2 == 0:
        print(step, np.transpose(sess.run(W)))
