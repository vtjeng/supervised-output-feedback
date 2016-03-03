import numpy as np
import os
import sklearn.linear_model
import matplotlib.pyplot as plt

target_directory = './resources/run_1'
target_filename = 'full'

X_data = np.genfromtxt(os.path.join(target_directory, target_filename+"_trainX.csv"), delimiter=',')
Y_data = np.genfromtxt(os.path.join(target_directory, target_filename+"_trainY.csv"), delimiter=',')
# If I'm still using the validation set above, it's because it's smaller and imports faster. Otherwise, you should ignore this message.

print('Succesfully imported dataset with of size {0}'.format(X_data.shape))

lr = sklearn.linear_model.LinearRegression()

lr.fit(X_data, Y_data)

lr.

print ("Regression weights for linear features = ")
print lr.coef_
plt.plot(lr.coef_)
plt.show()

Y_predicted = lr.predict(X_data)

# visualizing the difference between the actual and predicted values
# from this example http://scikit-learn.org/stable/auto_examples/plot_cv_predict.html
fig, ax = plt.subplots()
ax.scatter(Y_data, Y_predicted)
ax.plot([Y_data.min(), Y_data.max()], [Y_data.min(), Y_data.max()], 'k--', lw=4)
ax.set_xlabel('DP Controller')
ax.set_ylabel('Reactive Controller')
plt.show()