import numpy as np
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt


#Load data from file
datafile = open("../dataset/winequality-red.csv")
datafile.readline()
data = np.loadtxt(datafile, delimiter=",")

#Split data into X and Y
X = data[:, :11]
Y = data[:, 11]

#Set range of hyperparameters to try
params = {'alpha':np.arange(0,1,.001)}

#Calculate 10-fold cross validation error for Ridge regression with each value of lambda
model = GridSearchCV(linear_model.Ridge(), params, cv=10, scoring="neg_mean_squared_error")
model.fit(X,Y)

#Find best value for lambda
a = model.best_params_['alpha']

#Plot the negative square error against lambda
figure = plt.figure()
plot = figure.add_subplot(111)
plot.set(title=r'$\lambda$ vs. 10-Fold Cross-Validation Error', ylabel='10-Fold Cross-Validation Mean Squared Error', xlabel=r'$\lambda$')
plot.plot(params['alpha'],-1*model.cv_results_['mean_test_score'])
plot.axvline(x=a,linestyle='--',color='red')
plot.text(a+.02,.4342,r'$\lambda^*=%s$'%(str(a)[:5]))
plt.savefig('Plots/hyperparameter_selection.png')
plt.show()

#Get best estimator that was trained on entire set
estimator = model.best_estimator_

#Get the CV error for the best estimator
error = -model.best_score_
print("10-Fold Cross Validation Error: " + str(error))

