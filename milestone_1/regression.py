import numpy as np
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import pickle


#Load data from file
datafile = open("../dataset/winequality-red.csv")
datafile.readline()
data = np.loadtxt(datafile, delimiter=",")
datafile.close()

#Split data into X and Y
X = data[:, :11]
Y = data[:, 11]

#Show statistics for Y
print("Mean(Y) = " + str(Y.mean()))
print("Var(Y) = " + str(Y.var()))
figure1 = plt.figure()
hist = figure1.add_subplot(111)
hist.set(title="Frequency of wine ratings", ylabel="Count", xlabel="Wine Rating")
hist.hist(Y,bins=[0,1,2,3,4,5,6,7,8,9,10])
plt.savefig('Plots/ratings_histogram.png')

#Set range of hyperparameters to try
params = {'alpha':np.arange(0,1,.001)}

#Calculate 10-fold cross validation error for Ridge regression with each value of lambda
model = GridSearchCV(linear_model.Ridge(), params, cv=10, scoring="neg_mean_squared_error")
model.fit(X,Y)

#Find best value for lambda
a = model.best_params_['alpha']

#Plot the negative square error against lambda
figure2 = plt.figure()
plot = figure2.add_subplot(111)
plot.set(title=r'$\lambda$ vs. 10-Fold Cross-Validation Error', ylabel='10-Fold Cross-Validation Mean Squared Error', xlabel=r'$\lambda$')
plot.plot(params['alpha'],-1*model.cv_results_['mean_test_score'])
plot.axvline(x=a,linestyle='--',color='red')
plot.text(a+.02,.4342,r'$\lambda^*=%s$'%(str(a)[:5]))
plt.savefig('Plots/hyperparameter_selection.png')


#Get best estimator that was trained on entire set and save it to a file
estimator = model.best_estimator_
estimatorFile = open('best_estimator','wb')
pickle.dump(estimator,estimatorFile)
estimatorFile.close()

#Get the CV error for the best estimator
error = -model.best_score_
print("10-Fold Cross Validation Error: " + str(error))

#Show plots
plt.show()

