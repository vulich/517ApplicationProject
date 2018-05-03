import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from matplotlib import pyplot as plt
import pickle
import math


import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import pickle
import math


#Load data from file
datafile = open("../dataset/winequality-red.csv")
datafile.readline()
data = np.loadtxt(datafile, delimiter=",")
datafile.close()

#Split data into X and Y
X = data[:, :11]
Y = data[:, 11]

#Get 10-Fold CV Splits
kf = RepeatedKFold(10,10)

errors = np.zeros((100,4))
i = 0

for train, test in kf.split(X):
    print(i)
    
    #transform with PCA fit using training data
    pca = PCA(6)
    pca.fit(X[train])
    pcaX = pca.transform(X)
    
    
    #fit linear regressor
    #Set range of hyperparameters to try
    params = {'alpha':np.arange(0,1,.05)}
    
    #Calculate 10-fold cross validation error for Ridge regression with each value of lambda
    model = GridSearchCV(Ridge(), params, cv=10, scoring="neg_mean_squared_error")
    
    #Calculate validation error
    model.fit(X[train],Y[train])
    pred = model.best_estimator_.predict(X[test])
    errors[i,0]=sum((pred-Y[test])**2)/test.shape[0]
    
    #Calculate validation error with PCA transform
    model.fit(pcaX[train],Y[train])
    pred = model.best_estimator_.predict(pcaX[test])
    errors[i,1]=sum((pred-Y[test])**2)/test.shape[0]
    
    #fit GP
    kernel = Matern(nu=0.5)+WhiteKernel()
    gp = GaussianProcessRegressor(kernel=kernel).fit(X[train],Y[train])
    pred = gp.predict(X[test])
    errors[i,2]=sum((pred-Y[test])**2)/test.shape[0]
    
    #fit GP with PCA transform
    gp = GaussianProcessRegressor(kernel=kernel).fit(pcaX[train],Y[train])
    pred = gp.predict(pcaX[test])
    errors[i,3]=sum((pred-Y[test])**2)/test.shape[0]
    
    i = i+1
    
#write errors to file easily readable in python
pickle.dump(errors,open("CVerrors","wb"))

#print mean error for each method
print(errors.mean(0))

#write errors to file for usage in R
errorFile = open("errors.txt","w")
for i in range(0,100):
    for j in range(0,4):
        errorFile.writelines(str(int(j/2))+" "+str(j%2)+" "+str(i)+" "+str(errors[i,j])+"\n")
