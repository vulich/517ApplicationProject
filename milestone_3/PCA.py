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
kf = KFold(10)

err1 = []
err2 = []
err3 = []

for dim in range(1,12):

    errors = []
    PCAerrors = []
    KPCAerrors = []
    
    
    for train, test in kf.split(X):
        
        #transform with PCA fit using training data
        pca = PCA(dim)
        pca.fit(X[train])
        pcaX = pca.transform(X)
        
        #transform with kernel PCA
        kpca = KernelPCA(dim, kernel="rbf", gamma=.001)
        kpca.fit(X[train])
        kpcaX = kpca.transform(X)
        
        #Set range of hyperparameters to try
        params = {'alpha':np.arange(0,1,.05)}
        
        #Calculate 10-fold cross validation error for Ridge regression with each value of lambda
        model = GridSearchCV(Ridge(), params, cv=10, scoring="neg_mean_squared_error")
        
        #Calculate validation error for non-transformed data
        model.fit(X[train],Y[train])
        pred = model.best_estimator_.predict(X[test])
        errors.append(sum((pred-Y[test])**2)/test.shape[0])
        
        #Calculate validation error with PCA transform
        model.fit(pcaX[train],Y[train])
        pred = model.best_estimator_.predict(pcaX[test])
        PCAerrors.append(sum((pred-Y[test])**2)/test.shape[0])
        
        #Calculate validation error with kernel PCA
        model.fit(kpcaX[train],Y[train])
        pred = model.best_estimator_.predict(kpcaX[test])
        KPCAerrors.append(sum((pred-Y[test])**2)/test.shape[0])
        
        
    print(errors)
    print(PCAerrors)
    print(KPCAerrors)
    
    errors = np.array(errors)
    PCAerrors = np.array(PCAerrors)
    KPCAerrors = np.array(KPCAerrors)
    
    
    print("10-Fold CV error: " + str(errors.mean()))
    print("10-Fold CV error with dimensionality reduction: " + str(PCAerrors.mean()))
    print("10-Fold CV error with kernel dimensionality reduction: " + str(KPCAerrors.mean()))
    err1.append(errors.mean())
    err2.append(PCAerrors.mean())
    err3.append(KPCAerrors.mean())
    
plt.figure()
x = np.linspace(1,11,11)
plt.plot(x,err1)
plt.plot(x,err2)
plt.plot(x,err3)
plt.legend(["No Dim. Reduction", "PCA", "Kernel PCA"])
plt.xlabel("Components in Dimensionality Reduction")
plt.ylabel("10-Fold CV Error")
plt.title("10-Fold CV Error vs. PCA components")
plt.savefig("Plots/CVerrors.png")
plt.show()


    
    
    