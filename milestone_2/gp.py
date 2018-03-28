import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
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
RBFNLPD = []
Matern1NLPD = []
Matern2NLPD = []
Matern3NLPD = []
i = 0

for train,test in kf.split(X):

    
    kernel1 = RBF()+WhiteKernel()
    gp1 = GaussianProcessRegressor(kernel=kernel1).fit(X[train],Y[train])
    m,s = gp1.predict(X[test],return_std=True)
    RBFNLPD.append(sum(.5*np.log(2*np.pi*(s**2))+((Y[test]-m)**2/(2*(s**2))))/test.shape[0])
    
    kernel2 = Matern()+WhiteKernel()
    gp2 = GaussianProcessRegressor(kernel=kernel2).fit(X[train],Y[train])
    m,s = gp2.predict(X[test],return_std=True)
    Matern1NLPD.append(sum(.5*np.log(2*np.pi*(s**2))+((Y[test]-m)**2/(2*(s**2))))/test.shape[0])
    
    kernel3 = Matern(nu=0.5)+WhiteKernel()
    gp3 = GaussianProcessRegressor(kernel=kernel3).fit(X[train],Y[train])
    m,s = gp3.predict(X[test],return_std=True)
    Matern2NLPD.append(sum(.5*np.log(2*np.pi*(s**2))+((Y[test]-m)**2/(2*(s**2))))/test.shape[0])
    
    kernel4 = Matern(nu=2.5)+WhiteKernel()
    gp4 = GaussianProcessRegressor(kernel=kernel4).fit(X[train],Y[train])
    m,s = gp4.predict(X[test],return_std=True)
    Matern3NLPD.append(sum(.5*np.log(2*np.pi*(s**2))+((Y[test]-m)**2/(2*(s**2))))/test.shape[0])
    i = i + 1
    print(i)
    
    
RBFNLPD = np.array(RBFNLPD)
Matern1NLPD = np.array(Matern1NLPD)
Matern2NLPD = np.array(Matern2NLPD)
Matern3NLPD = np.array(Matern3NLPD)

print("RBF Kernel")
print("10-Fold CV Mean NLPD: " + str(RBFNLPD.mean()))
print("10-Fold CV Standard Deviation NLPD: " + str(RBFNLPD.std()))
print("")
print("Matern Kernel nu=0.5")
print("10-Fold CV Mean NLPD: " + str(Matern2NLPD.mean()))
print("10-Fold CV Standard Deviation NLPD: " + str(Matern2NLPD.std()))
print("")
print("Matern Kernel nu=1.5")
print("10-Fold CV Mean NLPD: " + str(Matern1NLPD.mean()))
print("10-Fold CV Standard Deviation NLPD: " + str(Matern1NLPD.std()))
print("")
print("Matern Kernel nu=2.5")
print("10-Fold CV Mean NLPD: " + str(Matern3NLPD.mean()))
print("10-Fold CV Standard Deviation NLPD: " + str(Matern3NLPD.std()))