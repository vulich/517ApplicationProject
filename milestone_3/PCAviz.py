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


pca = PCA(2)
pca.fit(X)
pcaX = pca.transform(X)
plt.figure()

for i in range(3,9):
    plt.scatter(pcaX[Y==i,0],pcaX[Y==i,1])
    
plt.title("PCA transformed data")
plt.savefig("Plots/PCAviz")
plt.show()

kpca = KernelPCA(2, kernel="rbf", gamma=.001)
kpca.fit(X)
kpcaX = kpca.transform(X)

plt.figure()

for i in range(3,9):
    plt.scatter(kpcaX[Y==i,0],pcaX[Y==i,1])
    
plt.title("Kernel PCA transformed data")
plt.savefig("Plots/KPCAviz")
plt.show()