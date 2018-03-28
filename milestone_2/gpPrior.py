import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from matplotlib import pyplot as plt

x = np.linspace(-3,3,1000).reshape(-1,1)

kernel1 = RBF()
gp1 = GaussianProcessRegressor(kernel=kernel1)
y_samples1 = gp1.sample_y(x,10)
plt.figure()
plt.plot(x,y_samples1)
plt.title("RBF Kernel Prior")
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Plots/Prior1.png')


kernel2 = Matern()
gp2 = GaussianProcessRegressor(kernel=kernel2)
y_samples2 = gp2.sample_y(x,10)
plt.figure()
plt.plot(x,y_samples2)
plt.title("Matern Kernel Prior nu=1.5")
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Plots/Prior3.png')

kernel3 = Matern(nu=0.5)
gp3 = GaussianProcessRegressor(kernel=kernel3)
y_samples3 = gp3.sample_y(x,10)
plt.figure()
plt.plot(x,y_samples3)
plt.title("Matern Kernel Prior nu=0.5")
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Plots/Prior2.png')

kernel4 = Matern(nu=2.5)
gp4 = GaussianProcessRegressor(kernel=kernel4)
y_samples4 = gp4.sample_y(x,10)
plt.figure()
plt.plot(x,y_samples4)
plt.title("Matern Kernel Prior nu=2.5")
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Plots/Prior4.png')

plt.show()