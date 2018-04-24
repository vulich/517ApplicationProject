# Milestone 3

For this milestone I used Principal Component Analysis (PCA) and kernel PCA to reduce the dimensionality of the dataset before fitting it using the transformed data to fit a ridge regression model as in Milestone 1. For Kernel PCA, I used the RBF kernel. Below are plots of the data transformed into two dimensions using both the linear and non-linear PCA.
![PCA visualization](https://github.com/vulich/517ApplicationProject/blob/master/milestone_3/Plots/PCAviz.png?raw=true)

![KPCA visualization](https://github.com/vulich/517ApplicationProject/blob/master/milestone_3/Plots/KPCAviz.png?raw=true)


I tried using between 1 and all 11 components of the original data and used various kernel widths for the kernel PCA. I used 10-Fold cross validation error to select the best model of those tried. 

![CV error plot](https://raw.githubusercontent.com/vulich/517ApplicationProject/master/milestone_3/Plots/CVerrors.png)

As can be seen in the above figure, the ridge regression on the non-transformed data performed the best out of all the models, with kernel PCA performing strictly worse than any of the other methods. This would indicate that all the dimensions of the data are important, and projecting the data into a smaller dimensional space will lead to worse results. It also suggests that the unlabeled data doesn't have any sort of nonlinear structure that can be exploited to make regression easier.

## To run the code
To run the code in this folder, you need to have python 3.6, scikit-learn, numpy, and matplotlib. Download the dataset from [here](https://archive.ics.uci.edu/ml/datasets/wine+quality) and place winequality-red.csv in the dataset folder. Run PCA.py to use PCA and kernel PCA to reduce the dimension of the data and learn ridge regression estimators to generate the plot above.

#### Citations
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.