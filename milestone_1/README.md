# Milestone 1
For this milestone I did ridge regression on the red wine dataset. I used the python package scikit-learn to do the regression and matplotlib to create plots. To choose the hyperparameter lambda in the ridge regression, I calculated the 10-fold cross-validation mean squared error for various values of lambda and chose the lambda that minimized the error.
![this plot](https://raw.githubusercontent.com/vulich/517ApplicationProject/master/milestone_1/Plots/hyperparameter_selection.png)
The best lambda was .567, and the lowest cross-validation mean squared error was 0.434062. After learning lambda, I used the entire dataset to train an estimator.

## To run the code
To run the code in this folder, you need to have python 3.6, scikit-learn, and matplotlib. Download the dataset from [here](https://archive.ics.uci.edu/ml/datasets/wine+quality) and place winequality-red.csv in the dataset folder. Run regression.py to learn a ridge regression estimator from the data and generate the plot above.

#### Citations
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.