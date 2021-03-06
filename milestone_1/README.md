# Milestone 1
## Dataset
The dataset used in this project relates the rating of 1,600 Vinho Verde red wines to their chemical properties. Each wine is rated from 0 to 10, using the median rating from at least 3 wine experts. Each wine had its fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol measured in a lab. The goal of this milestone is to use a regression model to predict the ratings based on a wine's chemical properties. Having a way to predict wine ratings from the chemical properties of wine would be useful because it would allow vineyards to predict how a wine will be rated and give them a more scientific approach to improving their wines by altering their process to create wine with chemical properties predicted to be good.
## Statistics
First, I computed statistics on the wine ratings. I found that most of the wine ratings were clustered around 5-7.

![Histogram of Y](https://raw.githubusercontent.com/vulich/517ApplicationProject/master/milestone_1/Plots/ratings_histogram.png)

Mean(Y)=5.6360 Var(Y)=0.6518
## Regression
For this milestone I did ridge regression on the red wine dataset. I used the python package scikit-learn to do the regression and matplotlib to create plots. To choose the hyperparameter lambda in the ridge regression, I calculated the 10-fold cross-validation mean squared error for various values of lambda and chose the lambda that minimized the error.
![this plot](https://raw.githubusercontent.com/vulich/517ApplicationProject/master/milestone_1/Plots/hyperparameter_selection.png)

The best lambda was .567. After learning lambda, I used the entire dataset to train an estimator and saved it to a file so that it can be easily used to get predictions of the quality of other wines. This achieved a 10-Fold cross validation mean squared error of 0.434063. Looking at the fitted coefficients suggests that low volatile acidity, low chlorides, and high sulfates are the most important of the measured features in determining a good wine.

## To run the code
To run the code in this folder, you need to have python 3.6, scikit-learn, numpy, and matplotlib. Download the dataset from [here](https://archive.ics.uci.edu/ml/datasets/wine+quality) and place winequality-red.csv in the dataset folder. Run regression.py to learn a ridge regression estimator from the data and generate the plot above.

#### Citations
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.