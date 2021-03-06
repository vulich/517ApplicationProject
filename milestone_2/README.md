# Milestone 2
For this milestone, I fit the wine data with a Gaussian process using four different kernels. The four kernels used were the RBF kernel and the Matern kernel with 𝜈=0.5, 1.5, and 2.5. The RBF kernel leads to infinitely differentiable functions. The Matern kernel with 𝜈=0.5 is the same as the absolute exponential kernel which leads to continuous but non-differentiable functions. The Matern kernel with 𝜈=1.5 leads to once-differentiable functions. The Matern kernel with 𝜈=2.5 leads to twice-differentiable functions. The difference in the functions generated by each of these kernels can be visualized in their priors.
![RBF Prior](https://raw.githubusercontent.com/vulich/517ApplicationProject/master/milestone_2/Plots/Prior1.png)
![Matern 0.5 Prior](https://raw.githubusercontent.com/vulich/517ApplicationProject/master/milestone_2/Plots/Prior2.png)
![Matern 1.5 Prior](https://raw.githubusercontent.com/vulich/517ApplicationProject/master/milestone_2/Plots/Prior3.png)
![Matern 2.5 Prior](https://raw.githubusercontent.com/vulich/517ApplicationProject/master/milestone_2/Plots/Prior4.png)

To choose between these kernels I used 10-fold cross validation. For the error measure, I used the negative log predictive density, since it takes into account both the predicted mean and standard deviation, which gives us a better measure of error than something that only looks at predicted mean. The results of the kernel comparison were:

RBF Kernel  
10-Fold CV Mean NLPD: 1.1020272820977661  
10-Fold CV Standard Deviation NLPD: 0.11405170688874963  

Matern Kernel 𝜈=0.5  
10-Fold CV Mean NLPD: 1.092544373073559  
10-Fold CV Standard Deviation NLPD: 0.10300318844973788  

Matern Kernel 𝜈=1.5  
10-Fold CV Mean NLPD: 1.098842201008224  
10-Fold CV Standard Deviation NLPD: 0.1077398469728382  

Matern Kernel 𝜈=2.5  
10-Fold CV Mean NLPD: 1.101355644167883  
10-Fold CV Standard Deviation NLPD: 0.11115255089976277  

Of the four kernels tested, the Matern kernel with 𝜈=0.5 was the best, with both the lowest cross-validation mean and smallest cross validation standard deviation. The fact that a non-differentiable function was the best fit for our data suggests that the target function is most likely fairly complicated and not smooth.

## To run the code
To run the code in this folder, you need to have python 3.6, scikit-learn, numpy, and matplotlib. Download the dataset from [here](https://archive.ics.uci.edu/ml/datasets/wine+quality) and place winequality-red.csv in the dataset folder. Run gpPrior.py to generate the plots of the GP priors for the different kernels and run gp.py to fit them to the data and calculate the cross validation errors for each kernel. The cross validation takes a while, so it will print out a loop index each time it finishes with one of the ten iterations to give a sense of progress. gp.py will create a runtime warning. It doesn't cause any issues and as far as I can tell, it comes from an issue inside of scikit-learn, so I can't really fix it.

#### Citations
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.