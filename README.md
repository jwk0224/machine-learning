# machine-learning

Exercise projects on basic machine learning algorithms from Coursera Machine Learning course by Andrew Ng

## ex1_Linear Regresssion

- ex1.m - Octave/MATLAB script that steps you through the exercise
- ex1_multi.m - Octave/MATLAB script for the later parts of the exercise
- ex1data1.txt - Dataset for linear regression with one variable
- ex1data2.txt - Dataset for linear regression with multiple variables
- warmUpExercise.m - Simple example function in Octave/MATLAB
- plotData.m - Function to display the dataset
- computeCost.m - Function to compute the cost of linear regression
- gradientDescent.m - Function to run gradient descent
- computeCostMulti.m - Cost function for multiple variables
- gradientDescentMulti.m - Gradient descent for multiple variables
- featureNormalize.m - Function to normalize features
- normalEqn.m - Function to compute the normal equations

## ex2_Logistic Regresssion

- ex2.m - Octave/MATLAB script that steps you through the exercise
- ex2_reg.m - Octave/MATLAB script for the later parts of the exercise
- ex2data1.txt - Training set for the first half of the exercise
- ex2data2.txt - Training set for the second half of the exercise
- mapFeature.m - Function to generate polynomial features
- plotDecisionBoundary.m - Function to plot classifier’s decision boundary
- plotData.m - Function to plot 2D classification data
- sigmoid.m - Sigmoid Function
- costFunction.m - Logistic Regression Cost Function
- predict.m - Logistic Regression Prediction Function
- costFunctionReg.m - Regularized Logistic Regression Cost

## ex3_Multi-class Classification and Neural Networks

- ex3.m - Octave/MATLAB script that steps you through part 1
- ex3 nn.m - Octave/MATLAB script that steps you through part 2
- ex3data1.mat - Training set of hand-written digits
- ex3weights.mat - Initial weights for the neural network exercise
- displayData.m - Function to help visualize the dataset
- fmincg.m - Function minimization routine (similar to fminunc)
- sigmoid.m - Sigmoid function
- lrCostFunction.m - Logistic regression cost function
- oneVsAll.m - Train a one-vs-all multi-class classifier
- predictOneVsAll.m - Predict using a one-vs-all multi-class classifier
- predict.m - Neural network prediction function

## ex4_Neural Networks Learning

- ex4.m - Octave/MATLAB script that steps you through the exercise
- ex4data1.mat - Training set of hand-written digits
- ex4weights.mat - Neural network parameters for exercise4
- displayData.m - Function to help visualize the dataset
- fmincg.m - Function minimization routine (similar to fminunc)
- sigmoid.m - Sigmoid function
- computeNumericalGradient.m - Numerically compute gradients
- checkNNGradients.m - Function to help check your gradients
- debugInitializeWeights.m - Function for initializing weights
- predict.m - Neural network prediction function
- sigmoidGradient.m - Compute the gradient of the sigmoid function
- randInitializeWeights.m - Randomly initialize weights
- nnCostFunction.m - Neural network cost function

## ex5_Regularized Linear Regression and Bias v.s. Variance

- ex5.m - Octave/MATLAB script that steps you through the exercise
- ex5data1.mat - Dataset
- featureNormalize.m - Feature normalization function
- fmincg.m - Function minimization routine (similar to fminunc)
- plotFit.m - Plot a polynomial fit
- trainLinearReg.m - Trains linear regression using your cost function
- linearRegCostFunction.m - Regularized linear regression cost func- tion
- learningCurve.m - Generates a learning curve
- polyFeatures.m - Maps data into polynomial feature space
- validationCurve.m - Generates a cross validation curve

## ex6_Support Vector Machines

- ex6.m - Octave/MATLAB script for the first half of the exercise
- ex6data1.mat - Example Dataset 1
- ex6data2.mat - Example Dataset 2
- ex6data3.mat - Example Dataset 3
- svmTrain.m - SVM training function
- svmPredict.m - SVM prediction function
- plotData.m - Plot 2D data
- visualizeBoundaryLinear.m - Plot linear boundary
- visualizeBoundary.m - Plot non-linear boundary
- linearKernel.m - Linear kernel for SVM
- gaussianKernel.m - Gaussian kernel for SVM
- dataset3Params.m - Parameters to use for Dataset 3
- ex6_spam.m - Octave/MATLAB script for the second half of the exercise
- spamTrain.mat - Spam training set
- spamTest.mat - Spam test set
- emailSample1.txt - Sample email 1
- emailSample2.txt - Sample email 2
- spamSample1.txt - Sample spam 1
- spamSample2.txt - Sample spam 2
- vocab.txt - Vocabulary list
- getVocabList.m - Load vocabulary list
- porterStemmer.m - Stemming function
- readFile.m - Reads a file into a character string
- processEmail.m - Email preprocessing
- emailFeatures.m - Feature extraction from emails

## ex7_K-means Clustering and Principal Component Analysis

- ex7.m - Octave/MATLAB script for the first exercise on K-means
- ex7_pca.m - Octave/MATLAB script for the second exercise on PCA
- ex7data1.mat - Example Dataset for PCA
- ex7data2.mat - Example Dataset for K-means
- ex7faces.mat - Faces Dataset
- bird_small.mat - Example Image
- bird_small.png - Example Image
- displayData.m - Displays 2D data stored in a matrix
- drawLine.m - Draws a line over an exsiting figure
- plotDataPoints.m - Initialization for K-means centroids
- plotProgresskMeans.m - Plots each step of K-means as it proceeds
- runkMeans.m - Runs the K-means algorithm
- featureNormalize.m - Function to normalize features
- pca.m - Perform principal component analysis
- projectData.m - Projects a data set into a lower dimensional space
- recoverData.m - Recovers the original data from the projection
- findClosestCentroids.m - Findclosestcentroids(usedinK-means)
- computeCentroids.m - Compute centroid means (used in K-means)
- kMeansInitCentroids.m - Initialization for K-means centroids

## ex8_Anomaly Detection and Recommender Systems

- ex8.m - Octave/MATLAB script for first part of exercise
- ex8_cofi.m - Octave/MATLAB script for second part of exercise
- ex8data1.mat - First example Dataset for anomaly detection
- ex8data2.mat - Second example Dataset for anomaly detection
- ex8_movies.mat - Movie Review Dataset
- ex8_movieParams.mat - Parameters provided for debugging
- multivariateGaussian.m - Computes the probability density function for a Gaussian distribution
- visualizeFit.m - 2D plot of a Gaussian distribution and a dataset
- checkCostFunction.m - Gradient checking for collaborative filtering
- computeNumericalGradient.m - Numerically compute gradients
- fmincg.m - Function minimization routine (similar to fminunc)
- loadMovieList.m - Loads the list of movies into a cell-array
- movie ids.txt - List of movies
- normalizeRatings.m - Mean normalization for collaborative filtering
- estimateGaussian.m - Estimate the parameters of a Gaussian distribution with a diagonal covariance matrix
- selectThreshold.m - Find a threshold for anomaly detection
- cofiCostFunc.m - Implement the cost function for collaborative filtering
