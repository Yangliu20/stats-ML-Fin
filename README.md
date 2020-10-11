# Data Analysis Intern Projects

## Table of contents

[Implied Volatility analysis](#lasso)

[Classification and prediction of Implied Volatility](#tree)

[Recognition and prediction of market regime](#gmm)

[Compression and analysis of intraday high-frequency data](#compress)

## <div id='lasso'></div>Implied Volatility analysis

**Input data**: Implied volatility and 36 historical volatilities

**Object**: SSE 510050

**Methods**: Linear regression, OLS, LASSO regression

**Achievements**: 

- Build a regression model, using implied volatility (IV) as dependent variable and 36 historical volatilities (HVs) as independent variables
- Prevent overfitting by using t-test, variance inflation factor and LASSO model, finally choosing LASSO model
- Reduce the dimension of features from 36 to 13
- Make out-of-sample predictions, also compare the performance of static model (use a single model from fixed in-sample data) and rolling model (update in-sample data and the model everyday)

<img src="https://github.com/Yangliu20/stats-ML-Fin/blob/main/docs/images/lasso_out_of_sample_test.png" width = 50% height = 50%/>

## <div id='tree'></div>Classification and prediction of Implied Volatility

**Input data**: Implied volatility and 13 historical volatilities

**Object**: SSE 510050

**Methods**: Decision Tree

**Achievements**: 

- Write my own decision tree classifier algorithm based on NumPy, and compare its performance with that of scikit-learn
- Prevent overfitting by cutting branches (maximal depth, information gain, and minimal sample points ...)
- Build a decision tree to classify IV trend based on the ratios of current IV and HVs
- Pick the best criterion among all nodes to generate trading signals

<img src="https://github.com/Yangliu20/stats-ML-Fin/blob/main/docs/images/tree.PNG" width = 60% height = 60%/>

## <div id='gmm'></div>Recognition and prediction of market regime

**Input data**: Daily close price and volume

**Object**: Shanghai Composite Index

**Methods**: Gaussian Mixture Model

**Achievements**: 

- Adopt Gaussian Mixture Model to classify everyday market regime based on logRet_1, logRet_5, logDel, logVol_5
- Determine the label for every market regime (up, down, or other) by combining components with similar cumulative daily return
- Develop a method to dynamically choose the optimal number of components

<img src="https://github.com/Yangliu20/stats-ML-Fin/blob/main/docs/images/gmm1.png" width = 50% height = 50%/>

<img src="https://github.com/Yangliu20/stats-ML-Fin/blob/main/docs/images/gmm2.png" width = 50% height = 50%/>

## <div id='compress'></div>Compression and analysis of intraday high-frequency data

**Input data**: every minute close price

**Object**: SSE 50 Index; Shanghai Composite Index

**Methods**: Fourier Transformation; Clustering algorithms (K-means, DBSCAN); Singular Spectrum Analysis

**Achievements**: 

- Apply Fourier Transformation to intraday high-frequency price data (every-minute), remove noises, and reduce the dimension from 120 to 10
- Cluster individual days with similar fluctuating patterns before noon, using DBSCAN algorithm

<img src="https://github.com/Yangliu20/stats-ML-Fin/blob/main/docs/images/ifft.png" width = 50% height = 50%/>

<img src="https://github.com/Yangliu20/stats-ML-Fin/blob/main/docs/images/cluster.png" width = 50% height = 50%/>

- Implement Singular Spectrum Analysis to extract information from the data, remove noises, reconstruct a smooth time series, and make predictions

<img src="https://github.com/Yangliu20/stats-ML-Fin/blob/main/docs/images/ssa1.png" width = 50% height = 50%/>

<img src="https://github.com/Yangliu20/stats-ML-Fin/blob/main/docs/images/ssa2.png" width = 50% height = 50%/>
