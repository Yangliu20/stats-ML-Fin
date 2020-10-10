# Data Analysis Intern Projects

## Table of contents

[Implied Volatility analysis with LASSO regression](#lasso)

[Classification and prediction of Implied Volatility with Decision Tree](#tree)

## <div id='lasso'></div>Implied Volatility analysis with LASSO regression

**Input data**: Implied volatility and 36 historical volatilities

**Object**: SSE 510050

**Achievements**: 

- Build a regression model, using implied volatility (IV) as dependent variable and 36 historical volatilities (HVs) as independent variables
- Prevent overfitting by using t-test, variance inflation factor and LASSO model, finally choosing LASSO model
- Reduce the dimension of features from 36 to 13
- Make out-of-sample predictions, also compare the performance of static model (use a single model from fixed in-sample data) and rolling model (update in-sample data and the model everyday)

<img src="https://github.com/Yangliu20/stats-ML-Fin/blob/main/docs/images/lasso_out_of_sample_test.png" width = 60% height = 60%/>

## <div id='tree'></div>Classification and prediction of Implied Volatility with Decision Tree

**Input data**: Implied volatility and 13 historical volatilities

**Object**: SSE 510050

**Achievements**: 

- Write my own decision tree classifier algorithm based on NumPy, and compare its performance with that of scikit-learn
- Prevent overfitting by cutting branches (maximal depth, information gain, and minimal sample points ...)
- Build a decision tree to classify IV trend based on the ratios of current IV and HVs
- Pick the best criterion among all nodes to generate trading signals

<img src="https://github.com/Yangliu20/stats-ML-Fin/blob/main/docs/images/tree.PNG" width = 60% height = 60%/>
