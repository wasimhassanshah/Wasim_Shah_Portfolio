## Wasim_Shah_Portfolio 

# [Project 1: House Price Predictor-Regression-Model : Project Overview](https://github.com/wasimhassanshah/Predicting-House-Prices-in-Python)
* Applied the Machine Laerning Algorithms : Linear Regression, Decision Tree Regression, Random Forest Regressor, and XGboost Regressor to predict House prices 
* After Exploratory Data Analysis, feature engineering is done to extract useful features to improve the accuracy of the model
* Significant features are used to train the linear regression model, and the best model is selected accordingly.
* Hyperparameter tunning is done on Decision Tree Regressor, Random Forest Regressor, and XGboost Regressor to find optimal parameters for the model using RandomSearchCV
* Optimal parameters are used to reach the best Decision Tree, Random Forest and XGboost Regressor model
* Out of four , the best performing model is found to be of a XG boost Regressor.
![](images/pics.PNG)


## [Project 2: Customer-Segmentation-Kmeans-Clustering-Model](https://github.com/wasimhassanshah/Customer-Segmentation-Kmeans-Clustering)
* Performed an exploratory analysis on the dataset.
* Applied Principal Component Analysis for Dimensionality Reduction
* Applied K-means clustering algorithm in order to segment customers.
* Customer Segmentation : Dividing customer base into several groups of individuals that share a similarity in different ways that are relevant to marketing such as gender, age, interests, and miscellaneous spending habits.
![](/images/PCA.png)
![](/images/k1.png)
![](/images/clusters.png)

## [Project 3: Electricity-Consumption-Forcasting-using-Deep-Learning-LSTM](https://github.com/wasimhassanshah/energy-prediction-model)
* Applied Data Analysis and Visualizations on the available dataset
* Predicted trend of future energy consumption using LSTM
* Predicted values of trained model (red line) are almost closed to actual values (green line)
![](/images/Energypredictvsactual.png)

## [Project 4: Cutomer-churn-Classification-Model-ANN-LR](https://github.com/wasimhassanshah/Cutomer-churn-ANN-LR)
* buildt a deep learning model ANN and a Supervised Learning Model Logistic Regression to predict the churn and comparing both models
* Cleaned the dataset to remove the missing variables
* Hot encoded several categorical variables.
* Explored the correlation between several features and the target variable.
* Used a correlation matrix to explore if there was any correlation between different features.
* fit a logistic regression model using all of the features.The accuracy score of this model is found to be 0.79. Then I fit ANN model. .The accuracy score of this model is found to be 0.78. Here Logistic Regression Model obtained the best accuracy, recall, F1 scores, and the best precision score, making it the most reliable machine learning classifier for this data set.
![](/images/Cchurn.png)
![](/images/ANNCM.png)
![](/images/LRCM.png)


## [Project 5: Stock_Price_Prediction_RNN](https://github.com/wasimhassanshah/Stock_Price_Prediction_RNN)
* Purpose of this project is to build stock price predictor model to help the quantitative traders to take calculated decisions. 
* In this project,Closing price is used for prediction.
* Three sequential LSTM layers have been stacked together and one dense layer is used to build the RNN model using Keras deep learning library.
* Results
- Original data close price plot
![](/images/OriginalColsepricevalue.png)
- After the training the fitted curve with original stock price:
![](/images/OrgnaltrainpredctLSTMStock.png)
- Future 30 days Prediction of Close price
![](/images/30dayspedctLSTMStock.png)
- Final plot Original + Predicted for 30 days Close price value of stocks
![](/images/FinalLSTMPredctStock.png)


## [Project 6: Employee-Turnover-Classification-Model](https://github.com/wasimhassanshah/Employee-Turnover-Prediction-Model)
* Encoded categorical features using dummy variables.
* Performed exploratory data analysis to find the correlation between the features using seaborn and matplotlib library.
* Performed feature elimination to compare models such as Logistic Regression and Random Forest Classifier
* Fit the model
* Used Receiver operating characteristic (ROC) analysis to analyse the performance of both classifiers
* Showed the most important features which are going to influence whether an employee will leave the company or not 
![](/images/EPRF.png)
![](/images/EPLR.png)
![](/images/EPTRN.png)


## [Project 7: Month_wise_Retail_Sales_Prediction_Model](https://github.com/wasimhassanshah/Monthly_Retail_Sales_Forecasting_fpp)
* fpp: Data for "Forecasting: principles and practice"
* fpp models (snaive, ETS (Exponential Smoothing models) and ARIMA models) are used to predict Retail sales for future 24 months (2018,2019) using 1992 to 2017 data *  * Almost Accurate prediction in result for future 2 year retail sales
![](/images/ETS_FORECATS.png)
![](/images/ARIMA.png)
![](/images/Actual_vs_forecasting_comparison_graphical.png)
![](/images/Actual_vs_forecasting_comparison_values.png) 
