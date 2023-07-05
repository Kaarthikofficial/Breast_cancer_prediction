# Industrial_copper_analysis
This is about determining price prediction of copper based on various factors using various regression models and also involves classifying the model based on the status of bidding. 

## Data Preprocessing
    The most crucial step of data analysis starts with data preprocessing where we are able to understand what actually the data has
and how the data is distributed. If we get those basic descriptive statistics, we can move forward on how to approach the problem.
In this project, several pre-processing steps were carried out like change the data type of features, filling Na's with "-", remove
Nas' and drop unwanted features. 

### Univariate and Bivariate analysis
In this step, I have done some basic univariate analysis on numeric features and found some skewness in some features.
With the help of bivariate analysis I found the outliers present in the features correspond to target feature.

### Log transformation
I used log transformation to handle skewness in the features. Several features were positively skewed which can be
efficiently transformed by using log transformer. Following are the features transformed with the help of log transformer 
* Quantity_tons
* Thickness
* Selling price
Once after transformed, checked the distribution with the help of distplot in seaborn.

### Encoding features
Many categorical features in the data were transformed into numerical features. This step is 
very much important before getting into model building process.

## Model building
The data gets split into independent variables and target variable. Then the data is scaled using Standard Scaler 
and the model is again split into training set and testing set.
    I used Lazypredict, a powerful tool to identify the best ML model for regression and classification. After find
the best model from that. I used it along with hyperparameter tuning to improve the performance of the model.

## Model Evaluation
    For regression, I used MSE metric and R2 score to find the performance of the model and for classification I used
accuracy score and confusion matrix to evaluate the model's performance.
