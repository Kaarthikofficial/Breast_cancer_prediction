# Breast Cancer prediction
It focus on predicting the patient having breast cancer or not with the available data points. This project uses classification supervised learning algorithm SVM for classifying the target. 

## Model creation

### Load data
To start with, first load the data using pandas and find some descriptive statistics of the data. This will help us to
understand the data in better manner. Then look for any missing or null values in the data.

### Data preprocessing
This is an important step in any model creation where we need to process the data to optimize the outcome. It starts 
with finding outliers and handling it, balance the dataset by finding whether target is balanced or not(for 
classification problem), scaling the values using standardization or normalization.

### Feature selection and feature engineering
Feature selection is a crucial step where we need to address the curse of dimensionality. Identify the potential 
features in the data and leave the unimportant feature will help us to improve the model. Feature engineering is 
done if two or more features tend to give some useful feature when combined. Feature extraction can also be done
with the help of PCA to shrink the dimensions without data loss.

### Split data and evaluation
Final step in model creation is to split the data into training and test sets. After the splitting, apply the
appropriate model over the data and find the results. At last, evaluate it with the help of different evluation 
metrics like confusion matrix, ROC-AUC etc.,

### Hyperparameter tuning
If the result from the model is not upto the level, we can improvise it using hyper parameter tuning. It varies 
from model to model, first identify the parameters that can be tuned further and applied it with the help of
grid search method. 

## App build
After all the process of building the model, I used it in our web app using pickle library through
which we can retrieve the created model and use it for the new data. Publish it with the help of streamlit.

## Learning outcomes
- *Statistical analysis of data*
- *Processing the data*
- *Suitable model selection*
- *Optimize the model*

