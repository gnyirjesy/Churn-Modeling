# Modeling Customer Churn
Perform exploratory data analysis and develop Machine Learning models to predict telco customer churn. Three different models are compared for performance: CatBoost, XGBoost, and Random Forest.

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Features](#features)
* [Status](#status)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)

## General info
The goal of churn prediction is to model customer data to identify customers likely to stop spending with the business. Once customers are identified, actions can be taken to intervene in the customer's journey and improve customer retention, and therefore improve the business. 

This project goes through EDA first, performing necessary cleaning steps outlined below:

* Correcting data types where necessary
* Removing uniform features 
* Removing features with high diversity of values
* Converting Yes/No features to binary
* Imputing values if necessary
* Viewing distributions of numeric features and the binary classifier

Then, the CatBoost, XGBoost, and Random Forest Models are created and improved through the following steps:

* Preliminary model run
* Removal of features with low feature_importances to reduce noise
* Hyper parameter tuning using RandomSearchCV and GridSearchCV

## Technologies
* Python 3.7

## Setup
Use requirements.txt file to install required packages, or install packages and versions listed below:

* pandas==1.0.1
* numpy==1.18.1
* catboost==0.23
* matplotlib==3.1.3
* scikit_learn==0.23.1
* xgboost==1.1.1

Data can be found on [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)

## Features
The following three Machine Learning models are included:

* CatBoost
* XGBoost
* Random Forest

To-do list:
* Add results to README

## Status
Project is: _in progress_

## Contact
Created by [Gabrielle Nyirjesy](https://www.linkedin.com/in/gabrielle-gabby-nyirjesy-524728a9/) - feel free to contact me!

## Acknowledgements
* Data pulled from [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
