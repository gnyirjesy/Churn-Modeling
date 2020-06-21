#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[1]:


#Import Python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,StratifiedKFold,RandomizedSearchCV,GridSearchCV
from catboost import CatBoostClassifier
from sklearn.metrics import *
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


# In[2]:


#Define functions
#Set of functions for data cleaning
class clean():
    '''
    Functions used to clean data sets. 
    to_binary: convert features consisting of only yes/no values to a binary indicator
    identify_uniform: Remove columns that contain only one value (ignoring N/A)
    identify_diverse_cat: Identify and/or delete columns above a specified limit of unique values
    lable_encoding: Label encode categorical variables
    
    '''
    
    def to_binary(df):
        '''
        Convert features consisting of only yes/no values to a binary indicator
        to_binary(df) --> df with yes/no columns converted to 1/0
        '''
        for col in df:
            if len(df[col].unique() == 2) and not df[col].isna().values.any():
                if 'No' in df[col].unique() and 'Yes' in df[col].unique():
                    df[col] = np.where(df[col] == 'No', 0, 1)
        return(df)
    
    def convert_to_float_try_except(col):
        '''
        Convert variable to float, with try-except loop to catch any type errors and set errors to zero
        convert_to_float_try_except(df, col) --> df with column converted to float
        '''
        try:
            new_col = float(col)
        except ValueError:
            new_col = 0
        new_col = float(new_col)
        return(new_col)

    def identify_uniform(df):
        '''
        Identify features that contain only one value (ignoring N/A)
        identify_uniform(df) --> list of uniform columns
        '''
        uniform_vars = []
        for col in df.columns:
            if len((df[col][df[col] != 'nan']).value_counts()) <= 1:
                uniform_vars.append(col)
        print('The following features are uniform:\n', uniform_vars)
        
        return(uniform_vars)
    
    def identify_diverse_cat(df, limit, delete):
        '''
        Identify and/or delete columns above a specified limit of unique values
        identify_diverse_cat(df, limit, delete) --> df
        '''
        cat_cols = list(df.select_dtypes(exclude=[np.number]).columns.values)
        
        for col in cat_cols:
            num_unique = len(list(df[col].unique()))
            if num_unique >= limit:
                if delete:
                    df = df.drop([col], axis=1)
                    print(f'{col} column was deleted becuase it had {num_unique} categories')
                else:
                    print(f'{col} column has {num_unique} categories and should be binned or transformed')
        return(df)
                
    
    def label_encoding(df):
        '''
        Label encode categorical variables
        label_encoding(df) --> df with categorical features label encoded
        '''
        label_encoder = LabelEncoder()
        categorical_list = list(df.select_dtypes(exclude=[np.number]).columns)
        for col in categorical_list:
            df[col] = df[col].apply(str)
            df[col] = label_encoder.fit_transform(df[col])
        return(df)

#Set of functions for hyper parameter turning for CatBoost, XGBoost, and Random Forest models
class parameter_tuning():
    '''
    A set of functions to assist with hyper parameter tuning for models
    Functions: randomsearch, gridsearch
    '''
    def randomsearch(seed, x, y, random_grid, model):
        '''
        Run a random search on parameters for randomforest (model='rf'), xgboost (model='xgb'),
        and catboost (model='catboost') models
        randomsearch(seed, x, y, model) --> best_params
        '''
        if model == 'rf':
            estimator = RandomForestClassifier()         
            
        elif model == 'xgboost':
            estimator = XGBClassifier(loss_function='Logloss', random_seed=seed)
            
        elif model == 'catboost':
            estimator = CatBoostClassifier(loss_function='Logloss', random_seed=seed)
       
        strat_kfold = StratifiedKFold(n_splits = 4, shuffle = True, random_state = seed)
        scoring = make_scorer(recall_score)
        random_search = RandomizedSearchCV(estimator=estimator, param_distributions=random_grid,
                                           n_iter=100, cv=strat_kfold.split(x, y), verbose=3, random_state=seed,
                                           n_jobs=-1, scoring = scoring)
        random_search.fit(x, y)
        return(random_search.best_params_)
    
    def gridsearch(seed, x, y, random_grid, model):
        '''
        Run a full grid search on parameters for randomforest (model='rf'), xgboost (model='xgb'),
        and catboost (model='catboost') models
        gridsearch(seed, x, y, model) --> best_params
        '''
        if model == 'rf':
            estimator = RandomForestClassifier()

        elif model == 'xgboost':
            estimator = XGBClassifier()
            
        elif model == 'catboost':
            estimator = CatBoostClassifier()
       
        strat_kfold = StratifiedKFold(n_splits = 4, shuffle = True, random_state = seed)
        scoring = make_scorer(recall_score)
        grid_search = GridSearchCV(estimator=estimator, param_grid=random_grid, cv=strat_kfold.split(x, y),
                                   verbose=3, n_jobs=-1, scoring = scoring)
        grid_search.fit(x, y)
        return(grid_search.best_params_)

#Set of functions for Machine Learning models
class models():
    
    '''
    Define Machine Learning models
    Functions: catboost_model, rf_model
    Will use recall as the evaluation metric because the cost of a false negative is low and we need to capture all
    potential churners
    '''
    def catboost_model(x_train, x_test, y_train, y_test, seed, use_best_mod, **kwargs):
        '''
        Define CatBoost Model
        catboost_model(x_train, x_test, y_train, y_test, seed, use_best_mod, **kwargs) --> model
        '''
        combined = pd.concat([x_train, x_test])
        cat_feat_index = np.where(combined.dtypes == object)[0]
        model = CatBoostClassifier(**kwargs, use_best_model=True,
                                       loss_function='Logloss', eval_metric='Recall', random_seed=seed)
            
        model.fit(x_train, y_train, plot=True, cat_features=cat_feat_index, eval_set=(x_test, y_test))
        
        return(model)
    
    def rf_model(x_cat_train, y_cat_train, seed, **kwargs):
        '''
        Define Random Forest Model
        rf_model(x_cat_train, y_cat_train, seed, **kwargs) --> model
        '''
        rf_mod = RandomForestClassifier(random_state=seed, **kwargs)
        rf_mod.fit(x_cat_train, y_cat_train)
        
        return(rf_mod)
    
    def xgboost_model(x_cat_train, y_cat_train, seed, **kwargs):
        '''
        Define XGBoost Model
        xgb_model(x_cat_train, y_cat_train, seed, **kwargs) -->
        '''
        xgb = XGBClassifier(random_state=seed, **kwargs)
        xgb.fit(x_cat_train, y_cat_train)
        return(xgb)

#Set of functions to evaluate model performance
class evaluation():
    '''
    A set of functions to evaluate model performance
    Functions: predictions, scores, results
    '''
    def predictions(model, x_train, x_test):
        '''
        Use model to predict y test and train
        predictions(mode, x_train, x_test) --> y_pred_train, y_pred_test
        '''
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)

        return(y_pred_train, y_pred_test)

    def scores(y, y_pred):
        '''
        Get model metrics
        scores(y, y_pred) --> AUC, accuracy, F1, recall, precision
        '''
        AUC = roc_auc_score(y, y_pred)
        accuracy = accuracy_score(y, y_pred)
        F1 = f1_score(y, y_pred)
        recall = recall_score(y, y_pred)
        precision = precision_score(y, y_pred)
        return(AUC, accuracy, F1, recall, precision)
    
    def results(y_train, y_test, y_pred_train, y_pred_test):
        '''
        Create train and test model metrics table
        results(y_train, y_test, y_pred_train, y_pred_test) --> final_results table
        '''
        AUC_test, accuracy_test, F1_test, recall_test, precision_test = evaluation.scores(y_test, y_pred_test)
        AUC_train, accuracy_train, F1_train, recall_train, precision_train = evaluation.scores(y_train, y_pred_train)
        final_results = {'Test': [AUC_test, accuracy_test, F1_test, recall_test, precision_test],
                        'Train': [AUC_train, accuracy_train, F1_train, recall_train, precision_train],
                        'Index Title': ['AUC','Accuracy', 'F1', 'Recall', 'Precision']}
        final_results = pd.DataFrame(final_results)
        final_results.index = final_results['Index Title']
        final_results.index.name = 'Metrics'
        del final_results['Index Title']
        return(final_results)    


# # Data Cleaning and Preparation

# In[3]:


#Read in data
df = pd.read_csv('/Users/gabbynyirjesy/Desktop/GitHub/Churn-Modeling/WA_Fn-UseC_-Telco-Customer-Churn.csv')
#Print an overview of the data set
df.head()


# In[4]:


#Explore columns and data types
df.info()


# In[5]:


#Print list of categorical columns
cat_cols = list(df.select_dtypes(exclude=[np.number]).columns.values)
cat_cols


# In[6]:


#Ensure each customerID is included only once in the data
len(df) == len(df['customerID'].unique())


# In[7]:


#Print list of numeric columns
numeric_cols = list(df.select_dtypes(include=[np.number]).columns.values)
numeric_cols


# In[8]:


#Identify any uniform features. 
#If any features are returned, remove them from the data as they will not add any predictiveness
uniform_cols = clean.identify_uniform(df)


# In[9]:


#Identify categorical features with more than 100 unique values.
#Consider binning, deleting, or converting to a continuous numeric variable
df = clean.identify_diverse_cat(df, 100, delete=False)


# In[10]:


#As seen above, the customerID column is very diverse, and will not add predictiveness to the model.
#Therefore, we will drop the customerID column
df_new = df.drop(['customerID'], axis=1)


# In[11]:


#The TotalCharges feature seems as though it should actually be a numeric variable
#We will look into this more by viewing the unique values of the 'TotalCharges' column
df_new['TotalCharges'].unique()


# In[12]:


df_new['TotalCharges'] = df_new['TotalCharges'].apply(clean.convert_to_float_try_except)


# In[13]:


df_new['TotalCharges'].dtype


# In[14]:


#Examine data
df_new.head()


# From examining data, it is clear that there are several columns containing only "Yes/No" values that can 
# be converted to "1/0". Use the to_binary function to convert those columns below.

# In[15]:


df_new = clean.to_binary(df_new)


# In[16]:


#As seen below, the "Yes/No" columns have been converted to binary
df_new.head()


# In[17]:


#Check to see if there are any N/A's remaining in the data (may need to impute)
df_new.isna().values.any()


# In[18]:


#Check to see if there are any null values remaining in the data (may need to impute)
df_new.isnull().values.any()


# In[19]:


#Print numeric columns
numeric_cols = list(df.select_dtypes(include=[np.number]).columns.values)
numeric_cols


# In[20]:


#Investigate distribution of values for numeric columns
for col in numeric_cols:
    plt.title(col)
    plt.hist(df_new[col])
    plt.show()


# From the distributions above, there do not seem to be any outliers in the distributions.

# In[21]:


#Explore the percentage of churners vs. non-churners in the data
df_new['Churn'].value_counts(normalize=True)


# There is a higher percentage of non-churners than churners in the dataset, which may skew the model. To correct for this, we will determine the class imbalance and re-sample the data within the respective models later in the code.

# In[22]:


#Set seed for reproducability
seed = 111


# In[23]:


#Define the binary target variable for classification model
binary_var = 'Churn'


# In[24]:


#Separate data into x and y
x = df_new[df_new.columns.difference([binary_var])]
y = df_new[binary_var]


# # Model Creation

# ## Random Forest Model

# In[25]:


#Label encode the categorical columns within the x data since categorical variables are not accepted in
#Random Forest models
x_cat = clean.label_encoding(x)


# In[26]:


#Ensure all columns are numeric
x_cat.info()


# In[27]:


#Split the encoded data into train and test sets
x_cat_train, x_cat_test, y_cat_train, y_cat_test = train_test_split(x_cat, y, stratify=y, 
                                                                    test_size=0.2, random_state=seed)


# In[28]:


#Define random_grid for parameter tuning random search
random_grid = {'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
                          'max_features': ['auto', 'sqrt'],
                          'max_depth': [int(x) for x in np.linspace(100, 110, num=2)],
                          'min_samples_split': [2, 5, 10],
                          'min_samples_leaf': [1, 2, 5],
                          'bootstrap': [True, False],
                          'class_weight': ['balanced']}


# In[29]:


#Run a random search of Random Forest parameters to find best initial hyper parameters. Use stratified k-fold
#cross-validation in randomsearch
params = parameter_tuning.randomsearch(seed, x_cat, y, random_grid, 'rf')


# In[30]:


print(params)


# In[31]:


#Define random grid for grid search parameter tuning
random_grid = {'n_estimators': [600, 700],
                          'max_features': ['auto','sqrt'],
                          'max_depth': [110],
                          'min_samples_split': [4, 5, 6],
                          'min_samples_leaf': [5, 6],
                          'bootstrap': [False],
                          'class_weight': ['balanced']}


# In[32]:


#Run a more intensive grid search to tune hyper parameters for Random Forest Model.  Use stratified k-fold
#cross-validation in randomsearch
params = parameter_tuning.gridsearch(seed, x_cat, y, random_grid, 'rf')


# In[33]:


print(params)


# In[34]:


#Run the preliminary Random Forest Model
rf_mod = models.rf_model(x_cat_train, y_cat_train, seed, **params)


# In[35]:


#Explore feature importance
rf_mod_imp = pd.DataFrame({'Gini-importance': rf_mod.feature_importances_, 'col': x_cat_train.columns})
rf_mod_imp = rf_mod_imp.sort_values(['Gini-importance', 'col'], ascending=False)
#Keep features with a Gini-importance >= 0.01
rf_mod_imp = rf_mod_imp[rf_mod_imp['Gini-importance'] >= 0.01]
print(rf_mod_imp)


# In[36]:


#Keep top importance features in the x data
x_rf_imp = x_cat[rf_mod_imp['col']]
x_train_rf_imp = x_cat_train[rf_mod_imp['col']]
x_test_rf_imp = x_cat_test[rf_mod_imp['col']]


# In[37]:


#Run Random Forest model with only important features
rf_mod = models.rf_model(x_train_rf_imp, y_cat_train, seed, **params)


# In[38]:


#Use Random Forest model to make predictions on train and test sets
y_pred_train_rf, y_pred_test_rf = evaluation.predictions(rf_mod, x_train_rf_imp, x_test_rf_imp)


# In[39]:


#Examine model performance metrics
rf_results = evaluation.results(y_cat_train, y_cat_test, y_pred_train_rf, y_pred_test_rf)


# In[40]:


print(rf_results)


# Random Forest seems to be slightly overfitting. Experiment with XGBoost model to see if there is better performance.

# ## XGBoost Model

# In[41]:


#Define scale_pos weight term to compensate for class imbalance in XGBoost model
scale_pos = y_cat_train.value_counts()[0]/y_cat_train.value_counts()[1]


# In[42]:


#Create random grid for random search parameter tuning
random_grid = {'min_child_weight': [1, 5, 10],
                          'gamma': [0.5, 1, 1.5, 2, 5],
                          'subsample': [0.6, 0.8, 1.0],
                          'colsample_bytree': [0.6, 0.8, 1.0],
                          'max_depth': [3, 4, 5],
                          'scale_pos_weight': [scale_pos]}


# In[43]:


#Run a random search of XGboost parameters to find best initial hyper parameters
params = parameter_tuning.randomsearch(seed, x_cat, y, random_grid, 'xgboost')


# In[44]:


print(params)


# In[45]:


#Define random grid for grid search parameter tuning
random_grid = {'min_child_weight': [1, 2],
                          'gamma': [0.5, 1, 1.5],
                          'subsample': [0.9, 1],
                          'colsample_bytree': [0.5, 0.6, 0.7],
                          'max_depth': [3, 4, 5],
                          'scale_pos_weight': [scale_pos]}


# In[46]:


#Run a more intensive grid search to tune hyper parameters for XGBoost Model.  Use stratified k-fold
#cross-validation in randomsearch
params = parameter_tuning.gridsearch(seed, x_cat, y, random_grid, 'xgboost')


# In[47]:


print(params)


# In[48]:


#Run the preliminary XGboost Model
xgb_mod = models.xgboost_model(x_cat_train, y_cat_train, seed, **params)


# In[49]:


print(params)


# In[50]:


#Explore feature importance
xgb_mod_imp = pd.DataFrame({'Gini-importance': xgb_mod.feature_importances_, 'col': x_cat_train.columns})
xgb_mod_imp = xgb_mod_imp.sort_values(['Gini-importance', 'col'], ascending=False)
#Keep features with a Gini-importance >= 0.01
xgb_mod_imp = xgb_mod_imp[xgb_mod_imp['Gini-importance'] >= 0.01]
print(xgb_mod_imp)


# In[51]:


#Keep top importance features in the x data
x_xgb_imp = x_cat[xgb_mod_imp['col']]
x_train_xgb_imp = x_cat_train[xgb_mod_imp['col']]
x_test_xgb_imp = x_cat_test[xgb_mod_imp['col']]


# In[52]:


#Run XGboost Model
xgb_mod = models.xgboost_model(x_train_xgb_imp, y_cat_train, seed, **params)


# In[53]:


#Make predictions on train and test sets
y_pred_train_xgb, y_pred_test_xgb = evaluation.predictions(xgb_mod, x_train_xgb_imp, x_test_xgb_imp)


# In[54]:


#Examine model metric esults
xgb_results = evaluation.results(y_cat_train, y_cat_test, y_pred_train_xgb, y_pred_test_xgb)
print(xgb_results)


# ## CatBoost Model

# In[55]:


#Split data into test, train, and validation sets
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.15, random_state=seed)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=0.15, random_state=seed)


# In[56]:


#Define scale_pos weight term to compensate for class imbalance in CatBoost model
scale_pos = y_train.value_counts()[0]/y_train.value_counts()[1]


# In[57]:


#Create random grid for random search parameter tuning
random_grid = {'learning_rate': [0.03, 0.1],
                          'l2_leaf_reg': [1, 3, 5, 7, 9],
                          'depth': [4, 6, 10],
                          'scale_pos_weight': [scale_pos]}


# In[58]:


#Run a random search of XGboost parameters to find best initial hyper parameters
params = parameter_tuning.randomsearch(seed, x_cat, y, random_grid, 'catboost')


# In[59]:


print(params)


# In[60]:


#Create random grid for grid search parameter tuning
random_grid = {'learning_rate': [0.03, 0.05],
                          'l2_leaf_reg': [8, 9, 10],
                          'depth': [4, 5, 6],
                          'scale_pos_weight': [scale_pos]}


# In[61]:


#Run a more intensive grid search to tune hyper parameters for CatBoost Model.  Use stratified k-fold
#cross-validation in randomsearch
params = parameter_tuning.gridsearch(seed, x_cat, y, random_grid, 'catboost')


# In[62]:


print(params)


# In[63]:


#Run preliminary catboost model
catboost_mod = models.catboost_model(x_train, x_val, y_train, y_val, seed, use_best_mod=True, **params)


# In[65]:


#Explore feature importances
feature_importances = pd.DataFrame({'imp': catboost_mod.feature_importances_, 'col': x_train.columns})
#Keep only top 10 most important features for model
feature_importances = feature_importances.sort_values(['imp', 'col'], ascending=False).iloc[:10]
print(feature_importances)


# In[66]:


#Keep top 10 most important features in x data sets to remove noise in the model
x_train_catboost_imp = x_train[feature_importances['col']]
x_val_catboost_imp = x_val[feature_importances['col']]
x_test_catboost_imp = x_test[feature_importances['col']]


# In[67]:


#Re-run catboost model with most important features only
catboost_mod = models.catboost_model(x_train_catboost_imp, x_val_catboost_imp, 
                                     y_train, y_val, seed, use_best_mod=False, **params)


# In[68]:


#Predict y train and test sets using CatBoost model
y_pred_train, y_pred_test = evaluation.predictions(catboost_mod, x_train_catboost_imp, x_test_catboost_imp)


# In[69]:


#Evaluate catboost model performance
catboost_results = evaluation.results(y_train, y_test, y_pred_train, y_pred_test)
print(catboost_results)


# As seen above, the CatBoost model should be selected for this data set because the model has good recall and does not seem to be overfitting. Recall will capture the most positive instances and the cost of contacting customers who are identified as churners but actually were not (false positives) is low, so this is the metric of choice.
