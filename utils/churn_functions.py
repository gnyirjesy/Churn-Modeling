#Define functions
#Set of functions for data cleaning
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import make_scorer, recall_score, roc_auc_score, accuracy_score,\
    f1_score, precision_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

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
                                           n_iter=100, cv=strat_kfold.split(x, y), verbose=0, random_state=seed,
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
                                       loss_function='Logloss', eval_metric='Recall', random_seed=seed, \
                                           logging_level = 'Silent')
            
        model.fit(x_train, y_train, plot=True, cat_features=cat_feat_index, eval_set=(x_test, y_test),\
            early_stopping_rounds=20)
        
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
