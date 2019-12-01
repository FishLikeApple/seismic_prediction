"""
This is a model from a kaggle master. All the code of this file is a whole and not modular. 
"""

from __future__ import unicode_literals
import numpy as np
import pandas as pd
import os

from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error
pd.options.display.precision = 15

import lightgbm as lgb
import xgboost as xgb
import time
import datetime
from catboost import CatBoostRegressor

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
import gc
#import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from scipy import stats
from sklearn.kernel_ridge import KernelRidge
import pickle

import data_processing
import seismic_prediction as context
import genetic_program
from gplearn import genetic
import threading

#import required packages
import catboost as cb
import gc
import hyperopt
from hyperopt import hp, tpe, Trials, STATUS_OK
from hyperopt.fmin import fmin
from hyperopt.pyll.stochastic import sample
from joblib import Parallel, delayed # needed for gp
#optional but advised
import warnings
warnings.filterwarnings('ignore')
import NN

#GLOBAL HYPEROPT PARAMETERS
NUM_EVALS = 1000 #number of hyperopt evaluation rounds
N_FOLDS = 8 #number of cross-validation folds on data in each evaluation round

#LIGHTGBM PARAMETERS
LGBM_MAX_LEAVES = 2**11 #maximum number of leaves per tree for LightGBM
LGBM_MAX_DEPTH = 25 #maximum tree depth for LightGBM
EVAL_METRIC_LGBM_REG = 'mae' #LightGBM regression metric. Note that 'rmse' is more commonly used 
EVAL_METRIC_LGBM_CLASS = 'auc'#LightGBM classification metric

#XGBOOST PARAMETERS
XGB_MAX_LEAVES = 2**12 #maximum number of leaves when using histogram splitting
XGB_MAX_DEPTH = 25 #maximum tree depth for XGBoost
EVAL_METRIC_XGB_REG = 'mae' #XGBoost regression metric
EVAL_METRIC_XGB_CLASS = 'auc' #XGBoost classification metric

#CATBOOST PARAMETERS
CB_MAX_DEPTH = 8 #maximum tree depth in CatBoost
OBJECTIVE_CB_REG = 'MAE' #CatBoost regression metric
OBJECTIVE_CB_CLASS = 'Logloss' #CatBoost classification metric

#OPTIONAL OUTPUT
BEST_SCORE = 0

def quick_hyperopt(data, labels, package='lgbm', num_evals=NUM_EVALS, diagnostic=False):
    
    #==========
    #LightGBM
    #==========
    
    if package=='lgbm':
        
        print('Running {} rounds of LightGBM parameter optimisation:'.format(num_evals))
        #clear space
        gc.collect()
        
        integer_params = ['max_depth',
                         'num_leaves',
                          'max_bin',
                         'min_data_in_leaf',
                         'min_data_in_bin']
        
        def objective(space_params):
            
            #cast integer params from float to int
            for param in integer_params:
                space_params[param] = int(space_params[param])
            
            #extract nested conditional parameters
            if space_params['boosting']['boosting'] == 'goss':
                top_rate = space_params['boosting'].get('top_rate')
                other_rate = space_params['boosting'].get('other_rate')
                #0 <= top_rate + other_rate <= 1
                top_rate = max(top_rate, 0)
                top_rate = min(top_rate, 0.5)
                other_rate = max(other_rate, 0)
                other_rate = min(other_rate, 0.5)
                space_params['top_rate'] = top_rate
                space_params['other_rate'] = other_rate
            
            subsample = space_params['boosting'].get('subsample', 1.0)
            space_params['boosting'] = space_params['boosting']['boosting']
            space_params['subsample'] = subsample
            
            #for classification, set stratified=True and metrics=EVAL_METRIC_LGBM_CLASS
            cv_results = lgb.cv(space_params, train, nfold = N_FOLDS, stratified=False,
                                early_stopping_rounds=100, metrics=EVAL_METRIC_LGBM_REG, seed=42)
            
            best_loss = cv_results['l1-mean'][-1] #'l2-mean' for rmse
            #for classification, comment out the line above and uncomment the line below:
            #best_loss = 1 - cv_results['auc-mean'][-1]
            #if necessary, replace 'auc-mean' with '[your-preferred-metric]-mean'
            return{'loss':best_loss, 'status': STATUS_OK }
        
        train = lgb.Dataset(data, labels)
                
        #integer and string parameters, used with hp.choice()
        boosting_list = [{'boosting': 'gbdt',
                          'subsample': hp.uniform('subsample', 0.5, 1)},
                         {'boosting': 'goss',
                          'subsample': 1.0,
                         'top_rate': hp.uniform('top_rate', 0, 0.5),
                         'other_rate': hp.uniform('other_rate', 0, 0.5)}] #if including 'dart', make sure to set 'n_estimators'
        metric_list = ['MAE', 'RMSE'] 
        #for classification comment out the line above and uncomment the line below
        #metric_list = ['auc'] #modify as required for other classification metrics
        objective_list_reg = ['huber', 'gamma', 'fair', 'tweedie']
        objective_list_class = ['binary', 'cross_entropy']
        #for classification set objective_list = objective_list_class
        objective_list = objective_list_reg

        space ={'boosting' : hp.choice('boosting', boosting_list),
                'num_leaves' : hp.quniform('num_leaves', 2, LGBM_MAX_LEAVES, 1),
                'max_depth': hp.quniform('max_depth', 2, LGBM_MAX_DEPTH, 1),
                'max_bin': hp.quniform('max_bin', 32, 255, 1),
                'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 256, 1),
                'min_data_in_bin': hp.quniform('min_data_in_bin', 1, 256, 1),
                'min_gain_to_split' : hp.quniform('min_gain_to_split', 0.1, 5, 0.01),
                'lambda_l1' : hp.uniform('lambda_l1', 0, 5),
                'lambda_l2' : hp.uniform('lambda_l2', 0, 5),
                'learning_rate' : hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
                'metric' : hp.choice('metric', metric_list),
                'objective' : hp.choice('objective', objective_list),
                'feature_fraction' : hp.quniform('feature_fraction', 0.5, 1, 0.01),
                'bagging_fraction' : hp.quniform('bagging_fraction', 0.5, 1, 0.01)
            }
        
        #optional: activate GPU for LightGBM
        #follow compilation steps here:
        #https://www.kaggle.com/vinhnguyen/gpu-acceleration-for-lightgbm/
        #then uncomment lines below:
        #space['device'] = 'gpu'
        #space['gpu_platform_id'] = 0,
        #space['gpu_device_id'] =  0

        trials = Trials()
        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=num_evals, 
                    trials=trials)
                
        #fmin() will return the index of values chosen from the lists/arrays in 'space'
        #to obtain actual values, index values are used to subset the original lists/arrays
        #extract nested conditional parameters
        if best['boosting']['boosting'] == 'goss':
            top_rate = best['boosting'].get('top_rate')
            other_rate = best['boosting'].get('other_rate')
            #0 <= top_rate + other_rate <= 1
            top_rate = max(top_rate, 0)
            top_rate = min(top_rate, 0.5)
            other_rate = max(other_rate, 0)
            other_rate = min(other_rate, 0.5)
            best['top_rate'] = top_rate
            best['other_rate'] = other_rate
        best['boosting'] = boosting_list[best['boosting']]['boosting']#nested dict, index twice
        best['metric'] = metric_list[best['metric']]
        best['objective'] = objective_list[best['objective']]
                
        #cast floats of integer params to int
        for param in integer_params:
            best[param] = int(best[param])
        
        print('{' + '\n'.join('{}: {}'.format(k, v) for k, v in best.items()) + '}')
        if diagnostic:
            return(best, trials)
        else:
            return(best)
    
    #==========
    #XGBoost
    #==========
    
    if package=='xgb':
        
        print('Running {} rounds of XGBoost parameter optimisation:'.format(num_evals))
        #clear space
        gc.collect()
        
        integer_params = ['max_depth']
        
        def objective(space_params):
            
            for param in integer_params:
                space_params[param] = int(space_params[param])
                
            #extract multiple nested tree_method conditional parameters
            #libera te tutemet ex inferis
            if space_params['tree_method']['tree_method'] == 'hist':
                max_bin = space_params['tree_method'].get('max_bin')
                space_params['max_bin'] = int(max_bin)
                if space_params['tree_method']['grow_policy']['grow_policy']['grow_policy'] == 'depthwise':
                    grow_policy = space_params['tree_method'].get('grow_policy').get('grow_policy').get('grow_policy')
                    space_params['grow_policy'] = grow_policy
                    space_params['tree_method'] = 'hist'
                else:
                    max_leaves = space_params['tree_method']['grow_policy']['grow_policy'].get('max_leaves')
                    space_params['grow_policy'] = 'lossguide'
                    space_params['max_leaves'] = int(max_leaves)
                    space_params['tree_method'] = 'hist'
            else:
                space_params['tree_method'] = space_params['tree_method'].get('tree_method')

            space_params['n_jobs'] = -1
            space_params['learning_rate'] = 0.01    

            print(space_params)
            #for classification replace EVAL_METRIC_XGB_REG with EVAL_METRIC_XGB_CLASS
            cv_results = xgb.cv(space_params, train, nfold=N_FOLDS, metrics=[EVAL_METRIC_XGB_REG],
                             early_stopping_rounds=100, stratified=False, seed=42)
            
            best_loss = cv_results['test-mae-mean'].iloc[-1] #or 'test-rmse-mean' if using RMSE
            #for classification, comment out the line above and uncomment the line below:
            #best_loss = 1 - cv_results['test-auc-mean'].iloc[-1]
            #if necessary, replace 'test-auc-mean' with 'test-[your-preferred-metric]-mean'
            return{'loss':best_loss, 'status': STATUS_OK }
        
        train = xgb.DMatrix(data, labels)
        
        #integer and string parameters, used with hp.choice()
        boosting_list = ['gbtree', 'gblinear'] #if including 'dart', make sure to set 'n_estimators'
        metric_list = ['MAE', 'RMSE'] 
        #for classification comment out the line above and uncomment the line below
        #metric_list = ['auc']
        #modify as required for other classification metrics classification
        
        tree_method = [{'tree_method' : 'exact'},
               {'tree_method' : 'approx'},
               {'tree_method' : 'hist',
                'max_bin': hp.quniform('max_bin', 2**3, 2**7, 1),
                'grow_policy' : {'grow_policy': {'grow_policy':'depthwise'},
                                'grow_policy' : {'grow_policy':'lossguide',
                                                  'max_leaves': hp.quniform('max_leaves', 32, XGB_MAX_LEAVES, 1)}}}]
        
        #if using GPU, replace 'exact' with 'gpu_exact' and 'hist' with
        #'gpu_hist' in the nested dictionary above
        
        objective_list_reg = ['reg:linear', 'reg:gamma', 'reg:tweedie']
        objective_list_class = ['reg:logistic', 'binary:logistic']
        #for classification change line below to 'objective_list = objective_list_class'
        objective_list = objective_list_reg
        
        space ={'boosting' : hp.choice('boosting', boosting_list),
                'tree_method' : hp.choice('tree_method', tree_method),
                'max_depth': hp.quniform('max_depth', 2, XGB_MAX_DEPTH, 1),
                'reg_alpha' : hp.uniform('reg_alpha', 0, 5),
                'reg_lambda' : hp.uniform('reg_lambda', 0, 5),
                'min_child_weight' : hp.uniform('min_child_weight', 0, 5),
                'gamma' : hp.uniform('gamma', 0, 5),
                'learning_rate' : hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
                'eval_metric' : hp.choice('eval_metric', metric_list),
                'objective' : hp.choice('objective', objective_list),
                'colsample_bytree' : hp.quniform('colsample_bytree', 0.1, 1, 0.01),
                'colsample_bynode' : hp.quniform('colsample_bynode', 0.1, 1, 0.01),
                'colsample_bylevel' : hp.quniform('colsample_bylevel', 0.1, 1, 0.01),
                'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
                'nthread' : -1
            }
        
        trials = Trials()
        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=num_evals, 
                    trials=trials)
        
        best['tree_method'] = tree_method[best['tree_method']]['tree_method']
        best['boosting'] = boosting_list[best['boosting']]
        best['eval_metric'] = metric_list[best['eval_metric']]
        best['objective'] = objective_list[best['objective']]
        
        #cast floats of integer params to int
        for param in integer_params:
            best[param] = int(best[param])
        if 'max_leaves' in best:
            best['max_leaves'] = int(best['max_leaves'])
        if 'max_bin' in best:
            best['max_bin'] = int(best['max_bin'])
        
        print('{' + '\n'.join('{}: {}'.format(k, v) for k, v in best.items()) + '}')
        
        if diagnostic:
            return(best, trials)
        else:
            return(best)
    
    #==========
    #CatBoost
    #==========
    
    if package=='cb':
        
        print('Running {} rounds of CatBoost parameter optimisation:'.format(num_evals))
        
        #clear memory 
        gc.collect()
            
        integer_params = ['depth',
                          #'one_hot_max_size', #for categorical data
                          'min_data_in_leaf',
                          'max_bin']
        
        def objective(space_params):
            
            #cast integer params from float to int
            for param in integer_params:
                try:
                    space_params[param] = int(space_params[param])
                except:
                    pass
                
            #extract nested conditional parameters
            try:
                if space_params['bootstrap_type']['bootstrap_type'] == 'Bayesian':
                    bagging_temp = space_params['bootstrap_type'].get('bagging_temperature')
                    space_params['bagging_temperature'] = bagging_temp
            except:
                pass
                
            try:
                if space_params['grow_policy']['grow_policy'] == 'LossGuide':
                    max_leaves = space_params['grow_policy'].get('max_leaves')
                    space_params['max_leaves'] = int(max_leaves)
            except:
                pass

            try:
                space_params['bootstrap_type'] = space_params['bootstrap_type']['bootstrap_type']
            except:
                pass
            try:
                space_params['grow_policy'] = space_params['grow_policy']['grow_policy']
            except:
                pass
                    
            #random_strength cannot be < 0
            try:
                space_params['random_strength'] = max(space_params['random_strength'], 0)
            except:
                pass
            #fold_len_multiplier cannot be < 1
            try:
                space_params['fold_len_multiplier'] = max(space_params['fold_len_multiplier'], 1)
            except:
                pass

            #use GPU
            space_params['task_type'] = 'GPU'        
               
            print(space_params)
            #for classification set stratified=True
            cv_results = cb.cv(train, space_params, fold_count=N_FOLDS, 
                               early_stopping_rounds=25, stratified=False, partition_random_seed=42)
            print("cv_results got")
            try:
                best_loss = cv_results['test-MAE-mean'].iloc[-1] #'test-RMSE-mean' for RMSE
            except:
                best_loss = cv_results['test-RMSE-mean'].iloc[-1]
            #for classification, comment out the line above and uncomment the line below:
            #best_loss = cv_results['test-Logloss-mean'].iloc[-1]
            #if necessary, replace 'test-Logloss-mean' with 'test-[your-preferred-metric]-mean'

            # print the result
            with open(context.saving_path+"master2_logs/logs.txt", 'a') as f:
                print("cat_result:", file=f)
                print(best_loss, file=f)
                print("", file=f)
            
            return{'loss':best_loss, 'status': STATUS_OK}
        
        train = cb.Pool(data, labels.astype('float32'))
        
        #integer and string parameters, used with hp.choice()
        bootstrap_type = [{'bootstrap_type':'Poisson'},   
                           {'bootstrap_type':'Bayesian',
                            'bagging_temperature' : hp.loguniform('bagging_temperature', np.log(1), np.log(50))},
                          {'bootstrap_type':'Bernoulli'}] 
        LEB = ['No', 'AnyImprovement', 'Armijo'] #remove 'Armijo' if not using GPU
        score_function = ['Correlation', 'L2', 'NewtonCorrelation', 'NewtonL2']
        grow_policy = [{'grow_policy':'SymmetricTree'},
                       #{'grow_policy':'Depthwise'},  # it may cause an error on GPU
                       {'grow_policy':'Lossguide',
                        'max_leaves': hp.quniform('max_leaves', 16, 48, 1)}]
        eval_metric_list_reg = ['MAE', 'RMSE', 'Poisson']
        eval_metric_list_class = ['Logloss', 'AUC', 'F1']
        #for classification change line below to 'eval_metric_list = eval_metric_list_class'
        eval_metric_list = eval_metric_list_reg
                
        space ={'depth': hp.quniform('depth', 2, CB_MAX_DEPTH, 1),
                'max_bin' : hp.quniform('max_bin', 32, 255, 1), #if using CPU just set this to 254
                'l2_leaf_reg' : hp.uniform('l2_leaf_reg', 0, 5),
                'min_data_in_leaf' : hp.quniform('min_data_in_leaf', 1, 10, 1),
                'random_strength' : hp.loguniform('random_strength', np.log(0.005), np.log(5)),

                #'one_hot_max_size' : hp.quniform('one_hot_max_size', 2, 16, 1), #uncomment if using categorical features
                'bootstrap_type' : hp.choice('bootstrap_type', bootstrap_type),
                'learning_rate' : hp.loguniform('learning_rate', np.log(0.01), np.log(0.25)),
                'eval_metric' : hp.choice('eval_metric', eval_metric_list),

                'objective' : OBJECTIVE_CB_REG,
                'score_function' : hp.choice('score_function', score_function), #crashes kernel - reason unknown
                'leaf_estimation_backtracking' : hp.choice('leaf_estimation_backtracking', LEB),
                
                #'colsample_bylevel' : hp.quniform('colsample_bylevel', 0.1, 1, 0.01),# CPU only
                'fold_len_multiplier' : hp.loguniform('fold_len_multiplier', np.log(1.5), np.log(2.5)),
                'od_type' : 'Iter',
                
                'grow_policy': hp.choice('grow_policy', grow_policy),

                'od_wait' : 25,
                'task_type' : 'GPU',
                'verbose' : 0
            }
        
        #optional: run CatBoost without GPU
        #uncomment line below
        #space['task_type'] = 'CPU'
            
        trials = Trials()
        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=num_evals, 
                    trials=trials)
        
        #unpack nested dicts first
        try:
            if best['bootstrap_type']['bootstrap_type'] == 'Bayesian':
                bagging_temp = best['bootstrap_type'].get('bagging_temperature')
                best['bagging_temperature'] = bagging_temp
        except:
            pass
        try:
            if best['grow_policy']['grow_policy'] == 'LossGuide':
                max_leaves = best['grow_policy'].get('max_leaves')
                best['max_leaves'] = int(max_leaves)
        except:
            pass
        best['bootstrap_type'] = bootstrap_type[best['bootstrap_type']]['bootstrap_type']
        best['grow_policy'] = grow_policy[best['grow_policy']]['grow_policy']
        best['eval_metric'] = eval_metric_list[best['eval_metric']]
        
        #best['score_function'] = score_function[best['score_function']] 
        #best['leaf_estimation_method'] = LEM[best['leaf_estimation_method']] #CPU only
        best['leaf_estimation_backtracking'] = LEB[best['leaf_estimation_backtracking']]        
        
        #cast floats of integer params to int
        for param in integer_params:
            best[param] = int(best[param])
        if 'max_leaves' in best:
            best['max_leaves'] = int(best['max_leaves'])
        
        print('{' + '\n'.join('{}: {}'.format(k, v) for k, v in best.items()) + '}')
        
        if diagnostic:
            return(best, trials)
        else:
            return(best)
    
    else:
        print('Package not recognised. Please use "lgbm" for LightGBM, "xgb" for XGBoost or "cb" for CatBoost.')      
        
def params_preprocessing(space_params, model_type='lgb'):
    """cast best params into available params, note that goss boosting is not included"""

    """
    if model_type == 'lgb':
        #cast integer params from float to int
        for param in integer_params:
            space_params[param] = int(space_params[param])
          
        '''
        #extract nested conditional parameters
        if space_params['boosting'] == 'goss':
            top_rate = space_params['boosting'].get('top_rate')
            other_rate = space_params['boosting'].get('other_rate')
            #0 <= top_rate + other_rate <= 1
            top_rate = max(top_rate, 0)
            top_rate = min(top_rate, 0.5)
            other_rate = max(other_rate, 0)
            other_rate = min(other_rate, 0.5)
            space_params['top_rate'] = top_rate
            space_params['other_rate'] = other_rate
        ''' 

        subsample = space_params['boosting'].get('subsample', 1.0)
        space_params['boosting'] = space_params['boosting']['boosting']
        space_params['subsample'] = subsample

    elif model_type == 'xgb':
        for param in integer_params:
            space_params[param] = int(space_params[param])
                
        #extract multiple nested tree_method conditional parameters
        #libera te tutemet ex inferis
        if space_params['tree_method']['tree_method'] == 'hist':
            max_bin = space_params['tree_method'].get('max_bin')
            space_params['max_bin'] = int(max_bin)
            if space_params['tree_method']['grow_policy']['grow_policy']['grow_policy'] == 'depthwise':
                grow_policy = space_params['tree_method'].get('grow_policy').get('grow_policy').get('grow_policy')
                space_params['grow_policy'] = grow_policy
                space_params['tree_method'] = 'hist'
            else:
                max_leaves = space_params['tree_method']['grow_policy']['grow_policy'].get('max_leaves')
                space_params['grow_policy'] = 'lossguide'
                space_params['max_leaves'] = int(max_leaves)
                space_params['tree_method'] = 'hist'
        else:
            space_params['tree_method'] = space_params['tree_method'].get('tree_method')

        space_params['n_jobs'] = -1
        space_params['learning_rate'] = 0.01

    elif model_type == 'cat':
        #cast integer params from float to int
        for param in integer_params:
            try:
                space_params[param] = int(space_params[param])
            except:
                pass
                
        #extract nested conditional parameters
        try:
            if space_params['bootstrap_type']['bootstrap_type'] == 'Bayesian':
                bagging_temp = space_params['bootstrap_type'].get('bagging_temperature')
                space_params['bagging_temperature'] = bagging_temp
        except:
            pass
                
        try:
            if space_params['grow_policy']['grow_policy'] == 'LossGuide':
                max_leaves = space_params['grow_policy'].get('max_leaves')
                space_params['max_leaves'] = int(max_leaves)
        except:
            pass

        try:
            space_params['bootstrap_type'] = space_params['bootstrap_type']['bootstrap_type']
        except:
            pass
        try:
            space_params['grow_policy'] = space_params['grow_policy']['grow_policy']
        except:
            pass
                    
        #random_strength cannot be < 0
        try:
            space_params['random_strength'] = max(space_params['random_strength'], 0)
        except:
            pass
        #fold_len_multiplier cannot be < 1
        try:
            space_params['fold_len_multiplier'] = max(space_params['fold_len_multiplier'], 1)
        except:
            pass

        #use GPU
        space_params['task_type'] = 'GPU'
    """
    return space_params

feature_csv_path = 'D:/Backup/Documents/Visual Studio 2015/LANL-Earthquake-Prediction/features/'
if context.on_cloud:
    feature_csv_path = "/cos_person/275/1745/eq_files/features/"

def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]

def classic_sta_lta(x, length_sta, length_lta):
    
    sta = np.cumsum(x ** 2)

    # Convert to float
    sta = np.require(sta, dtype=np.float)

    # Copy for LTA
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta

    # Pad zeros
    sta[:length_lta - 1] = 0

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny

    return sta / lta

def calc_change_rate(x):
    change = (np.diff(x) / x[:-1]).values
    change = change[np.nonzero(change)[0]]
    change = change[~np.isnan(change)]
    change = change[change != -np.inf]
    change = change[change != np.inf]
    return np.mean(change)

def feature_filter(features, feature_importance , threshold=30):
    """filter features by importance"""

    returned_features = pd.DataFrame(index=features.index)
    for i in range(len(features.columns)):
        right_lines = feature_importance[feature_importance["feature"]==features.columns[i]]
        if np.mean(right_lines.loc[:,'importance']) > threshold:
            a = features[features.columns[i]]
            returned_features = pd.concat([returned_features, features[features.columns[i]]], axis=1)
    
    return returned_features 

def get_features(preparation_threads=[], random_sample_num=None, sample_multiple=1, 
                 watch_dog=None, filter_features=True):
    """changed by me, note that random_sample_num will not influence the number of random samples 
       if corresponding files already exist."""

    try:
        if random_sample_num != None:
            X_tr = pd.read_csv(feature_csv_path+'ra_X_tr.csv', index_col=0)
            y_tr = pd.read_csv(feature_csv_path+'ra_y_tr.csv', index_col=0)
        else:
            X_tr = pd.read_csv(feature_csv_path+'X_tr.csv', index_col=0)
            y_tr = pd.read_csv(feature_csv_path+'y_tr.csv', index_col=0)
    except:
        GP_outputer = genetic_program.evaluator()
        #train = pd.read_csv(train_csv_path+'train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
        rows = 150000
        #segments = int(np.floor(train.shape[0] / rows))
        if random_sample_num != None:
            segments = random_sample_num
            #batch_generator = data_processing.batch_generator_V2(context.traning_csv_path_name)
        else:
            segments = int(np.floor(629145480 / rows))
        X_tr = pd.DataFrame(index=range(segments), dtype=np.float64)
        y_tr = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])

        for preparation_thread in preparation_threads:
            if preparation_thread != None:
                preparation_thread.join()

        for segment in tqdm_notebook(range(segments)):
            #seg = train.iloc[segment*rows:segment*rows+rows]
            #x = pd.Series(seg['acoustic_data'].values)
            #y = seg['time_to_failure'].values[-1]
            if watch_dog != None:
                watch_dog.interval_feeding(200, "segment=%d"%(segment))

            if random_sample_num != None:
                index = None
            else:
                index = segment*rows
            x, y = data_processing.get_data_from_pieces(context.traning_csv_piece_path, 
                                                        "acoustic_data", "time_to_failure", index)
            x = pd.Series(x)
            y = y[-1]

            y_tr.loc[segment, 'time_to_failure'] = y
            X_tr.loc[segment, 'mean'] = x.mean()
            X_tr.loc[segment, 'std'] = x.std()
            X_tr.loc[segment, 'max'] = x.max()
            X_tr.loc[segment, 'min'] = x.min()
    
            X_tr.loc[segment, 'mean_change_abs'] = np.mean(np.diff(x))
            X_tr.loc[segment, 'mean_change_rate'] = calc_change_rate(x)
            X_tr.loc[segment, 'abs_max'] = np.abs(x).max()
            X_tr.loc[segment, 'abs_min'] = np.abs(x).min()
    
            X_tr.loc[segment, 'std_first_50000'] = x[:50000].std()
            X_tr.loc[segment, 'std_last_50000'] = x[-50000:].std()
            X_tr.loc[segment, 'std_first_10000'] = x[:10000].std()
            X_tr.loc[segment, 'std_last_10000'] = x[-10000:].std()
    
            X_tr.loc[segment, 'avg_first_50000'] = x[:50000].mean()
            X_tr.loc[segment, 'avg_last_50000'] = x[-50000:].mean()
            X_tr.loc[segment, 'avg_first_10000'] = x[:10000].mean()
            X_tr.loc[segment, 'avg_last_10000'] = x[-10000:].mean()
    
            X_tr.loc[segment, 'min_first_50000'] = x[:50000].min()
            X_tr.loc[segment, 'min_last_50000'] = x[-50000:].min()
            X_tr.loc[segment, 'min_first_10000'] = x[:10000].min()
            X_tr.loc[segment, 'min_last_10000'] = x[-10000:].min()
    
            X_tr.loc[segment, 'max_first_50000'] = x[:50000].max()
            X_tr.loc[segment, 'max_last_50000'] = x[-50000:].max()
            X_tr.loc[segment, 'max_first_10000'] = x[:10000].max()
            X_tr.loc[segment, 'max_last_10000'] = x[-10000:].max()
    
            X_tr.loc[segment, 'max_to_min'] = x.max() / np.abs(x.min())
            X_tr.loc[segment, 'max_to_min_diff'] = x.max() - np.abs(x.min())
            X_tr.loc[segment, 'count_big'] = len(x[np.abs(x) > 500])
            X_tr.loc[segment, 'sum'] = x.sum()
    
            X_tr.loc[segment, 'mean_change_rate_first_50000'] = calc_change_rate(x[:50000])
            X_tr.loc[segment, 'mean_change_rate_last_50000'] = calc_change_rate(x[-50000:])
            X_tr.loc[segment, 'mean_change_rate_first_10000'] = calc_change_rate(x[:10000])
            X_tr.loc[segment, 'mean_change_rate_last_10000'] = calc_change_rate(x[-10000:])
    
            X_tr.loc[segment, 'q95'] = np.quantile(x, 0.95)
            X_tr.loc[segment, 'q99'] = np.quantile(x, 0.99)
            X_tr.loc[segment, 'q05'] = np.quantile(x, 0.05)
            X_tr.loc[segment, 'q01'] = np.quantile(x, 0.01)
    
            X_tr.loc[segment, 'abs_q95'] = np.quantile(np.abs(x), 0.95)
            X_tr.loc[segment, 'abs_q99'] = np.quantile(np.abs(x), 0.99)
            X_tr.loc[segment, 'abs_q05'] = np.quantile(np.abs(x), 0.05)
            X_tr.loc[segment, 'abs_q01'] = np.quantile(np.abs(x), 0.01)
    
            X_tr.loc[segment, 'trend'] = add_trend_feature(x)
            X_tr.loc[segment, 'abs_trend'] = add_trend_feature(x, abs_values=True)
            X_tr.loc[segment, 'abs_mean'] = np.abs(x).mean()
            X_tr.loc[segment, 'abs_std'] = np.abs(x).std()
    
            X_tr.loc[segment, 'mad'] = x.mad()
            X_tr.loc[segment, 'kurt'] = x.kurtosis()
            X_tr.loc[segment, 'skew'] = x.skew()
            X_tr.loc[segment, 'med'] = x.median()
    
            X_tr.loc[segment, 'Hilbert_mean'] = np.abs(hilbert(x)).mean()
            X_tr.loc[segment, 'Hann_window_mean'] = (convolve(x, hann(150), mode='same') / sum(hann(150))).mean()
            X_tr.loc[segment, 'classic_sta_lta1_mean'] = classic_sta_lta(x, 500, 10000).mean()
            X_tr.loc[segment, 'classic_sta_lta2_mean'] = classic_sta_lta(x, 5000, 100000).mean()
            X_tr.loc[segment, 'classic_sta_lta3_mean'] = classic_sta_lta(x, 3333, 6666).mean()
            X_tr.loc[segment, 'classic_sta_lta4_mean'] = classic_sta_lta(x, 10000, 25000).mean()
            X_tr.loc[segment, 'classic_sta_lta5_mean'] = classic_sta_lta(x, 50, 1000).mean()
            X_tr.loc[segment, 'classic_sta_lta6_mean'] = classic_sta_lta(x, 100, 5000).mean()
            X_tr.loc[segment, 'classic_sta_lta7_mean'] = classic_sta_lta(x, 333, 666).mean()
            X_tr.loc[segment, 'classic_sta_lta8_mean'] = classic_sta_lta(x, 4000, 10000).mean()
            X_tr.loc[segment, 'Moving_average_700_mean'] = x.rolling(window=700).mean().mean(skipna=True)
            ewma = pd.Series.ewm
            X_tr.loc[segment, 'exp_Moving_average_300_mean'] = (ewma(x, span=300).mean()).mean(skipna=True)
            X_tr.loc[segment, 'exp_Moving_average_3000_mean'] = ewma(x, span=3000).mean().mean(skipna=True)
            X_tr.loc[segment, 'exp_Moving_average_30000_mean'] = ewma(x, span=30000).mean().mean(skipna=True)
            no_of_std = 3
            X_tr.loc[segment, 'MA_700MA_std_mean'] = x.rolling(window=700).std().mean()
            X_tr.loc[segment,'MA_700MA_BB_high_mean'] = (X_tr.loc[segment, 'Moving_average_700_mean'] + no_of_std * X_tr.loc[segment, 'MA_700MA_std_mean']).mean()
            X_tr.loc[segment,'MA_700MA_BB_low_mean'] = (X_tr.loc[segment, 'Moving_average_700_mean'] - no_of_std * X_tr.loc[segment, 'MA_700MA_std_mean']).mean()
            X_tr.loc[segment, 'MA_400MA_std_mean'] = x.rolling(window=400).std().mean()
            X_tr.loc[segment,'MA_400MA_BB_high_mean'] = (X_tr.loc[segment, 'Moving_average_700_mean'] + no_of_std * X_tr.loc[segment, 'MA_400MA_std_mean']).mean()
            X_tr.loc[segment,'MA_400MA_BB_low_mean'] = (X_tr.loc[segment, 'Moving_average_700_mean'] - no_of_std * X_tr.loc[segment, 'MA_400MA_std_mean']).mean()
            X_tr.loc[segment, 'MA_1000MA_std_mean'] = x.rolling(window=1000).std().mean()
            X_tr.drop('Moving_average_700_mean', axis=1, inplace=True)
    
            X_tr.loc[segment, 'iqr'] = np.subtract(*np.percentile(x, [75, 25]))
            X_tr.loc[segment, 'q999'] = np.quantile(x,0.999)
            X_tr.loc[segment, 'q001'] = np.quantile(x,0.001)
            X_tr.loc[segment, 'ave10'] = stats.trim_mean(x, 0.1)

            #new features
            frequency_features = data_processing.get_frequency_feature(x)
            X_tr.loc[segment, 'frequency_feature20'] = frequency_features[20]
            X_tr.loc[segment, 'frequency_feature40'] = frequency_features[40]
            X_tr.loc[segment, 'frequency_feature81'] = frequency_features[81]

            #X_tr.loc[segment, 'GP_output'] = GP_outputer.get_output(x)[0]

            for windows in [10, 100, 1000]:
                x_roll_std = x.rolling(windows).std().dropna().values
                x_roll_mean = x.rolling(windows).mean().dropna().values
        
                X_tr.loc[segment, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()
                X_tr.loc[segment, 'std_roll_std_' + str(windows)] = x_roll_std.std()
                X_tr.loc[segment, 'max_roll_std_' + str(windows)] = x_roll_std.max()
                X_tr.loc[segment, 'min_roll_std_' + str(windows)] = x_roll_std.min()
                X_tr.loc[segment, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
                X_tr.loc[segment, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)
                X_tr.loc[segment, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
                X_tr.loc[segment, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
                X_tr.loc[segment, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
                X_tr.loc[segment, 'av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
                X_tr.loc[segment, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()
        
                X_tr.loc[segment, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
                X_tr.loc[segment, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
                X_tr.loc[segment, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
                X_tr.loc[segment, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
                X_tr.loc[segment, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
                X_tr.loc[segment, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
                X_tr.loc[segment, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
                X_tr.loc[segment, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
                X_tr.loc[segment, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
                X_tr.loc[segment, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
                X_tr.loc[segment, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()
   
        if random_sample_num != None:
            X_tr.to_csv(feature_csv_path+'ra_X_tr.csv')
            y_tr.to_csv(feature_csv_path+'ra_y_tr.csv')
        else:
            X_tr.to_csv(feature_csv_path+'X_tr.csv')
            y_tr.to_csv(feature_csv_path+'y_tr.csv')

    # filter features
    if filter_features:
        feature_importance = pd.read_csv(context.saving_path+'feature_importance.csv')
        X_tr = feature_filter(X_tr, feature_importance)

    #deal with inf
    means_dict = {}
    for col in X_tr.columns:
        if X_tr[col].isnull().any():
            mean_value = X_tr.loc[X_tr[col] != -np.inf, col].mean()
            X_tr.loc[X_tr[col] == -np.inf, col] = mean_value
            X_tr[col] = X_tr[col].fillna(mean_value)
            means_dict[col] = mean_value
    with open(context.saving_path+"means_dict", "wb") as f:
        pickle.dump(means_dict, f)

    scaler = StandardScaler()
    scaler.fit(X_tr)
    with open(context.saving_path+"scaler", "wb") as f:
        pickle.dump(scaler, f)

    X_train_scaled = pd.DataFrame(scaler.transform(X_tr), columns=X_tr.columns)
    #X_train_scaled.to_csv(feature_csv_path+'X_train_scaled.csv')
    submission = pd.read_csv(context.sample_submission_path+'sample_submission.csv', index_col='seg_id')

    try:
        X_test = pd.read_csv(feature_csv_path+'X_test.csv', index_col=0)
    except:
        X_test = pd.DataFrame(columns=X_tr.columns, dtype=np.float64, index=submission.index)

        for i, seg_id in enumerate(tqdm_notebook(X_test.index)):
            seg = pd.read_csv(context.test_csv_path + seg_id + '.csv')
    
            x = pd.Series(seg['acoustic_data'].values)
            X_test.loc[seg_id, 'mean'] = x.mean()
            X_test.loc[seg_id, 'std'] = x.std()
            X_test.loc[seg_id, 'max'] = x.max()
            X_test.loc[seg_id, 'min'] = x.min()
        
            X_test.loc[seg_id, 'mean_change_abs'] = np.mean(np.diff(x))
            X_test.loc[seg_id, 'mean_change_rate'] = calc_change_rate(x)
            X_test.loc[seg_id, 'abs_max'] = np.abs(x).max()
            X_test.loc[seg_id, 'abs_min'] = np.abs(x).min()
    
            X_test.loc[seg_id, 'std_first_50000'] = x[:50000].std()
            X_test.loc[seg_id, 'std_last_50000'] = x[-50000:].std()
            X_test.loc[seg_id, 'std_first_10000'] = x[:10000].std()
            X_test.loc[seg_id, 'std_last_10000'] = x[-10000:].std()
    
            X_test.loc[seg_id, 'avg_first_50000'] = x[:50000].mean()
            X_test.loc[seg_id, 'avg_last_50000'] = x[-50000:].mean()
            X_test.loc[seg_id, 'avg_first_10000'] = x[:10000].mean()
            X_test.loc[seg_id, 'avg_last_10000'] = x[-10000:].mean()
    
            X_test.loc[seg_id, 'min_first_50000'] = x[:50000].min()
            X_test.loc[seg_id, 'min_last_50000'] = x[-50000:].min()
            X_test.loc[seg_id, 'min_first_10000'] = x[:10000].min()
            X_test.loc[seg_id, 'min_last_10000'] = x[-10000:].min()
    
            X_test.loc[seg_id, 'max_first_50000'] = x[:50000].max()
            X_test.loc[seg_id, 'max_last_50000'] = x[-50000:].max()
            X_test.loc[seg_id, 'max_first_10000'] = x[:10000].max()
            X_test.loc[seg_id, 'max_last_10000'] = x[-10000:].max()
    
            X_test.loc[seg_id, 'max_to_min'] = x.max() / np.abs(x.min())
            X_test.loc[seg_id, 'max_to_min_diff'] = x.max() - np.abs(x.min())
            X_test.loc[seg_id, 'count_big'] = len(x[np.abs(x) > 500])
            X_test.loc[seg_id, 'sum'] = x.sum()
    
            X_test.loc[seg_id, 'mean_change_rate_first_50000'] = calc_change_rate(x[:50000])
            X_test.loc[seg_id, 'mean_change_rate_last_50000'] = calc_change_rate(x[-50000:])
            X_test.loc[seg_id, 'mean_change_rate_first_10000'] = calc_change_rate(x[:10000])
            X_test.loc[seg_id, 'mean_change_rate_last_10000'] = calc_change_rate(x[-10000:])
    
            X_test.loc[seg_id, 'q95'] = np.quantile(x,0.95)
            X_test.loc[seg_id, 'q99'] = np.quantile(x,0.99)
            X_test.loc[seg_id, 'q05'] = np.quantile(x,0.05)
            X_test.loc[seg_id, 'q01'] = np.quantile(x,0.01)
    
            X_test.loc[seg_id, 'abs_q95'] = np.quantile(np.abs(x), 0.95)
            X_test.loc[seg_id, 'abs_q99'] = np.quantile(np.abs(x), 0.99)
            X_test.loc[seg_id, 'abs_q05'] = np.quantile(np.abs(x), 0.05)
            X_test.loc[seg_id, 'abs_q01'] = np.quantile(np.abs(x), 0.01)
    
            X_test.loc[seg_id, 'trend'] = add_trend_feature(x)
            X_test.loc[seg_id, 'abs_trend'] = add_trend_feature(x, abs_values=True)
            X_test.loc[seg_id, 'abs_mean'] = np.abs(x).mean()
            X_test.loc[seg_id, 'abs_std'] = np.abs(x).std()
    
            X_test.loc[seg_id, 'mad'] = x.mad()
            X_test.loc[seg_id, 'kurt'] = x.kurtosis()
            X_test.loc[seg_id, 'skew'] = x.skew()
            X_test.loc[seg_id, 'med'] = x.median()
    
            X_test.loc[seg_id, 'Hilbert_mean'] = np.abs(hilbert(x)).mean()
            X_test.loc[seg_id, 'Hann_window_mean'] = (convolve(x, hann(150), mode='same') / sum(hann(150))).mean()
            X_test.loc[seg_id, 'classic_sta_lta1_mean'] = classic_sta_lta(x, 500, 10000).mean()
            X_test.loc[seg_id, 'classic_sta_lta2_mean'] = classic_sta_lta(x, 5000, 100000).mean()
            X_test.loc[seg_id, 'classic_sta_lta3_mean'] = classic_sta_lta(x, 3333, 6666).mean()
            X_test.loc[seg_id, 'classic_sta_lta4_mean'] = classic_sta_lta(x, 10000, 25000).mean()
            X_test.loc[seg_id, 'classic_sta_lta5_mean'] = classic_sta_lta(x, 50, 1000).mean()
            X_test.loc[seg_id, 'classic_sta_lta6_mean'] = classic_sta_lta(x, 100, 5000).mean()
            X_test.loc[seg_id, 'classic_sta_lta7_mean'] = classic_sta_lta(x, 333, 666).mean()
            X_test.loc[seg_id, 'classic_sta_lta8_mean'] = classic_sta_lta(x, 4000, 10000).mean()
            X_test.loc[seg_id, 'Moving_average_700_mean'] = x.rolling(window=700).mean().mean(skipna=True)
            ewma = pd.Series.ewm
            X_test.loc[seg_id, 'exp_Moving_average_300_mean'] = (ewma(x, span=300).mean()).mean(skipna=True)
            X_test.loc[seg_id, 'exp_Moving_average_3000_mean'] = ewma(x, span=3000).mean().mean(skipna=True)
            X_test.loc[seg_id, 'exp_Moving_average_30000_mean'] = ewma(x, span=30000).mean().mean(skipna=True)
            no_of_std = 3
            X_test.loc[seg_id, 'MA_700MA_std_mean'] = x.rolling(window=700).std().mean()
            X_test.loc[seg_id,'MA_700MA_BB_high_mean'] = (X_test.loc[seg_id, 'Moving_average_700_mean'] + no_of_std * X_test.loc[seg_id, 'MA_700MA_std_mean']).mean()
            X_test.loc[seg_id,'MA_700MA_BB_low_mean'] = (X_test.loc[seg_id, 'Moving_average_700_mean'] - no_of_std * X_test.loc[seg_id, 'MA_700MA_std_mean']).mean()
            X_test.loc[seg_id, 'MA_400MA_std_mean'] = x.rolling(window=400).std().mean()
            X_test.loc[seg_id,'MA_400MA_BB_high_mean'] = (X_test.loc[seg_id, 'Moving_average_700_mean'] + no_of_std * X_test.loc[seg_id, 'MA_400MA_std_mean']).mean()
            X_test.loc[seg_id,'MA_400MA_BB_low_mean'] = (X_test.loc[seg_id, 'Moving_average_700_mean'] - no_of_std * X_test.loc[seg_id, 'MA_400MA_std_mean']).mean()
            X_test.loc[seg_id, 'MA_1000MA_std_mean'] = x.rolling(window=1000).std().mean()
            X_test.drop('Moving_average_700_mean', axis=1, inplace=True)
    
            X_test.loc[seg_id, 'iqr'] = np.subtract(*np.percentile(x, [75, 25]))
            X_test.loc[seg_id, 'q999'] = np.quantile(x,0.999)
            X_test.loc[seg_id, 'q001'] = np.quantile(x,0.001)
            X_test.loc[seg_id, 'ave10'] = stats.trim_mean(x, 0.1)

            #new features
            frequency_features = data_processing.get_frequency_feature(x)
            X_test.loc[seg_id, 'frequency_feature20'] = frequency_features[20]
            X_test.loc[seg_id, 'frequency_feature40'] = frequency_features[40]
            X_test.loc[seg_id, 'frequency_feature81'] = frequency_features[81]

            #X_test.loc[seg_id, 'GP_output'] = GP_outputer.get_output(x)[0]
    
            for windows in [10, 100, 1000]:
                x_roll_std = x.rolling(windows).std().dropna().values
                x_roll_mean = x.rolling(windows).mean().dropna().values
        
                X_test.loc[seg_id, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()
                X_test.loc[seg_id, 'std_roll_std_' + str(windows)] = x_roll_std.std()
                X_test.loc[seg_id, 'max_roll_std_' + str(windows)] = x_roll_std.max()
                X_test.loc[seg_id, 'min_roll_std_' + str(windows)] = x_roll_std.min()
                X_test.loc[seg_id, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
                X_test.loc[seg_id, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)
                X_test.loc[seg_id, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
                X_test.loc[seg_id, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
                X_test.loc[seg_id, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
                X_test.loc[seg_id, 'av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
                X_test.loc[seg_id, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()
        
                X_test.loc[seg_id, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
                X_test.loc[seg_id, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
                X_test.loc[seg_id, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
                X_test.loc[seg_id, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
                X_test.loc[seg_id, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
                X_test.loc[seg_id, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
                X_test.loc[seg_id, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
                X_test.loc[seg_id, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
                X_test.loc[seg_id, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
                X_test.loc[seg_id, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
                X_test.loc[seg_id, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()

        X_test.to_csv(feature_csv_path+'X_test.csv')

    if filter_features:
        X_test = feature_filter(X_test, feature_importance)

    for col in X_test.columns:
        if X_test[col].isnull().any():
            X_test.loc[X_test[col] == -np.inf, col] = means_dict[col]
            X_test[col] = X_test[col].fillna(means_dict[col])
                
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # multiply the train samples
    if sample_multiple > 1:
        for i in range(sample_multiple-1):
            X_train_scaled = pd.concat([X_train_scaled, X_train_scaled])
            y_tr = pd.concat([y_tr, y_tr])

    return X_train_scaled, X_test_scaled, y_tr

def sample_filter(X, y, sample_threshold, train_result_DF):
    """filter out outlets"""

    sample_index = []
    train_result = train_result_DF 
    for i in range(len(X)):
        if abs(train_result.iloc[i].values-y.iloc[i].values) < sample_threshold:
            sample_index.append(i)

    print("sample number before filter: "+str(len(X)))
    X_output = X.iloc[sample_index]
    y_output = y.iloc[sample_index]
    print("sample number after filter: "+str(len(X_output)))

    return X_output, y_output

# set global args
X_train_scaled = X_test_scaled = y_tr = 0

def train_model(X, X_test, y, params=None, model_type='lgb', name='lgb', n_fold=1, X_valid=[], y_valid=[],
                plot_feature_importance=False, model=None, sample_threshold=999999, feature_threshold=0, 
                n_estimators=25000, use_in_param_turn=False, output_result_csv=False):

    if sample_threshold != 999999:
        train_result = pd.read_csv(context.saving_path+'master2_output/train_result.csv')
        X, y = sample_filter(X, y, sample_threshold, train_result)        

    if feature_threshold != 0:
        feature_importance = pd.read_csv(context.saving_path+'feature_importance.csv')
        X = feature_filter(X, feature_importance)
        X_test = feature_filter(X_test, feature_importance)

    use_default_valid = False
    if (len(X_valid)==0) or (len(y_valid)==0):
        use_default_valid = True

    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    train_prediction = np.zeros(len(X))
    if n_fold != 1:
        folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)
    scores = []
    feature_importance = pd.DataFrame()
    if n_fold == 1:
        fold_n = 1
        with open(context.saving_path+"logs.txt", 'a') as f:
            print('Fold', fold_n, 'started at', time.ctime(), file=f)

        if use_default_valid:
            X_train = X.sample(frac=1, random_state=42)
            y_train = y.sample(frac=1, random_state=42)
        
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators = n_estimators, n_jobs = -1)
            model.fit(X_train, y_train,
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
                    verbose=1000, early_stopping_rounds=200)
            
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
            train_pred = model.predict(X, num_iteration=model.best_iteration_)
            
        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=n_estimators, evals=watchlist, early_stopping_rounds=200, 
                              verbose_eval=10, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            train_pred = model.predict(xgb.DMatrix(X, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
        
        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            
            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = mean_absolute_error(y_valid, y_pred_valid)
            
            if use_in_param_turn == False:
                y_pred = model.predict(X_test).reshape(-1,)
                train_pred = model.predict(X).reshape(-1,)
            else:
                y_pred = prediction
                train_pred = train_prediction

        if model_type == 'cat':
            model = CatBoostRegressor(iterations=n_estimators, **params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=1000,
                      early_stopping_rounds=25)
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)
            train_pred = model.predict(X)

        if model_type == 'gp':
            model = genetic.SymbolicRegressor(**params)
            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid)
            train_pred = model.predict(X)
            y_pred = model.predict(X_test)

        if model_type == 'nn':
            y_pred_valid = NN.NN_model_hr(X_train, X_valid, y_train, y_valid, params)
            y_pred = prediction  # not used
            train_pred = prediction # not used

        #save the model
        #joblib.dump(model, context.saving_path+name+"_"+str(fold_n))
        
        #oof[valid_index] = y_pred_valid.reshape(-1,)
        oof = 0  # not used
        scores.append(mean_absolute_error(y_valid, y_pred_valid))

        prediction += y_pred 
        train_prediction += train_pred
        
        if model_type == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    else:
        if type(n_estimators) == type(1):
            n_estimators = np.ones(n_fold, int) * n_estimators
        for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
            print('Fold', fold_n, 'started at', time.ctime())
            X_train = X.iloc[train_index] 
            y_train = y.iloc[train_index]

            if use_default_valid:
                y_valid = y.iloc[valid_index]
                X_valid = X.iloc[valid_index]
        
            if model_type == 'lgb':
                model = lgb.LGBMRegressor(**params, n_estimators=n_estimators[fold_n], n_jobs=-1)
                model.fit(X_train, y_train,
                        eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
                        verbose=1000, early_stopping_rounds=200)
            
                y_pred_valid = model.predict(X_valid)
                y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
                train_pred = model.predict(X, num_iteration=model.best_iteration_)
            
            if model_type == 'xgb':
                train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
                valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

                watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
                
                model = xgb.train(dtrain=train_data, num_boost_round=n_estimators[fold_n], evals=watchlist,
                                  early_stopping_rounds=200, verbose_eval=10, params=params)
                y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
                y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
                train_pred = model.predict(xgb.DMatrix(X, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
        
            if model_type == 'sklearn':
                model = model
                model.fit(X_train, y_train)
            
                y_pred_valid = model.predict(X_valid).reshape(-1,)
                score = mean_absolute_error(y_valid, y_pred_valid)
            
                y_pred = model.predict(X_test).reshape(-1,)
                train_pred = model.predict(X).reshape(-1,)
        
            if model_type == 'cat':
                model = CatBoostRegressor(iterations=n_estimators[fold_n], **params)
                model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=1000,
                          early_stopping_rounds=25)
                y_pred_valid = model.predict(X_valid)
                y_pred = model.predict(X_test)
                train_pred = model.predict(X)

            if model_type == 'gp':
                threading.Thread()
                model = genetic.SymbolicRegressor(**params)
                model.fit(X_train, y_train)

                y_pred_valid = model.predict(X_valid)
                train_pred = model.predict(X)
                y_pred = model.predict(X_test)

            if model_type == 'nn':
                y_pred_valid = NN.NN_model_hr(X_train, X_valid, y_train, y_valid, params)
                y_pred = prediction  # not used
                train_pred = prediction # not used

            #save the model
            #joblib.dump(model, context.saving_path+name+"_"+str(fold_n))
        
            #oof[valid_index] = y_pred_valid.reshape(-1,)  # not used
            scores.append(mean_absolute_error(y_valid, y_pred_valid))

            prediction += y_pred
            train_prediction += train_pred
        
            if model_type == 'lgb':
                # feature importance
                fold_importance = pd.DataFrame()
                fold_importance["feature"] = X.columns
                fold_importance["importance"] = model.feature_importances_
                fold_importance["fold"] = fold_n + 1
                feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
                if feature_threshold == 0:
                    feature_importance.to_csv(context.saving_path+'master2_output/fold_imp_lgb_1_80k_108dp.csv', 
                                              line_terminator=os.linesep)  # index needed, it is seg id

    prediction /= n_fold
    train_prediction /= n_fold

    if output_result_csv:
        train_result = pd.DataFrame(columns=['time_to_failure'])
        train_result.time_to_failure = train_prediction
        train_result.to_csv(context.saving_path+'master2_output/train_result.csv', 
                            index=False, line_terminator=os.linesep)


    with open(context.saving_path+"logs.txt", 'a') as f:
        print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)), file=f)

    if model_type == 'lgb':
        feature_importance["importance"] /= n_fold
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            #sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            feature_importance.to_csv(context.saving_path+'feature_importance.csv', line_terminator=os.linesep)
        
            return oof, prediction, np.mean(scores)
        return oof, prediction, np.mean(scores)
    
    else:
        return oof, prediction, np.mean(scores)

def model1(params):
    """used for hyperopt"""

    global X_train_scaled, X_test_scaled, y_tr

    n_fold = params['n_fold'] + 5
    lgb_params = {'num_leaves': params['num_leaves']+50, #54,
                  'min_data_in_leaf': params['min_data_in_leaf']+30, #79,
                  'objective': params['objective'],
                  'max_depth': -1,
                  'learning_rate': 1/params['learning_rate'], #0.01,
                  "boosting": "gbdt",
                  "bagging_freq": params["bagging_freq"]+1, #5,
                  "bagging_fraction": params["bagging_fraction"], #0.8126672064208567,
                  "bagging_seed": params["bagging_seed"]+3, #11,
                  "metric": 'mae',
                  "verbosity": -1,
                  'reg_alpha': params['reg_alpha'], #0.1302650970728192,
                  'reg_lambda': params['reg_lambda'], #0.3603427518866501,
                  'feature_fraction': params['feature_fraction']} #0.1}
    _, _, _, error = train_model(X=X_train_scaled, X_test=X_test_scaled, y=y_tr,
                                    params=lgb_params, name='lgb1', model_type='lgb', 
                                    n_fold=n_fold, plot_feature_importance=True)

    return error

def model2(params):

    global X_train_scaled, X_test_scaled, y_tr

    n_fold = params['n_fold'] + 5
    xgb_params = {'eta': 1/params['eta'],
              'max_depth': params['max_depth']+5,
              'subsample': params['subsample'],
              'objective': 'reg:linear',
              'eval_metric': 'mae',
              'silent': True,
              'nthread': 4+(4*context.on_cloud)}
    _, _, error = train_model(X=X_train_scaled, X_test=X_test_scaled, y=y_tr, params=xgb_params,
                                 n_fold=n_fold, name='xgb1',model_type='xgb')
    return error

def model3(params):

    global X_train_scaled, X_test_scaled, y_tr
    model = NuSVR(gamma='scale', nu=params['nu'], tol=params['tol'], C=params['C'])
    _, _, error = train_model(X=X_train_scaled, X_test=X_test_scaled, y=y_tr, params=None, 
                                    name='sklearn2', model_type='sklearn', model=model)

    return error

def model4(params):

    global X_train_scaled, X_test_scaled, y_tr

    params = {'loss_function':'MAE',
              'task_type':'GPU',
              'early_stopping_rounds':500,
              'depth':params['depth']+4,
              'l2_leaf_reg':params['l2_leaf_reg'],
              'learning_rate':1/params['learning_rate']}
    _, _, error = train_model(X=X_train_scaled, X_test=X_test_scaled, y=y_tr, params=params, 
                                                  name='cat1', model_type='cat')

    return error

def model5(params):

    global X_train_scaled, X_test_scaled, y_tr

    n_fold = 5
    _, _, error = train_model(X=X_train_scaled, X_test=X_test_scaled, y=y_tr, params=params,
                              model_type='nn')

    return error

class MyThread(threading.Thread):
    """this modified class is from https://blog.csdn.net/zzzzjh/article/details/80614897"""

    def __init__(self,func,args=()):
        super(MyThread,self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.result = self.func(*self.args)
    def get_result(self):
        try:
            return self.result
        except Exception:
            return None

def model6(params):

    global X_train_scaled, X_test_scaled, y_tr

    params['function_set'] = ('add', 'sub', 'mul', 'div', "sqrt", "log", "max", "min", "sin", "cos", "tan")
    
    # these 4 params's sum shouldn't be higher than 1
    sum = params['p_crossover proportion'] + params['p_subtree_mutation proportion']
    sum += params['p_hoist_mutation proportion']
    sum += params['p_point_mutation proportion'] 
    if sum < 1.0:
        sum = 1.0
    params['p_crossover'] = params['p_crossover proportion'] / sum
    params['p_subtree_mutation'] = params['p_subtree_mutation proportion'] / sum
    params['p_hoist_mutation'] = params['p_hoist_mutation proportion'] / sum
    params['p_point_mutation'] = params['p_point_mutation proportion'] / sum
    params.pop('p_crossover proportion')
    params.pop('p_subtree_mutation proportion')
    params.pop('p_hoist_mutation proportion')
    params.pop('p_point_mutation proportion')

    # cast integer params from float to int
    integer_params = ['population_size',
                      'generations',
                      'tournament_size']
    for param in integer_params:
        params[param] = int(params[param])
    print(params)

    # manual multithread
    threads = []
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=11)
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train_scaled)):
        X_train, X_valid = X_train_scaled.iloc[train_index], X_train_scaled.iloc[valid_index]
        y_train, y_valid = y_tr.iloc[train_index], y_tr.iloc[valid_index]
        threads.append(MyThread(train_model,(X_train, X_train[:1], y_train, params, 
                                             'gp', 'gp', 1, X_valid, y_valid)))
        threads[-1].daemon = True
        threads[-1].start()

    errors = 0
    for thread in threads:
        thread.join()
        output = None
        while output == None:
            time.sleep(0.1)
            output = thread.get_result()
        errors += output[2]
    errors /= N_FOLDS

    return errors

X_train_scaled = 0
X_test_scaled = 0
y_tr = 0

def hyperopt_test(preparation_threads, model_num=None):

    global X_train_scaled, y_tr, X_valid, y_valid, X_test_scaled
    OUTPUT_DIR = context.saving_path 
    """
    X_train = pd.read_csv(os.path.join(OUTPUT_DIR, 'scaled_train_X.csv'))
    new_X_train = pd.read_csv(os.path.join(OUTPUT_DIR, 'scaled_new_samples_X.csv'))
    more_X_train = pd.concat([X_train, new_X_train])
    y_train = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_y.csv'))
    new_y_train = pd.read_csv(os.path.join(OUTPUT_DIR, 'new_samples_y.csv'))
    more_y_train = pd.concat([y_train, new_y_train])
    X_valid=X_train.sample(frac=0.2, random_state=11)
    y_valid=X_train.sample(frac=0.2, random_state=11)
    X_test = pd.read_csv(os.path.join(OUTPUT_DIR, 'scaled_test_X.csv'))
    """
    X_train = pd.read_csv(os.path.join(OUTPUT_DIR, 'scaled_no_lap_train_X.csv'))
    X_test = pd.read_csv(os.path.join(OUTPUT_DIR, 'scaled_test_X.csv'))
    y_train = pd.read_csv(os.path.join(OUTPUT_DIR, 'no_lap_train_y.csv'))

    X_train_scaled = X_train
    X_test_scaled = X_test
    y_tr = y_train

    for preparation_thread in preparation_threads:
        if preparation_thread != None:
            preparation_thread.join()
    """
    space = {'nu': hyperopt.hp.uniform('nu', 0.6, 0.8),
             'tol':hyperopt.hp.uniform('tol', 0.005, 0.015),
             'C':  hyperopt.hp.uniform('C', 0.9, 1.1)}
    algo = hyperopt.partial(hyperopt.tpe.suggest, n_startup_jobs=10)
    best = hyperopt.fmin(model4, space, algo, 200)
    print(best)
    """

    #model1
    if (model_num==None) or (model_num==1):
        try:
            best1, trials1 = quick_hyperopt(X_train_scaled, y_tr, 'lgbm', 1500, diagnostic=True)
        except:
            print("model1 error")
            best1 = None
            trials1 = None
        with open(context.saving_path+"logs.txt", 'a') as f:
            print("", file=f)
            print("best_lgbm:", file=f)
            print(best1, file=f)
            print("", file=f)
            print("best_trials1:", file=f)
            print(trials1, file=f)
            print("", file=f)
        np.save(context.saving_path+'lgbm_params.npy', best1)
        with open(context.saving_path+'best_trials1', "wb") as f:
            pickle.dump(trials1, f)
    #{'min_gain_to_split': 5.0, 'max_bin': 224, 'other_rate': 0.46915923708341856, 
    #'top_rate': 0.36289022236006685, 'lambda_l1': 1.027236870543288, 'min_data_in_leaf': 12, 
    #'metric': 'RMSE', 'feature_fraction': 0.55, 'max_depth': 5, 'boosting': 'goss', 
    #'learning_rate': 0.19952347506492585, 'objective': 'huber', 'min_data_in_bin': 95, 
    #'lambda_l2': 4.816194305992925, 'bagging_fraction': 0.64, 'num_leaves': 708}
    #{'max_depth': 20, 'learning_rate': 0.037491213086246415, 'num_leaves': 1352, 'min_data_in_leaf': 1, 'metric': 'RMSE', 'lambda_l2': 3.0472684549095574, 'boosting': 'gbdt', 'bagging_fraction': 0.53, 'min_gain_to_split': 1.01, 'lambda_l1': 0.38083001367253844, 'min_data_in_bin': 251, 'objective': 'gamma', 'feature_fraction': 0.99, 'max_bin': 215, 'subsample': 0.6587682468738089}

    
    #model2
    if (model_num==None) or (model_num==2):
        try:
            best2, trials2 = quick_hyperopt(X_train_scaled, y_tr, 'xgb', 2000, diagnostic=True)
        except:
            print("model2 error")
            best2 = None
            trials2 = None

        with open(context.saving_path+"logs.txt", 'a') as f:
            print("", file=f)
            print("best_xgb:", file=f)
            print(best2, file=f)
            print("", file=f)
            print("trials_xgb:", file=f)
            print(trials2, file=f)
            print("", file=f)
        np.save(context.saving_path+'xgb_params.npy', best2)
        with open(context.saving_path+'best_trials2', "wb") as f:
            pickle.dump(trials2, f)

    #model3
    if (model_num==None) or (model_num==3):

        best3, trials3 = quick_hyperopt(X_train_scaled, y_tr, 'cb', 600,  diagnostic=True)

        with open(context.saving_path+"logs.txt", 'a') as f:
            print("", file=f)
            print("best_cat:", file=f)
            print(best3, file=f)
            print("", file=f)
        np.save(context.saving_path+'cat_params.npy', best3)
        with open(context.saving_path+'best_trials3', "wb") as f:
            pickle.dump(trials3, f)
    """
    
    #modelNN
    from keras import optimizers

    hidden_layer_num = 3  # <8
    space = {} 
    for i in range(hidden_layer_num+1):
        space['Dense'+str(i+1)+'_size'] = hyperopt.hp.choice('Dense'+str(i+1)+'_size', [2**(10-i), 2**(9-i)])
        space['Activation'+str(i+1)] = hyperopt.hp.choice('Activation'+str(i+1), ['linear', 'tanh', 'elu', 'relu'])
        space['BatchNorm'+str(i+1)] = hyperopt.hp.choice('BatchNorm'+str(i+1), [True, False])
        space['Dropout_rate'+str(i+1)] = hyperopt.hp.uniform('Dropout_rate'+str(i+1), 0.0, 0.5)
    space['optimizer'] = hyperopt.hp.choice('optimizer', [optimizers.Adam, optimizers.SGD, optimizers.RMSprop])
    space['learn_rate'] = hyperopt.hp.loguniform('learn_rate', np.log(0.0003), np.log(0.2))

    best4 = hyperopt.fmin(model5, space, hyperopt.tpe.suggest, 60)

    with open(context.saving_path+"logs.txt", 'a') as f:
        print("", file=f)
        print("best4:", file=f)
        print(best4, file=f)
        print("", file=f)
    np.save(context.saving_path+'nn_params.npy', best4)
    """
    #modelgp
    if (model_num=='gp') or  (model_num==None):
        space = {'population_size': hyperopt.hp.quniform('population_size', 2000, 5000, 1),
                 'generations':hyperopt.hp.quniform('generations', 11, 30, 1),
                 'tournament_size': hyperopt.hp.quniform('tournament_size', 15, 40, 1),
                 'p_crossover proportion': hyperopt.hp.uniform('p_crossover proportion', 0.6, 0.95),
                 'p_subtree_mutation proportion' : hyperopt.hp.uniform('p_subtree_mutation proportion', 0.0001, 0.1),
                 'p_hoist_mutation proportion' : hyperopt.hp.uniform('p_hoist_mutation proportion', 0.0001, 0.1),
                 'p_point_mutation proportion' : hyperopt.hp.uniform('p_point_mutation proportion', 0.0001, 0.1),
                 'metric' : hyperopt.hp.choice('metric', ['mse', 'rmse']),
                 'max_samples' : hyperopt.hp.uniform('max_samples', 0.5, 1.0)}

        trials6 = Trials()
        best6 = hyperopt.fmin(model6, space, hyperopt.tpe.suggest, 300, trials=trials6)

        with open(context.saving_path+"logs.txt", 'a') as f:
            print("", file=f)
            print("best_gp:", file=f)
            print(best6, file=f)
            print("", file=f)
        np.save(context.saving_path+'gp_params.npy', best6)
        with open(context.saving_path+'best_trials6', "wb") as f:
            pickle.dump(trials6, f)
    
