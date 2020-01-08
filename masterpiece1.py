'''
changed from 
'''

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
import gc
#import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from scipy import stats
from sklearn.kernel_ridge import KernelRidge
from itertools import product

from tsfresh.feature_extraction import feature_calculators
from joblib import Parallel, delayed
import joblib

import data_processing
import seismic_prediction as context

# Create a training file with simple derived features
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

class FeatureGenerator(object):
    def __init__(self, dtype, n_jobs=1, chunk_size=None):
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.filename = None
        self.n_jobs = n_jobs
        self.test_files = []
        if self.dtype == 'train':
            self.filename = context.traning_csv_path_name
            self.total_data = int(629145481 / self.chunk_size)
        else:
            submission = pd.read_csv(context.sample_submission_path+'sample_submission.csv')
            for seg_id in submission.seg_id.values:
                self.test_files.append((seg_id, context.test_csv_path + seg_id + '.csv'))
            self.total_data = int(len(submission))

    def read_chunks(self):
        if self.dtype == 'train':
            iter_df = pd.read_csv(self.filename, iterator=True, chunksize=self.chunk_size,
                                    dtype={'acoustic_data': np.float64, 'time_to_failure': np.float64})
            for counter, df in enumerate(iter_df):
                x = df.acoustic_data.values
                y = df.time_to_failure.values[-1]
                seg_id = 'train_' + str(counter)
                del df
                yield seg_id, x, y
        else:
            for seg_id, f in self.test_files:
                df = pd.read_csv(f, dtype={'acoustic_data': np.float64})
                x = df.acoustic_data.values[-self.chunk_size:]
                del df
                yield seg_id, x, -999
    
    def get_features(self, x, y, seg_id):

        x = pd.Series(x)

        x_denoised = data_processing.high_pass_filter(x, low_cutoff=10000, SAMPLE_RATE=4000000)
        x_denoised = data_processing.denoise_signal(x_denoised, wavelet='haar', level=1)
        x_denoised = pd.Series(x_denoised)
    
        zc = np.fft.fft(x)
        realFFT = pd.Series(np.real(zc))
        imagFFT = pd.Series(np.imag(zc))
        zc_denoised = np.fft.fft(x_denoised)
        realFFT_denoised = pd.Series(np.real(zc_denoised))
        imagFFT_denoised = pd.Series(np.imag(zc_denoised))
        
        main_dict = self.features(x, y, seg_id)
        r_dict = self.features(realFFT, y, seg_id)
        i_dict = self.features(imagFFT, y, seg_id)
        main_dict_denoised = self.features(x_denoised, y, seg_id)
        r_dict_denoised = self.features(realFFT_denoised, y, seg_id)
        i_dict_denoised = self.features(imagFFT_denoised, y, seg_id)
        
        for k, v in r_dict.items():
            if k not in ['target', 'seg_id']:
                main_dict['fftr_' + str(k)] = v
                
        for k, v in i_dict.items():
            if k not in ['target', 'seg_id']:
                main_dict['ffti_' + str(k)] = v

        for k, v in main_dict_denoised.items():
            if k not in ['target', 'seg_id']:
                main_dict['denoised_' + str(k)] = v

        for k, v in r_dict_denoised.items():
            if k not in ['target', 'seg_id']:
                main_dict['denoised_fftr_' + str(k)] = v
                
        for k, v in i_dict_denoised.items():
            if k not in ['target', 'seg_id']:
                main_dict['denoised_ffti_' + str(k)] = v
        
        return main_dict
        
    
    def features(self, x, y, seg_id):
        with open(context.saving_path+'logs.txt', 'a') as f:
            print("start to get features at " + str(seg_id), file=f)
        feature_dict = dict()
        feature_dict['target'] = y
        feature_dict['seg_id'] = seg_id

        # create features here
        # numpy

        # lists with parameters to iterate over them
        percentiles = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]
        hann_windows = [50, 150, 1500, 15000]
        spans = [300, 3000, 30000, 50000]
        windows = [10, 50, 100, 500, 1000, 10000]
        borders = list(range(-4000, 4001, 1000))
        peaks = [10, 20, 50, 100]
        coefs = [1, 5, 10, 50, 100]
        lags = [10, 100, 1000, 10000]
        autocorr_lags = [5, 10, 50, 100, 500, 1000, 5000, 10000]

        # basic stats
        feature_dict['mean'] = x.mean()
        feature_dict['std'] = x.std()
        feature_dict['max'] = x.max()
        feature_dict['min'] = x.min()

        # basic stats on absolute values
        feature_dict['mean_change_abs'] = np.mean(np.diff(x))
        feature_dict['abs_max'] = np.abs(x).max()
        feature_dict['abs_mean'] = np.abs(x).mean()
        feature_dict['abs_std'] = np.abs(x).std()

        # geometric and harminic means
        feature_dict['hmean'] = stats.hmean(np.abs(x[np.nonzero(x)[0]]))
        feature_dict['gmean'] = stats.gmean(np.abs(x[np.nonzero(x)[0]])) 

        # k-statistic and moments
        for i in range(1, 5):
            feature_dict['kstat_' + str(i)] = stats.kstat(x, i)
            feature_dict['moment_' + str(i)] = stats.moment(x, i)

        for i in [1, 2]:
            feature_dict['kstatvar_' +str(i)] = stats.kstatvar(x, i)

        # aggregations on various slices of data
        for agg_type, slice_length, direction in product(['std', 'min', 'max', 'mean'], [1000, 10000, 50000], ['first', 'last']):
            if direction == 'first':
                feature_dict[agg_type + '_' + direction + '_' + str(slice_length)] = x[:slice_length].agg(agg_type)
            elif direction == 'last':
                feature_dict[agg_type + '_' + direction + '_' + str(slice_length)] = x[-slice_length:].agg(agg_type)

        feature_dict['max_to_min'] = x.max() / np.abs(x.min())
        feature_dict['max_to_min_diff'] = x.max() - np.abs(x.min())
        feature_dict['count_big'] = len(x[np.abs(x) > 500])
        feature_dict['sum'] = x.sum()

        feature_dict['mean_change_rate'] = calc_change_rate(x)
        # calc_change_rate on slices of data
        for slice_length, direction in product([1000, 10000, 50000], ['first', 'last']):
        #for slice_length, direction in product([50000], ['first', 'last']):
            if direction == 'first':
                feature_dict['mean_change_rate_' + direction+ '_' + str(slice_length)] = calc_change_rate(x[:slice_length])
            elif direction == 'last':
                feature_dict['mean_change_rate_' + direction+ '_' + str(slice_length)] = calc_change_rate(x[-slice_length:])

        # percentiles on original and absolute values
        for p in percentiles:
            feature_dict['percentile_' + str(p)] = np.percentile(x, p)
            feature_dict['abs_percentile_' + str(p)] = np.percentile(np.abs(x), p)

        feature_dict['trend'] = add_trend_feature(x)
        feature_dict['abs_trend'] = add_trend_feature(x, abs_values=True)

        feature_dict['mad'] = x.mad()
        feature_dict['kurt'] = x.kurtosis()
        feature_dict['skew'] = x.skew()
        feature_dict['med'] = x.median()

        feature_dict['Hilbert_mean'] = np.abs(hilbert(x)).mean()

        for hw in hann_windows:
            feature_dict['Hann_window_mean_' + str(hw)] = (convolve(x, hann(hw), mode='same') / sum(hann(hw))).mean()

        feature_dict['classic_sta_lta1_mean'] = classic_sta_lta(x, 500, 10000).mean()
        feature_dict['classic_sta_lta2_mean'] = classic_sta_lta(x, 5000, 100000).mean()
        feature_dict['classic_sta_lta3_mean'] = classic_sta_lta(x, 3333, 6666).mean()
        feature_dict['classic_sta_lta4_mean'] = classic_sta_lta(x, 10000, 25000).mean()
        feature_dict['classic_sta_lta5_mean'] = classic_sta_lta(x, 50, 1000).mean()
        feature_dict['classic_sta_lta6_mean'] = classic_sta_lta(x, 100, 5000).mean()
        feature_dict['classic_sta_lta7_mean'] = classic_sta_lta(x, 333, 666).mean()
        feature_dict['classic_sta_lta8_mean'] = classic_sta_lta(x, 4000, 10000).mean()

        # exponential rolling statistics
        ewma = pd.Series.ewm
        for s in spans:
            feature_dict['exp_Moving_average_'+ str(s) + '_mean'] = (ewma(x, span=s).mean(skipna=True)).mean(skipna=True)
            feature_dict['exp_Moving_average_'+ str(s) + '_std'] = (ewma(x, span=s).mean(skipna=True)).std(skipna=True)
            feature_dict['exp_Moving_std_' + str(s) + '_mean'] = (ewma(x, span=s).std(skipna=True)).mean(skipna=True)
            feature_dict['exp_Moving_std_' + str(s) + '_std'] = (ewma(x, span=s).std(skipna=True)).std(skipna=True)

        feature_dict['iqr'] = np.subtract(*np.percentile(x, [75, 25]))
        feature_dict['iqr1'] = np.subtract(*np.percentile(x, [95, 5]))
        feature_dict['ave10'] = stats.trim_mean(x, 0.1)

        # tfresh features take too long to calculate, so I comment them for now

        feature_dict['abs_energy'] = feature_calculators.abs_energy(x)
        feature_dict['abs_sum_of_changes'] = feature_calculators.absolute_sum_of_changes(x)
        feature_dict['count_above_mean'] = feature_calculators.count_above_mean(x)
        feature_dict['count_below_mean'] = feature_calculators.count_below_mean(x)
        feature_dict['mean_abs_change'] = feature_calculators.mean_abs_change(x)
        feature_dict['mean_change'] = feature_calculators.mean_change(x)
        feature_dict['var_larger_than_std_dev'] = feature_calculators.variance_larger_than_standard_deviation(x)
        feature_dict['range_minf_m4000'] = feature_calculators.range_count(x, -np.inf, -4000)
        feature_dict['range_p4000_pinf'] = feature_calculators.range_count(x, 4000, np.inf)

        for i, j in zip(borders, borders[1:]):
            feature_dict['range_'+ str(i) +'_' + str(j)] = feature_calculators.range_count(x, i, j)

        feature_dict['ratio_unique_values'] = feature_calculators.ratio_value_number_to_time_series_length(x)
        feature_dict['first_loc_min'] = feature_calculators.first_location_of_minimum(x)
        feature_dict['first_loc_max'] = feature_calculators.first_location_of_maximum(x)
        feature_dict['last_loc_min'] = feature_calculators.last_location_of_minimum(x)
        feature_dict['last_loc_max'] = feature_calculators.last_location_of_maximum(x)

        for autocorr_lag in autocorr_lags:
            feature_dict['autocorrelation_' + str(autocorr_lag)] = feature_calculators.autocorrelation(x, autocorr_lag)
            feature_dict['c3_' + str(autocorr_lag)] = feature_calculators.c3(x, autocorr_lag)

        for coeff, attr in product([1, 2, 3, 4, 5], ['real', 'imag', 'angle']):
            feature_dict['fft_'+str(coeff)+'_'+attr] = list(feature_calculators.fft_coefficient(x, [{'coeff': coeff, 'attr': attr}]))[0][1]

        feature_dict['long_strk_above_mean'] = feature_calculators.longest_strike_above_mean(x)
        feature_dict['long_strk_below_mean'] = feature_calculators.longest_strike_below_mean(x)
        feature_dict['cid_ce_0'] = feature_calculators.cid_ce(x, 0)
        feature_dict['cid_ce_1'] = feature_calculators.cid_ce(x, 1)

        for p in percentiles:
            feature_dict['binned_entropy_'+str(p)] = feature_calculators.binned_entropy(x, p)

        feature_dict['num_crossing_0'] = feature_calculators.number_crossing_m(x, 0)

        for peak in peaks:
            feature_dict['num_peaks_' + str(peak)] = feature_calculators.number_peaks(x, peak)

        for c in coefs:
            feature_dict['spkt_welch_density_' + str(c)] = list(feature_calculators.spkt_welch_density(x, [{'coeff': c}]))[0][1]
            feature_dict['time_rev_asym_stat_' + str(c)] = feature_calculators.time_reversal_asymmetry_statistic(x, c)  

        # statistics on rolling windows of various sizes
        
        for w in windows:
            x_roll_std = x.rolling(w).std().dropna().values
            x_roll_mean = x.rolling(w).mean().dropna().values

            feature_dict['ave_roll_std_' + str(w)] = x_roll_std.mean()
            feature_dict['std_roll_std_' + str(w)] = x_roll_std.std()
            feature_dict['maxx_roll_std_' + str(w)] = x_roll_std.max()
            feature_dict['min_roll_std_' + str(w)] = x_roll_std.min()

            for p in percentiles:
                feature_dict['percentile_roll_std_' + str(p)] = np.percentile(x_roll_std, p)

            feature_dict['av_change_abs_roll_std_' + str(w)] = np.mean(np.diff(x_roll_std))
            feature_dict['av_change_rate_roll_std_' + str(w)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
            feature_dict['abs_max_roll_std_' + str(w)] = np.abs(x_roll_std).max()

            feature_dict['ave_roll_mean_' + str(w)] = x_roll_mean.mean()
            feature_dict['std_roll_mean_' + str(w)] = x_roll_mean.std()
            feature_dict['max_roll_mean_' + str(w)] = x_roll_mean.max()
            feature_dict['min_roll_mean_' + str(w)] = x_roll_mean.min()

            for p in percentiles:
                feature_dict['percentile_roll_mean_' + str(p)] = np.percentile(x_roll_mean, p)

            feature_dict['av_change_abs_roll_mean_' + str(w)] = np.mean(np.diff(x_roll_mean))
            feature_dict['av_change_rate_roll_mean_' + str(w)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
            feature_dict['abs_max_roll_mean_' + str(w)] = np.abs(x_roll_mean).max()       
        
        return feature_dict

    def generate(self):
        feature_list = []
        res = Parallel(n_jobs=self.n_jobs,
                        backend='threading')(delayed(self.get_features)(x, y, s)
                                            for s, x, y in tqdm_notebook(self.read_chunks(), total=self.total_data))
        for r in res:
            feature_list.append(r)
        return pd.DataFrame(feature_list)

def train_model(X, X_test, y, n_fold, params=None, model_type='lgb', 
                name='lgb', plot_feature_importance=False, model=None):

    folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators=50000, n_jobs=-1)
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], 
                      eval_metric='mae', early_stopping_rounds=100)
            
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
            
        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
        
        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            
            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = mean_absolute_error(y_valid, y_pred_valid)
            print('Fold {fold_n}. MAE: {score:.4f}.')
            print('')
            
            y_pred = model.predict(X_test).reshape(-1,)
        
        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000,  eval_metric='MAE', **params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)

        #save the model
        joblib.dump(model, context.saving_path+name+"_"+str(fold_n))
        
        oof[valid_index] = y_pred_valid.reshape(-1,)
        scores.append(mean_absolute_error(y_valid, y_pred_valid))

        prediction += y_pred    
        
        if model_type == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_fold
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    if model_type == 'lgb':
        feature_importance["importance"] /= n_fold
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            #sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
        
            return oof, prediction, feature_importance, np.mean(scores)
        return oof, prediction, np.mean(scores)
    
    else:
        return oof, prediction, np.mean(scores)

def master_test(preparation_threads):
    """test of this code"""

    print("wait for presetting")
    for preparation_thread in preparation_threads:
        if preparation_thread != None:
            preparation_thread.join()

    print("start")
    try:
        a = 1 / 0 
        train_features = pd.read_csv(context.saving_path+'train_features2.csv')
        test_features = pd.read_csv(context.saving_path+'test_features2.csv')
        train_features_denoised = pd.read_csv(context.saving_path+'train_features_denoised.csv')
        test_features_denoised = pd.read_csv(context.saving_path+'test_features_denoised.csv')
        train_features_denoised.columns = [(str(i)+'_denoised') for i in train_features_denoised.columns]
        test_features_denoised.columns = [(str(i)+'_denoised') for i in test_features_denoised.columns]
        y = pd.read_csv(context.saving_path+'y.csv')
        X = pd.concat([train_features, train_features_denoised], axis=1).drop(['seg_id_denoised', 'target_denoised'], axis=1)
        X_test = pd.concat([test_features, test_features_denoised], axis=1).drop(['seg_id_denoised', 'target_denoised'], axis=1)

    except:
        training_fg = FeatureGenerator(dtype='train', n_jobs=10, chunk_size=150000)
        training_data = training_fg.generate()
        training_data.to_csv(context.saving_path + 'training_features_150000.csv')

        test_fg = FeatureGenerator(dtype='test', n_jobs=10, chunk_size=150000)
        test_data = test_fg.generate()
        test_data.to_csv(context.saving_path + 'test_features_150000.csv')

        X = training_data.drop(['target', 'seg_id'], axis=1)
        X_test = test_data.drop(['target', 'seg_id'], axis=1)
        y = training_data.target

    means_dict = {}
    for col in X.columns:
        if X[col].isnull().any():
            print(col)
            mean_value = X.loc[X[col] != -np.inf, col].mean()
            X.loc[X[col] == -np.inf, col] = mean_value
            X[col] = X[col].fillna(mean_value)
            means_dict[col] = mean_value

    for col in X_test.columns:
        if X_test[col].isnull().any():
            X_test.loc[X_test[col] == -np.inf, col] = means_dict[col]
            X_test[col] = X_test[col].fillna(means_dict[col])

    n_fold = 5
    submission = pd.read_csv(context.sample_submission_path+'sample_submission.csv', index_col='seg_id')

    # model 1
    """
    params = {'num_leaves': 128,
              'min_data_in_leaf': 79,
              'objective': 'gamma',
              'max_depth': -1,
              'learning_rate': 0.01,
              "boosting": "gbdt",
              "bagging_freq": 5,
              "bagging_fraction": 0.8126672064208567,
              "bagging_seed": 11,
              "metric": 'mae',
              "verbosity": -1,
              'reg_alpha': 0.1302650970728192,
              'reg_lambda': 0.3603427518866501,
              'feature_fraction': 0.1}
    """
    params = {'min_gain_to_split': 5.0, 'max_bin': 224, 'other_rate': 0.46915923708341856, 
              'top_rate': 0.36289022236006685, 'lambda_l1': 1.027236870543288, 'min_data_in_leaf': 12, 
              'metric': 'RMSE', 'feature_fraction': 0.55, 'max_depth': 5, 'boosting': 'goss', 
              'learning_rate': 0.19952347506492585, 'objective': 'huber', 'min_data_in_bin': 95, 
              'lambda_l2': 4.816194305992925, 'bagging_fraction': 0.64, 'num_leaves': 708}
    oof_lgb, prediction_lgb, feature_importance, loss_lgb = train_model(X=X, X_test=X_test, y=y, 
                                                                        params=params, name='lgb1', model_type='lgb', 
                                                                        n_fold=n_fold, plot_feature_importance=True)
    submission['time_to_failure'] = prediction_lgb
    print("prediction_lgb:")
    print(submission.head())
    print("")
    submission.to_csv(context.saving_path+'submission_master_lgb.csv')

    # model 2
    xgb_params = {'eta': 0.03,
              'max_depth': 10,
              'subsample': 0.9,
              'objective': 'reg:linear',
              'eval_metric': 'mae',
              'silent': True,
              'nthread': 4}
    oof_xgb, prediction_xgb, loss_xgb = train_model(X=X, X_test=X_test, y=y, params=xgb_params, 
                                                  n_fold=n_fold, name='xgb1',model_type='xgb')
    submission['time_to_failure'] = prediction_xgb
    print("prediction_xgb:")
    print(submission.head())
    print("")
    submission.to_csv(context.saving_path+'submission_master_xgb.csv')

    # model 3
    model = NuSVR(gamma='scale', nu=0.9, C=10.0, tol=0.01)
    oof_svr, prediction_svr, loss_svr = train_model(X=X, X_test=X_test, y=y, params=None, n_fold=n_fold,
                                                  name='sklearn1', model_type='sklearn', model=model)
    submission['time_to_failure'] = prediction_svr
    print("prediction_svr:")
    print(submission.head())
    print("")
    submission.to_csv(context.saving_path+'submission_master_svr.csv')

    # model 4
    params = {'loss_function':'MAE',
              'task_type':'GPU'}
    oof_cat, prediction_cat, loss_cat = train_model(X=X, X_test=X_test, y=y, params=params, n_fold=n_fold,
                                                  name='cat1', model_type='cat')
    submission['time_to_failure'] = prediction_cat
    print("prediction_cat:")
    print(submission.head())
    print("")
    submission.to_csv(context.saving_path+'submission_master_cat.csv')

    # model 5
    model = KernelRidge(kernel='rbf', alpha=0.15, gamma=0.01)
    oof_r, prediction_r, loss_r = train_model(X=X, X_test=X_test, y=y, params=None, n_fold=n_fold,
                                              name='sklearn3', model_type='sklearn', model=model)
    submission['time_to_failure'] = prediction_r
    print("prediction_r:")
    print(submission.head())
    print("")
    submission.to_csv(context.saving_path+'submission_master_r.csv')

    # output
    submission['time_to_failure'] = (prediction_lgb + prediction_xgb + prediction_svr + prediction_svr1 + prediction_cat + prediction_r) / 6
    print("weighted mean:")
    print(submission.head())
    print("")
    submission.to_csv(context.saving_path+'submission_master.csv')