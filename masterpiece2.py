"""
This code is mainly from https://www.kaggle.com/vettejeep/masters-final-project-model-lb-1-392
"""

import os
import time
import warnings
import traceback
import numpy as np
import pandas as pd
from scipy import stats
import scipy.signal as sg
import multiprocessing as mp
from scipy.signal import hann
from scipy.signal import hilbert
from scipy.signal import convolve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from tqdm import tqdm
warnings.filterwarnings("ignore")

import seismic_prediction as context
import pickle
from tsfresh.feature_extraction import feature_calculators
from scipy import stats

OUTPUT_DIR = context.saving_path  # set for local environment
DATA_DIR = context.sample_submission_path  # set for local environment
TEST_DIR = '/'

SIG_LEN = 150000
NUM_SEG_PER_PROC = 4000
NUM_THREADS = 6

NY_FREQ_IDX = 75000  # the test signals are 150k samples long, Nyquist is thus 75k.
CUTOFF = 18000
MAX_FREQ_IDX = 20000
FREQ_STEP = 2500

def split_raw_data():
    df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

    max_start_index = len(df.index) - SIG_LEN
    slice_len = int(max_start_index / 6)

    for i in range(NUM_THREADS):
        print('working', i)
        df0 = df.iloc[slice_len * i: (slice_len * (i + 1)) + SIG_LEN]
        print(df0.columns)
        df0.to_csv(os.path.join(DATA_DIR, 'raw_data_%d.csv' % i), index=False, line_terminator=os.linesep)
        del df0

    del df

def build_rnd_idxs():
    rnd_idxs = np.zeros(shape=(NUM_THREADS, NUM_SEG_PER_PROC), dtype=np.int32)
    max_start_idx = 100000000

    for i in range(NUM_THREADS):
        np.random.seed(5591 + i)
        start_indices = np.random.randint(0, max_start_idx, size=NUM_SEG_PER_PROC, dtype=np.int32)
        rnd_idxs[i, :] = start_indices

    for i in range(NUM_THREADS):
        print(rnd_idxs[i, :8])
        print(rnd_idxs[i, -8:])
        print(min(rnd_idxs[i,:]), max(rnd_idxs[i,:]))

    np.savetxt(fname=os.path.join(OUTPUT_DIR, 'start_indices_4k.csv'), X=np.transpose(rnd_idxs), fmt='%d', delimiter=',')

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

def des_bw_filter_lp(cutoff=CUTOFF):  # low pass filter
    b, a = sg.butter(4, Wn=cutoff/NY_FREQ_IDX)
    return b, a

def des_bw_filter_hp(cutoff=CUTOFF):  # high pass filter
    b, a = sg.butter(4, Wn=cutoff/NY_FREQ_IDX, btype='highpass')
    return b, a

def des_bw_filter_bp(low, high):  # band pass filter
    b, a = sg.butter(4, Wn=(low/NY_FREQ_IDX, high/NY_FREQ_IDX), btype='bandpass')
    return b, a

def create_features(seg_id, seg, X, st, end):
    print("start "+str(seg_id)) #fortest
    try:
        X.loc[seg_id, 'seg_id'] = np.int32(seg_id)
        X.loc[seg_id, 'seg_start'] = np.int32(st)
        X.loc[seg_id, 'seg_end'] = np.int32(end)
    except:
        pass
    peaks = [10, 20, 50, 100]
    coefs = [1, 5, 10, 50, 100]
    borders = list(range(-4000, 4001, 1000))
    lags = [10, 100, 1000, 10000]
    autocorr_lags = [5, 10, 50, 100, 500, 1000, 5000, 10000]

    xc = pd.Series(seg['acoustic_data'].values)
    xcdm = xc - np.mean(xc)

    b, a = des_bw_filter_lp(cutoff=18000)
    xcz = sg.lfilter(b, a, xcdm)

    zc = np.fft.fft(xcz)
    zc = zc[:MAX_FREQ_IDX]

    # FFT transform values
    realFFT = np.real(zc)
    imagFFT = np.imag(zc)

    freq_bands = [x for x in range(0, MAX_FREQ_IDX, FREQ_STEP)]
    magFFT = np.sqrt(realFFT ** 2 + imagFFT ** 2)
    phzFFT = np.arctan(imagFFT / realFFT)
    phzFFT[phzFFT == -np.inf] = -np.pi / 2.0
    phzFFT[phzFFT == np.inf] = np.pi / 2.0
    phzFFT = np.nan_to_num(phzFFT)

    print("place1 "+str(seg_id)) #fortest
    for freq in freq_bands:
        X.loc[seg_id, 'FFT_Mag_01q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.01)
        X.loc[seg_id, 'FFT_Mag_10q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.1)
        X.loc[seg_id, 'FFT_Mag_90q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.9)
        X.loc[seg_id, 'FFT_Mag_99q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.99)
        X.loc[seg_id, 'FFT_Mag_mean%d' % freq] = np.mean(magFFT[freq: freq + FREQ_STEP])
        X.loc[seg_id, 'FFT_Mag_std%d' % freq] = np.std(magFFT[freq: freq + FREQ_STEP])
        X.loc[seg_id, 'FFT_Mag_max%d' % freq] = np.max(magFFT[freq: freq + FREQ_STEP])

        X.loc[seg_id, 'FFT_Phz_mean%d' % freq] = np.mean(phzFFT[freq: freq + FREQ_STEP])
        X.loc[seg_id, 'FFT_Phz_std%d' % freq] = np.std(phzFFT[freq: freq + FREQ_STEP])
    print("place2 "+str(seg_id)) #fortest
    X.loc[seg_id, 'FFT_Rmean'] = realFFT.mean()
    X.loc[seg_id, 'FFT_Rstd'] = realFFT.std()
    X.loc[seg_id, 'FFT_Rmax'] = realFFT.max()
    X.loc[seg_id, 'FFT_Rmin'] = realFFT.min()
    X.loc[seg_id, 'FFT_Imean'] = imagFFT.mean()
    X.loc[seg_id, 'FFT_Istd'] = imagFFT.std()
    X.loc[seg_id, 'FFT_Imax'] = imagFFT.max()
    X.loc[seg_id, 'FFT_Imin'] = imagFFT.min()

    X.loc[seg_id, 'FFT_Rmean_first_6000'] = realFFT[:6000].mean()
    X.loc[seg_id, 'FFT_Rstd__first_6000'] = realFFT[:6000].std()
    X.loc[seg_id, 'FFT_Rmax_first_6000'] = realFFT[:6000].max()
    X.loc[seg_id, 'FFT_Rmin_first_6000'] = realFFT[:6000].min()
    X.loc[seg_id, 'FFT_Rmean_first_18000'] = realFFT[:18000].mean()
    X.loc[seg_id, 'FFT_Rstd_first_18000'] = realFFT[:18000].std()
    X.loc[seg_id, 'FFT_Rmax_first_18000'] = realFFT[:18000].max()
    X.loc[seg_id, 'FFT_Rmin_first_18000'] = realFFT[:18000].min()
    print("place2 "+str(seg_id)) #fortest
    del xcz
    del zc

    b, a = des_bw_filter_lp(cutoff=2500)
    xc0 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=2500, high=5000)
    xc1 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=5000, high=7500)
    xc2 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=7500, high=10000)
    xc3 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=10000, high=12500)
    xc4 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=12500, high=15000)
    xc5 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=15000, high=17500)
    xc6 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=17500, high=20000)
    xc7 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_hp(cutoff=20000)
    xc8 = sg.lfilter(b, a, xcdm)

    sigs = [xc, pd.Series(xc0), pd.Series(xc1), pd.Series(xc2), pd.Series(xc3),
            pd.Series(xc4), pd.Series(xc5), pd.Series(xc6), pd.Series(xc7), pd.Series(xc8)]
    print("place3 "+str(seg_id)) #fortest
    for i, sig in enumerate(sigs):
        X.loc[seg_id, 'mean_%d' % i] = sig.mean()
        X.loc[seg_id, 'std_%d' % i] = sig.std()
        X.loc[seg_id, 'max_%d' % i] = sig.max()
        X.loc[seg_id, 'min_%d' % i] = sig.min()

        X.loc[seg_id, 'mean_change_abs_%d' % i] = np.mean(np.diff(sig))
        X.loc[seg_id, 'mean_change_rate_%d' % i] = np.mean(np.nonzero((np.diff(sig) / sig[:-1]))[0])
        X.loc[seg_id, 'abs_max_%d' % i] = np.abs(sig).max()
        X.loc[seg_id, 'abs_min_%d' % i] = np.abs(sig).min()

        X.loc[seg_id, 'std_first_50000_%d' % i] = sig[:50000].std()
        X.loc[seg_id, 'std_last_50000_%d' % i] = sig[-50000:].std()
        X.loc[seg_id, 'std_first_10000_%d' % i] = sig[:10000].std()
        X.loc[seg_id, 'std_last_10000_%d' % i] = sig[-10000:].std()

        X.loc[seg_id, 'avg_first_50000_%d' % i] = sig[:50000].mean()
        X.loc[seg_id, 'avg_last_50000_%d' % i] = sig[-50000:].mean()
        X.loc[seg_id, 'avg_first_10000_%d' % i] = sig[:10000].mean()
        X.loc[seg_id, 'avg_last_10000_%d' % i] = sig[-10000:].mean()

        X.loc[seg_id, 'min_first_50000_%d' % i] = sig[:50000].min()
        X.loc[seg_id, 'min_last_50000_%d' % i] = sig[-50000:].min()
        X.loc[seg_id, 'min_first_10000_%d' % i] = sig[:10000].min()
        X.loc[seg_id, 'min_last_10000_%d' % i] = sig[-10000:].min()

        X.loc[seg_id, 'max_first_50000_%d' % i] = sig[:50000].max()
        X.loc[seg_id, 'max_last_50000_%d' % i] = sig[-50000:].max()
        X.loc[seg_id, 'max_first_10000_%d' % i] = sig[:10000].max()
        X.loc[seg_id, 'max_last_10000_%d' % i] = sig[-10000:].max()

        X.loc[seg_id, 'max_to_min_%d' % i] = sig.max() / np.abs(sig.min())
        X.loc[seg_id, 'max_to_min_diff_%d' % i] = sig.max() - np.abs(sig.min())
        X.loc[seg_id, 'count_big_%d' % i] = len(sig[np.abs(sig) > 500])
        X.loc[seg_id, 'sum_%d' % i] = sig.sum()

        X.loc[seg_id, 'mean_change_rate_first_50000_%d' % i] = np.mean(np.nonzero((np.diff(sig[:50000]) / sig[:50000][:-1]))[0])
        X.loc[seg_id, 'mean_change_rate_last_50000_%d' % i] = np.mean(np.nonzero((np.diff(sig[-50000:]) / sig[-50000:][:-1]))[0])
        X.loc[seg_id, 'mean_change_rate_first_10000_%d' % i] = np.mean(np.nonzero((np.diff(sig[:10000]) / sig[:10000][:-1]))[0])
        X.loc[seg_id, 'mean_change_rate_last_10000_%d' % i] = np.mean(np.nonzero((np.diff(sig[-10000:]) / sig[-10000:][:-1]))[0])

        X.loc[seg_id, 'q95_%d' % i] = np.quantile(sig, 0.95)
        X.loc[seg_id, 'q99_%d' % i] = np.quantile(sig, 0.99)
        X.loc[seg_id, 'q05_%d' % i] = np.quantile(sig, 0.05)
        X.loc[seg_id, 'q01_%d' % i] = np.quantile(sig, 0.01)

        X.loc[seg_id, 'abs_q95_%d' % i] = np.quantile(np.abs(sig), 0.95)
        X.loc[seg_id, 'abs_q99_%d' % i] = np.quantile(np.abs(sig), 0.99)
        X.loc[seg_id, 'abs_q05_%d' % i] = np.quantile(np.abs(sig), 0.05)
        X.loc[seg_id, 'abs_q01_%d' % i] = np.quantile(np.abs(sig), 0.01)

        X.loc[seg_id, 'trend_%d' % i] = add_trend_feature(sig)
        X.loc[seg_id, 'abs_trend_%d' % i] = add_trend_feature(sig, abs_values=True)
        X.loc[seg_id, 'abs_mean_%d' % i] = np.abs(sig).mean()
        X.loc[seg_id, 'abs_std_%d' % i] = np.abs(sig).std()

        X.loc[seg_id, 'mad_%d' % i] = sig.mad()
        X.loc[seg_id, 'kurt_%d' % i] = sig.kurtosis()
        X.loc[seg_id, 'skew_%d' % i] = sig.skew()
        X.loc[seg_id, 'med_%d' % i] = sig.median()

        X.loc[seg_id, 'Hilbert_mean_%d' % i] = np.abs(hilbert(sig)).mean()
        X.loc[seg_id, 'Hann_window_mean'] = (convolve(xc, hann(150), mode='same') / sum(hann(150))).mean()

        X.loc[seg_id, 'classic_sta_lta1_mean_%d' % i] = classic_sta_lta(sig, 500, 10000).mean()
        X.loc[seg_id, 'classic_sta_lta2_mean_%d' % i] = classic_sta_lta(sig, 5000, 100000).mean()
        X.loc[seg_id, 'classic_sta_lta3_mean_%d' % i] = classic_sta_lta(sig, 3333, 6666).mean()
        X.loc[seg_id, 'classic_sta_lta4_mean_%d' % i] = classic_sta_lta(sig, 10000, 25000).mean()

        X.loc[seg_id, 'Moving_average_700_mean_%d' % i] = sig.rolling(window=700).mean().mean(skipna=True)
        X.loc[seg_id, 'Moving_average_1500_mean_%d' % i] = sig.rolling(window=1500).mean().mean(skipna=True)
        X.loc[seg_id, 'Moving_average_3000_mean_%d' % i] = sig.rolling(window=3000).mean().mean(skipna=True)
        X.loc[seg_id, 'Moving_average_6000_mean_%d' % i] = sig.rolling(window=6000).mean().mean(skipna=True)

        ewma = pd.Series.ewm
        X.loc[seg_id, 'exp_Moving_average_300_mean_%d' % i] = ewma(sig, span=300).mean().mean(skipna=True)
        X.loc[seg_id, 'exp_Moving_average_3000_mean_%d' % i] = ewma(sig, span=3000).mean().mean(skipna=True)
        X.loc[seg_id, 'exp_Moving_average_30000_mean_%d' % i] = ewma(sig, span=6000).mean().mean(skipna=True)

        no_of_std = 2
        X.loc[seg_id, 'MA_700MA_std_mean_%d' % i] = sig.rolling(window=700).std().mean()
        X.loc[seg_id, 'MA_700MA_BB_high_mean_%d' % i] = (
                    X.loc[seg_id, 'Moving_average_700_mean_%d' % i] + no_of_std * X.loc[seg_id, 'MA_700MA_std_mean_%d' % i]).mean()
        X.loc[seg_id, 'MA_700MA_BB_low_mean_%d' % i] = (
                    X.loc[seg_id, 'Moving_average_700_mean_%d' % i] - no_of_std * X.loc[seg_id, 'MA_700MA_std_mean_%d' % i]).mean()
        X.loc[seg_id, 'MA_400MA_std_mean_%d' % i] = sig.rolling(window=400).std().mean()
        X.loc[seg_id, 'MA_400MA_BB_high_mean_%d' % i] = (
                    X.loc[seg_id, 'Moving_average_700_mean_%d' % i] + no_of_std * X.loc[seg_id, 'MA_400MA_std_mean_%d' % i]).mean()
        X.loc[seg_id, 'MA_400MA_BB_low_mean_%d' % i] = (
                    X.loc[seg_id, 'Moving_average_700_mean_%d' % i] - no_of_std * X.loc[seg_id, 'MA_400MA_std_mean_%d' % i]).mean()
        X.loc[seg_id, 'MA_1000MA_std_mean_%d' % i] = sig.rolling(window=1000).std().mean()

        X.loc[seg_id, 'iqr_%d' % i] = np.subtract(*np.percentile(sig, [75, 25]))
        X.loc[seg_id, 'q999_%d' % i] = np.quantile(sig, 0.999)
        X.loc[seg_id, 'q001_%d' % i] = np.quantile(sig, 0.001)
        X.loc[seg_id, 'ave10_%d' % i] = stats.trim_mean(sig, 0.1)

        # new features-----------------------------------------------------------------------------------------------------------------
        X.loc[seg_id, 'hmean_'+str(i)] = stats.hmean(np.abs(sig[np.nonzero(sig)[0]]))
        X.loc[seg_id, 'gmean_'+str(i)] = stats.gmean(np.abs(sig[np.nonzero(sig)[0]])) 

        for j in range(1, 5):
            X.loc[seg_id, 'kstat_'+str(i)+'_'+str(j)] = stats.kstat(sig, j)
            X.loc[seg_id, 'moment_'+str(i)+'_'+str(j)] = stats.moment(sig, j)

        for j in [1, 2]:
            X.loc[seg_id, 'kstatvar_'+str(i)+'_'+str(j)] = stats.kstatvar(sig, j)

        for c in coefs:
            X.loc[seg_id, 'spkt_welch_density_'+str(i)+'_'+str(c)] = list(feature_calculators.spkt_welch_density(sig, [{'coeff': c}]))[0][1]
            X.loc[seg_id, 'time_rev_asym_stat_'+str(i)+'_'+str(c)] = feature_calculators.time_reversal_asymmetry_statistic(sig, c) 

        X.loc[seg_id, 'abs_energy_'+str(i)] = feature_calculators.abs_energy(sig)
        X.loc[seg_id, 'abs_sum_of_changes_'+str(i)] = feature_calculators.absolute_sum_of_changes(sig)
        X.loc[seg_id, 'count_above_mean_'+str(i)] = feature_calculators.count_above_mean(sig)
        X.loc[seg_id, 'count_below_mean_'+str(i)] = feature_calculators.count_below_mean(sig)
        X.loc[seg_id, 'mean_abs_change_'+str(i)] = feature_calculators.mean_abs_change(sig)
        X.loc[seg_id, 'mean_change_'+str(i)] = feature_calculators.mean_change(sig)
        X.loc[seg_id, 'var_larger_than_std_dev_'+str(i)] = feature_calculators.variance_larger_than_standard_deviation(sig)
        X.loc[seg_id, 'range_minf_m4000_'+str(i)] = feature_calculators.range_count(sig, -np.inf, -4000)
        X.loc[seg_id, 'range_p4000_pinf_'+str(i)] = feature_calculators.range_count(sig, 4000, np.inf)

        for j, k in zip(borders, borders[1:]):
            X.loc[seg_id, 'range_'+str(i)+'_'+str(j)+'_'+str(k)] = feature_calculators.range_count(sig, j, k)

        X.loc[seg_id, 'ratio_unique_values_'+str(i)] = feature_calculators.ratio_value_number_to_time_series_length(sig)
        X.loc[seg_id, 'first_loc_min_'+str(i)] = feature_calculators.first_location_of_minimum(sig)
        X.loc[seg_id, 'first_loc_max_'+str(i)] = feature_calculators.first_location_of_maximum(sig)
        X.loc[seg_id, 'last_loc_min_'+str(i)] = feature_calculators.last_location_of_minimum(sig)
        X.loc[seg_id, 'last_loc_max_'+str(i)] = feature_calculators.last_location_of_maximum(sig)

        for lag in lags:
            X.loc[seg_id, 'time_rev_asym_stat_'+str(i)+'_'+str(lag)] = feature_calculators.time_reversal_asymmetry_statistic(sig, lag)
        for autocorr_lag in autocorr_lags:
            X.loc[seg_id, 'autocorrelation_'+str(i)+'_'+str(autocorr_lag)] = feature_calculators.autocorrelation(sig, autocorr_lag)
            X.loc[seg_id, 'c3_'+str(i)+'_'+str(autocorr_lag)] = feature_calculators.c3(sig, autocorr_lag)

        X.loc[seg_id, 'long_strk_above_mean_'+str(i)] = feature_calculators.longest_strike_above_mean(sig)
        X.loc[seg_id, 'long_strk_below_mean_'+str(i)] = feature_calculators.longest_strike_below_mean(sig)
        X.loc[seg_id, 'cid_ce_0_'+str(i)] = feature_calculators.cid_ce(sig, 0)
        X.loc[seg_id, 'cid_ce_1_'+str(i)] = feature_calculators.cid_ce(sig, 1)

        X.loc[seg_id, 'num_crossing_0_'+str(i)] = feature_calculators.number_crossing_m(sig, 0)

        for peak in peaks:
            X.loc[seg_id, 'num_peaks_' + str(peak)] = feature_calculators.number_peaks(sig, peak)
        # new features over------------------------------------------------------------------------------------------------------------
    print("place4 "+str(seg_id)) #fortest
    for windows in [10, 100, 1000]:
        x_roll_std = xc.rolling(windows).std().dropna().values
        x_roll_mean = xc.rolling(windows).mean().dropna().values

        X.loc[seg_id, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()
        X.loc[seg_id, 'std_roll_std_' + str(windows)] = x_roll_std.std()
        X.loc[seg_id, 'max_roll_std_' + str(windows)] = x_roll_std.max()
        X.loc[seg_id, 'min_roll_std_' + str(windows)] = x_roll_std.min()
        X.loc[seg_id, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
        X.loc[seg_id, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)
        X.loc[seg_id, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
        X.loc[seg_id, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
        X.loc[seg_id, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
        X.loc[seg_id, 'av_change_rate_roll_std_' + str(windows)] = np.mean(
            np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
        X.loc[seg_id, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()

        X.loc[seg_id, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
        X.loc[seg_id, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
        X.loc[seg_id, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
        X.loc[seg_id, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
        X.loc[seg_id, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
        X.loc[seg_id, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
        X.loc[seg_id, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
        X.loc[seg_id, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
        X.loc[seg_id, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
        X.loc[seg_id, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(
            np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
        X.loc[seg_id, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()

    return X

def build_fields(proc_id):
    success = 1
    count = 0
    with open(context.saving_path+'master2_logs/logs'+str(proc_id)+'.txt', 'a') as f:
        print("%d starts to build fields", file=f)
    try:
        seg_st = int(NUM_SEG_PER_PROC * proc_id)
        with open(context.saving_path+'master2_logs/logs'+str(proc_id)+'.txt', 'a') as f:
            print("%d loads some files", file=f)
        train_df = pd.read_csv(os.path.join(DATA_DIR, 'raw_data_%d.csv' % proc_id), dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
        len_df = len(train_df.index)
        start_indices = (np.loadtxt(fname=os.path.join(OUTPUT_DIR, 'start_indices_4k.csv'), dtype=np.int32, delimiter=','))[:, proc_id]
        train_X = pd.DataFrame(dtype=np.float64)
        train_y = pd.DataFrame(dtype=np.float64, columns=['time_to_failure'])
        t0 = time.time()

        with open(context.saving_path+'master2_logs/logs'+str(proc_id)+'.txt', 'a') as f:
            print("%d start iteration", file=f)
        for seg_id, start_idx in zip(range(seg_st, seg_st + NUM_SEG_PER_PROC), start_indices):
            end_idx = np.int32(start_idx + 150000)
            with open(context.saving_path+'master2_logs/logs'+str(proc_id)+'.txt', 'a') as f:
                print('working: %d, %d, %d to %d of %d' % (proc_id, seg_id, start_idx, end_idx, len_df), file=f)
            seg = train_df.iloc[start_idx: end_idx]
            # train_X = create_features_pk_det(seg_id, seg, train_X, start_idx, end_idx)
            train_X = create_features(seg_id, seg, train_X, start_idx, end_idx)
            train_y.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]

            if count == 10: 
                print('saving: %d, %d to %d' % (seg_id, start_idx, end_idx))
                train_X.to_csv('train_x_%d.csv' % proc_id, index=False, line_terminator=os.linesep)
                train_y.to_csv('train_y_%d.csv' % proc_id, index=False, line_terminator=os.linesep)

            count += 1

        print('final_save, process id: %d, loop time: %.2f for %d iterations' % (proc_id, time.time() - t0, count))
        train_X.to_csv(os.path.join(OUTPUT_DIR, 'train_x_%d.csv' % proc_id), index=False, line_terminator=os.linesep)
        train_y.to_csv(os.path.join(OUTPUT_DIR, 'train_y_%d.csv' % proc_id), index=False, line_terminator=os.linesep)

    except:
        print(traceback.format_exc())
        success = 0

    return success  # 1 on success, 0 if fail

def run_mp_build():
    t0 = time.time()
    num_proc = NUM_THREADS
    pool = mp.Pool(processes=num_proc)
    results = [pool.apply_async(build_fields, args=(pid, )) for pid in range(NUM_THREADS)]
    output = [p.get() for p in results]
    num_built = sum(output)
    pool.close()
    pool.join()
    print(num_built)
    print('Run time: %.2f' % (time.time() - t0))

def join_mp_build():
    df0 = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_x_%d.csv' % 0))
    df1 = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_y_%d.csv' % 0))

    for i in range(1, NUM_THREADS):
        print('working %d' % i)
        temp = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_x_%d.csv' % i))
        df0 = df0.append(temp)

        temp = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_y_%d.csv' % i))
        df1 = df1.append(temp)

    df0.to_csv(os.path.join(OUTPUT_DIR, 'train_x.csv'), index=False, line_terminator=os.linesep)
    df1.to_csv(os.path.join(OUTPUT_DIR, 'train_y.csv'), index=False, line_terminator=os.linesep)

def create_features_pk_det(seg_id, seg, X, st, end):
    X.loc[seg_id, 'seg_id'] = np.int32(seg_id)
    X.loc[seg_id, 'seg_start'] = np.int32(st)
    X.loc[seg_id, 'seg_end'] = np.int32(end)

    sig = pd.Series(seg['acoustic_data'].values)
    b, a = des_bw_filter_lp(cutoff=18000)
    sig = sg.lfilter(b, a, sig)

    peakind = []
    noise_pct = .001
    count = 0

    while len(peakind) < 12 and count < 24:
        peakind = sg.find_peaks_cwt(sig, np.arange(1, 16), noise_perc=noise_pct, min_snr=4.0)
        noise_pct *= 2.0
        count += 1

    if len(peakind) < 12:
        print('Warning: Failed to find 12 peaks for %d' % seg_id)

    while len(peakind) < 12:
        peakind.append(149999)

    df_pk = pd.DataFrame(data={'pk': sig[peakind], 'idx': peakind}, columns=['pk', 'idx'])
    df_pk.sort_values(by='pk', ascending=False, inplace=True)

    for i in range(0, 12):
        X.loc[seg_id, 'pk_idx_%d' % i] = df_pk['idx'].iloc[i]
        X.loc[seg_id, 'pk_val_%d' % i] = df_pk['pk'].iloc[i]

    return X

def build_test_fields():
    train_X = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_x.csv'))
    try:
        train_X.drop(labels=['seg_id', 'seg_start', 'seg_end'], axis=1, inplace=True)
    except:
        pass

    submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'), index_col='seg_id')
    test_X = pd.DataFrame(columns=train_X.columns, dtype=np.float64, index=submission.index)

    print('start for loop')
    count = 0
    for seg_id in test_X.index:  # just tqdm in IDE
        seg = pd.read_csv(os.path.join(TEST_DIR, 'test', str(seg_id) + '.csv'))
        # train_X = create_features_pk_det(seg_id, seg, train_X, start_idx, end_idx)
        test_X = create_features(seg_id, seg, test_X, 0, 0)

        if count % 1 == 0:
            with open(context.saving_path+'master2_logs/test_logs.txt', 'a') as f:
                print('working'+ seg_id, file=f)
        count += 1

    test_X.to_csv(os.path.join(OUTPUT_DIR, 'test_x.csv'), index=False, line_terminator=os.linesep)

def scale_fields(fn_train='train_x.csv', fn_test='test_x.csv', 
                 fn_out_train='scaled_train_X.csv' , fn_out_test='scaled_test_X.csv'):
    train_X = pd.read_csv(os.path.join(OUTPUT_DIR, fn_train))
    try:
        train_X.drop(labels=['seg_id', 'seg_start', 'seg_end'], axis=1, inplace=True)
    except:
        pass
    test_X = pd.read_csv(os.path.join(OUTPUT_DIR, fn_test))

    print('start scaler')
    scaler = StandardScaler()
    scaler.fit(train_X)
    with open(context.saving_path+"scaler_master2", "wb") as f:
        pickle.dump(scaler, f)

    scaled_train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)
    scaled_test_X = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns)

    scaled_train_X.to_csv(os.path.join(OUTPUT_DIR, fn_out_train), index=False, line_terminator=os.linesep)
    scaled_test_X.to_csv(os.path.join(OUTPUT_DIR, fn_out_test), index=False, line_terminator=os.linesep)

def scale_one_field(fn, fn_out, fn_refer=None):
    """scale a file with a reference with a file name fn_refer"""

    train_X = pd.read_csv(os.path.join(OUTPUT_DIR, fn))

    try:
        train_X.drop(labels=['seg_id', 'seg_start', 'seg_end'], axis=1, inplace=True)
    except:
        pass

    print('start scaler')
    with open(context.saving_path+"scaler_master2", "rb") as f:
        scaler = pickle.load(f)

    if fn_refer != None:
        refer_X = pd.read_csv(os.path.join(OUTPUT_DIR, fn_refer))
        means_dict = {}
        for col in refer_X.columns:
            mean_value = refer_X.loc[refer_X[col] != -np.inf, col].mean()
            means_dict[col] = mean_value

        for col in train_X.columns:
            if train_X[col].isnull().any():
                train_X.loc[train_X[col] == -np.inf, col] = means_dict[col]
                train_X[col] = train_X[col].fillna(means_dict[col])

    scaled_train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)

    if context.on_cloud:
        scaled_train_X.to_csv(os.path.join(OUTPUT_DIR, fn_out), index=False, line_terminator=os.linesep)
    else:
        scaled_train_X.to_csv(os.path.join(OUTPUT_DIR, fn_out), index=False)
"""
params = {'num_leaves': 21,
         'min_data_in_leaf': 20,
         'objective':'regression',
         'learning_rate': 0.001,
         'max_depth': 108,
         "boosting": "gbdt",
         "feature_fraction": 0.91,
         "bagging_freq": 1,
         "bagging_fraction": 0.91,
         "bagging_seed": 42,
         "metric": 'mae',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "random_state": 42}
"""
params = {'min_gain_to_split': 5.0, 'max_bin': 224, 'other_rate': 0.46915923708341856, 
          'top_rate': 0.36289022236006685, 'lambda_l1': 1.027236870543288, 'min_data_in_leaf': 12, 
          'metric': 'RMSE', 'feature_fraction': 0.55, 'max_depth': 5, 'boosting': 'goss', 
          'learning_rate': 0.01, 'objective': 'huber', 'min_data_in_bin': 95, 
          'lambda_l2': 4.816194305992925, 'bagging_fraction': 0.64, 'num_leaves': 708}

def feature_filter(features, feature_importance , threshold=30):
    """filter features by importance"""

    returned_features = pd.DataFrame(index=features.index)
    for i in range(len(features.columns)):
        right_lines = feature_importance[feature_importance["feature"]==features.columns[i]]
        if np.mean(right_lines.loc[:,'importance']) > threshold:
            a = features[features.columns[i]]
            returned_features = pd.concat([returned_features, features[features.columns[i]]], axis=1)
    
    return returned_features 

def lgb_base_model(sample_threshold=999999, feature_threshold=0):
    """note that samples of errors lower than sample_threshold and
       features of importance higher that feature_threshold will be 
       used for training,"""

    maes = []
    rmses = []
    submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'), index_col='seg_id')
    scaled_train_X = pd.read_csv(os.path.join(OUTPUT_DIR, 'scaled_train_X.csv'))
    scaled_test_X = pd.read_csv(os.path.join(OUTPUT_DIR, 'scaled_test_X.csv'))
    train_y = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_y.csv'))
    predictions = np.zeros(len(scaled_test_X))
    train_predictions = np.zeros(len(scaled_train_X))

    if sample_threshold != 999999:
        sample_index = []
        train_result = pd.read_csv(context.saving_path+'master2_output/train_result.csv')
        for i in range(len(sample_threshold)):
            if abs(train_result[i].values-train_y) < sample_threshold:
                sample_index.append(i)
                
    if feature_threshold != 0:
        scaled_train_X = scaled_train_X

    n_fold = 8
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = scaled_train_X.columns

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_train_X, train_y.values)):
        print('working fold %d' % fold_)
        strLog = "fold {}".format(fold_)
        print(strLog)

        X_tr, X_val = scaled_train_X.iloc[trn_idx], scaled_train_X.iloc[val_idx]
        y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params, n_estimators=60000, n_jobs=-1)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_tr, y_tr), (X_val, y_val)], eval_metric='mae',
                  verbose=1000, early_stopping_rounds=200)

        # predictions
        preds = model.predict(scaled_test_X, num_iteration=model.best_iteration_)
        predictions += preds / folds.n_splits
        preds = model.predict(scaled_train_X, num_iteration=model.best_iteration_)
        train_predictions += preds / folds.n_splits
        preds = model.predict(X_val, num_iteration=model.best_iteration_)

        # mean absolute error
        mae = mean_absolute_error(y_val, preds)
        print('MAE: %.6f' % mae)
        maes.append(mae)

        # root mean squared error
        rmse = mean_squared_error(y_val, preds)
        print('RMSE: %.6f' % rmse)
        rmses.append(rmse)

        fold_importance_df['importance_%d' % fold_] = model.feature_importances_[:len(scaled_train_X.columns)]

    print('MAEs', maes)
    print('MAE mean: %.6f' % np.mean(maes))
    print('RMSEs', rmses)
    print('RMSE mean: %.6f' % np.mean(rmses))

    submission.time_to_failure = predictions
    submission.to_csv(context.saving_path+'master2_output/submission_lgb_8_80k_108dp.csv', 
                      index=False, line_terminator=os.linesep)
    if feature_threshold == 0:
        fold_importance_df.to_csv(context.saving_path+'master2_output/fold_imp_lgb_8_80k_108dp.csv', 
                                  line_terminator=os.linesep)  # index needed, it is seg id
    if sample_threshold == 999999:
        train_result = pd.DataFrame(columns=['time_to_failure'])
        train_result.time_to_failure = train_predictions
        train_result.to_csv(context.saving_path+'master2_output/train_result.csv', 
                            index=False, line_terminator=os.linesep)

def make_new_sample(event):
    """get new training samples from new data"""

    acData = np.load(os.path.join(context.sample_submission_path+'p4581/', "earthquake_%03d.npz"%(event,)))['acoustic_data'] 
    acTime = np.load(os.path.join(context.sample_submission_path+'p4581/', "earthquake_%03d.npz"%(event,)))['ttf'] 

    steps = np.arange(4096) * 0.252016890769332e-6
    t = acTime[:, np.newaxis] + np.flip(steps)[np.newaxis]

    return acData.flatten(), t.flatten()

def train_model(X_list=[], X_test=None, y_list=[], params=None, n_fold=6, model_type='lgb', name='lgb',
                plot_feature_importance=False, model=None):

    if len(X_list) == 0:
        y_list = []
        for i in range(NUM_THREADS):
            X_list.append(pd.read_csv(os.path.join(OUTPUT_DIR, 'scaled_train_x_%d.csv' % i)))
            y_list.append(pd.read_csv(os.path.join(OUTPUT_DIR, 'train_y_%d.csv' % i)))
    
    if X_test == None:
        X_test = pd.read_csv(os.path.join(OUTPUT_DIR, 'test_x.csv'))

    total_len = 0
    for X in X_list:
        total_len += len(X)
    oof = np.zeros(total_len)
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()

    for fold_n in range(n_fold):
        with open(context.saving_path+"logs.txt", 'a') as f:
            print('Fold', fold_n, 'started at', time.ctime(), file=f)
        X_train = pd.concat(X_list[:fold_n]+X_list[fold_n+1:]).sample(frac=1, random_state=42)
        y_train = pd.concat(y_list[:fold_n]+y_list[fold_n+1:]).sample(frac=1, random_state=42)
        X_valid = X_list[fold_n]
        y_valid = y_list[fold_n]
        print(len(X_train))
        print(len(y_train))
        print(len(X_valid))
        print(len(y_valid))

        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators = 100000, n_jobs = -1)
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
                      verbose=1000, early_stopping_rounds=10000)
            
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
            
        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=40000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
        
        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            
            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = mean_absolute_error(y_valid, y_pred_valid)
            
            y_pred = model.predict(X_test).reshape(-1,)
        
        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000,  eval_metric='MAE', **params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)

        if model_type == 'gp':
            model = genetic.SymbolicRegressor(**params, verbose=1)
            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)

        if model_type == 'nn':
            y_pred_valid = NN.NN_model_hr(X_train, X_valid, y_train, y_valid, params)
            y_pred = prediction  # not used

        #save the model
        #joblib.dump(model, context.saving_path+name+"_"+str(fold_n))
        
        #oof[valid_index] = y_pred_valid.reshape(-1,)
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

    with open(context.saving_path+"logs.txt", 'a') as f:
        print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)), file=f)

    if model_type == 'lgb':
        feature_importance["importance"] /= n_fold
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            #sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
        
            return prediction, feature_importance, np.mean(scores)
        return prediction, np.mean(scores)
    
    else:
        return prediction, np.mean(scores)

import multiprocessing
def process_func(data_list, output_file_name):
    """process function"""

    output_file = pd.DataFrame()
    for data in data_list:
        #seg_id, seg = data
        create_features(data[0], data[1], output_file, 0, 0)
        print('seg_'+str(data[0])+"is generated")

    output_file.to_csv(context.saving_path+output_file_name, index=False, line_terminator=os.linesep)

import gc
def make_new_data_csv(event_st=27, sample_num=8400, sample_skipped=0, output_file_name='new_samples', 
                      samples_each_proc=1000):
    """generate a new data csv"""

    processes = []
    data_cache_X = np.array([], dtype=np.int16)
    data_cache_y = np.array([])
    data_list_X = []
    output_file_y = pd.DataFrame()
    try:
        for i in range(sample_num+sample_skipped):
            if len(data_cache_X) < SIG_LEN:
                X, y = make_new_sample(event_st)
                event_st += 1
                data_cache_X = np.concatenate([data_cache_X, X])
                data_cache_y = np.concatenate([data_cache_y, y])
                gc.collect()


                seg = pd.DataFrame(columns=['acoustic_data'])
                seg['acoustic_data'] = data_cache_X[:SIG_LEN]
                data_list_X.append([i, seg])
                output_file_y.loc[i, 'time_to_failure'] = data_cache_y[SIG_LEN]
                
            data_cache_X = np.delete(data_cache_X, np.s_[:SIG_LEN])
            data_cache_y = np.delete(data_cache_y, np.s_[:SIG_LEN])

            if len(data_list_X) >= samples_each_proc:
                processes.append(multiprocessing.Process(target=process_func, 
                                                            args=(data_list_X, 'new_feature_X_%d.csv'%(len(processes),))))
                processes[-1].daemon = True
                processes[-1].start()
                data_list_X = []
                print('process%d_started'%(len(processes),))
    except:
        print("no new data to load, loaded samples : %d"%(i,))
        
    # last process
    if len(data_list_X) != 0:
        processes.append(multiprocessing.Process(target=process_func, 
                                                    args=(data_list_X, 'new_feature_X_%d.csv'%(len(processes),))))
        processes[-1].daemon = True
        processes[-1].start()
        print('process%d_started'%(len(processes),))
        
    # wait for end of all processes 
    for process in processes:
        process.join()

    #output csv
    df0 = pd.read_csv(context.saving_path+'new_feature_X_%d.csv'%(0,))

    for i in range(1, len(processes)):

        temp = pd.read_csv(context.saving_path+'new_feature_X_%d.csv'%(i,))
        df0 = df0.append(temp)

    df0.to_csv(context.saving_path+output_file_name+'_X.csv', index=False, line_terminator=os.linesep)
    output_file_y.to_csv(context.saving_path+output_file_name+'_y.csv', index=False, line_terminator=os.linesep)

def make_on_overlap_data_csv(output_file_name='train_on_overlap', samples_each_proc=1000):

    """generate a new train csv without overlapping"""

    processes = []
    data_list_X = []
    output_file_y = pd.DataFrame()
    raw_data = pd.read_csv(context.traning_csv_path_name)

    try:
        for i in range(int(len(raw_data)/SIG_LEN)):
            
            seg = raw_data[i*SIG_LEN:(i+1)*SIG_LEN]
            data_list_X.append([i, seg])
            output_file_y.loc[i, 'time_to_failure'] = seg['time_to_failure'].values[-1]
                
            if len(data_list_X) >= samples_each_proc:
                processes.append(multiprocessing.Process(target=process_func, 
                                                            args=(data_list_X, 'new_feature_X_%d.csv'%(len(processes),))))
                processes[-1].daemon = True
                processes[-1].start()
                data_list_X = []
                print('process%d_started'%(len(processes),))
                gc.collect()
    except:
        print("no new data to load, loaded samples : %d"%(i,))
        
    # last process
    if len(data_list_X) != 0:
        processes.append(multiprocessing.Process(target=process_func, 
                                                    args=(data_list_X, 'new_feature_X_%d.csv'%(len(processes),))))
        processes[-1].daemon = True
        processes[-1].start()
        print('process%d_started'%(len(processes),))
        
    # wait for end of all processes 
    for process in processes:
        process.join()

    #output csv
    df0 = pd.read_csv(context.saving_path+'new_feature_X_%d.csv'%(0,))

    for i in range(1, len(processes)):

        temp = pd.read_csv(context.saving_path+'new_feature_X_%d.csv'%(i,))
        df0 = df0.append(temp)

    df0.to_csv(context.saving_path+output_file_name+'_X.csv', index=False, line_terminator=os.linesep)
    output_file_y.to_csv(context.saving_path+output_file_name+'_y.csv', index=False, line_terminator=os.linesep)

import hyperopt_test

X_train_scaled = 0
X_test_scaled = 0
y_tr = 0

def model5(params):

    global X_train_scaled, X_test_scaled, y_tr

    n_fold = 1
    _, _, error = hyperopt_test.train_model(X=X_train_scaled, X_test=X_test_scaled, y=y_tr, params=params,
                                model_type='nn')

    return error

X_train = 0
y_train = 0
X_valid = 0
y_valid = 0
X_test = 0

def model3(params):

    global X_train, y_train, X_valid, y_valid, X_test

    """
    refer_X = X_train
    X_train_cache = X_train.copy()
    X_valid_cache = X_valid.copy()
    X_test_cache = X_test.copy()

    means_dict = {}
    for col in refer_X.columns:
        mean_value = refer_X.loc[refer_X[col] != -np.inf, col].mean()
        means_dict[col] = mean_value

    
    for col in X_train_cache.columns:
        if X_train_cache[col].isnull().any():
            X_train_cache.loc[X_train_cache[col] == -np.inf, col] = means_dict[col]
            X_train_cache[col] = X_train_cache[col].fillna(means_dict[col])
        if X_valid_cache[col].isnull().any():
            X_valid_cache.loc[X_valid_cache[col] == -np.inf, col] = means_dict[col]
            X_valid_cache[col] = X_valid_cache[col].fillna(means_dict[col])
        if X_test_cache[col].isnull().any():
            X_test_cache.loc[X_test_cache[col] == -np.inf, col] = means_dict[col]
            X_test_cache[col] = X_test_cache[col].fillna(means_dict[col])
    """

    model = hyperopt_test.NuSVR(nu=params['nu'], C=params['C'], kernel=params['kernel'], degree=params['degree'],
                                gamma=params['gamma'], coef0=params['coef0'], shrinking=params['shrinking'], 
                                tol=params['tol'])
    _, _, error = hyperopt_test.train_model(X=X_train, X_test=X_test, y=y_train, params=None, 
                                            name='sklearn2', model_type='sklearn', model=model, n_fold=8,
                                            use_in_param_turn=True) #X_valid=X_valid_cache, y_valid=y_valid)

    return error

def master_test(preparation_threads, model_name=None, n_estimators=20000, n_fold=8, postfix='', use_new_valid=False):
    """Note that postfix is used for differentiating csv names"""

    print("wait for presetting")
    for preparation_thread in preparation_threads:
        if preparation_thread != None:
            preparation_thread.join()
    '''
    import data_processing
    data_processing.delete_csv_space_bar(context.saving_path+'master2_output/submission_master2_cat_5000.csv',
                                            context.saving_path+'master2_output/submission_master2_cat_5000_local.csv')
    data_processing.delete_csv_space_bar(context.saving_path+'master2_output/submission_master2_cat_7000.csv',
                                            context.saving_path+'master2_output/submission_master2_cat_7000_local.csv')
    data_processing.delete_csv_space_bar(context.saving_path+'master2_output/submission_master2_cat_8000.csv',
                                            context.saving_path+'master2_output/submission_master2_cat_8000_local.csv')
    '''
    #make_new_data_csv(27, 99999999999, 'new_samples_all', 10000)
    """
    scale_one_field(os.path.join(OUTPUT_DIR, 'new_samples_X.csv'), os.path.join(OUTPUT_DIR, 'scaled_new_samples_X.csv'),
                    os.path.join(OUTPUT_DIR, 'scaled_train_X.csv'))

    
    
    print("split_raw_data")
    split_raw_data()
    print("build_rnd_idxs")
    build_rnd_idxs()
    
    print("run_mp_build")
    run_mp_build()
    join_mp_build()
    
    build_test_fields()
    scale_fields()
    
    for i in range(NUM_THREADS):
        scale_one_field(os.path.join(OUTPUT_DIR, 'train_x_%d.csv' % i), 
                        os.path.join(OUTPUT_DIR, 'scaled_train_x_%d.csv' % i))
    """
    global X_train, y_train, X_valid, y_valid, X_test
    
    X_train = pd.read_csv(os.path.join(OUTPUT_DIR, 'scaled_train_X.csv'))
    new_X_train = pd.read_csv(os.path.join(OUTPUT_DIR, 'scaled_new_samples_X.csv'))
    more_X_train = pd.concat([X_train, new_X_train])
    y_train = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_y.csv'))
    new_y_train = pd.read_csv(os.path.join(OUTPUT_DIR, 'new_samples_y.csv'))
    more_y_train = pd.concat([y_train, new_y_train])
    X_valid = []
    y_valid = []
    if use_new_valid:
        X_valid=new_X_train.sample(frac=0.3, random_state=11)
        y_valid=new_y_train.sample(frac=0.3, random_state=11)
    
    X_test = pd.read_csv(os.path.join(OUTPUT_DIR, 'scaled_test_X.csv'))
    n_est = '_' + postfix
    if type(n_estimators)==type(1) and n_estimators != 20000:
        n_est = '_'+str(n_estimators)+ '_' + postfix

    # test
    if model_name=='test' or model_name==None:
        params = {'max_depth': 20, 'learning_rate': 0.037491213086246415, 'num_leaves': 1352, 'min_data_in_leaf': 1,
                  'metric': 'RMSE', 'lambda_l2': 3.0472684549095574, 'boosting': 'gbdt', 'bagging_fraction': 0.53, 
                  'min_gain_to_split': 1.01, 'lambda_l1': 0.38083001367253844, 'min_data_in_bin': 251, 'objective': 'gamma', 
                  'feature_fraction': 0.99, 'max_bin': 215, 'subsample': 0.6587682468738089}
        _, prediction_lgb, scr = hyperopt_test.train_model(X_train, X_test, y_train, params=params, n_fold=n_fold, 
                                                           n_estimators=n_estimators, model_type='cat', name='cat',
                                                           X_valid=X_valid, y_valid=y_valid)

    if model_name=='lgb' or model_name==None:
        # output a csv
        submission = pd.read_csv(context.sample_submission_path+'sample_submission.csv', index_col='seg_id')
        params = {'max_depth': 20, 'learning_rate': 0.037491213086246415, 'num_leaves': 1352, 'min_data_in_leaf': 1,
                  'metric': 'RMSE', 'lambda_l2': 3.0472684549095574, 'boosting': 'gbdt', 'bagging_fraction': 0.53, 
                  'min_gain_to_split': 1.01, 'lambda_l1': 0.38083001367253844, 'min_data_in_bin': 251, 'objective': 'gamma', 
                  'feature_fraction': 0.99, 'max_bin': 215, 'subsample': 0.6587682468738089}
        params = hyperopt_test.params_preprocessing(params, model_name)
        _, prediction_lgb, scr = hyperopt_test.train_model(X_train, X_test, y_train, params=params, plot_feature_importance=True, 
                                                           n_estimators=n_estimators, n_fold=n_fold, X_valid=X_valid, y_valid=y_valid)
        submission['time_to_failure'] = prediction_lgb
        print("prediction_lgb:")
        print(submission.head())
        print("")
        if context.on_cloud:
            submission.to_csv(context.saving_path+'submission_master2_lgb_bestparams_fold'+str(n_fold)+n_est+'_oncloud.csv', line_terminator=os.linesep)
        else:
            submission.to_csv(context.saving_path+'submission_master2_lgb_bestparams_foold8'+str(n_fold)+n_est+'.csv')

    if model_name=='xgb' or model_name==None:

        # output a csv
        submission = pd.read_csv(context.sample_submission_path+'sample_submission.csv', index_col='seg_id')
        '''
        params = {'boosting': 'gbtree', 'eval_metric' : 'rmse', 'reg_alpha': 0.37824240549600363, 'max_depth': 4, 
                  'colsample_bylevel': 0.26, 'subsample': 0.9500000000000001, 'reg_lambda': 3.2599423926177638, 
                  'objective': 'reg:linear', 'min_child_weight': 2.8828969938855473, 'gamma': 0.2543374108559115, 
                  'colsample_bytree': 0.86, 'colsample_bynode': 0.44, 'tree_method': 'approx', #'learning_rate': 0.00199865978032934}
                  'learning_rate': 0.19986597803293443}
        '''
        params = {'eval_metric': 'mae', 'objective': 'reg:linear', 'subsample': 0.9500000000000001, 'colsample_bytree': 0.72, 
                  'colsample_bynode': 0.9400000000000001, 'reg_lambda': 1.3521810608413789e-05, 'boosting': 'gblinear', 
                  'gamma': 0.46930956951472047, 'colsample_bylevel': 0.18, 'learning_rate': 0.01, 'tree_method': 'approx', 
                  'reg_alpha': 0.010103208706898292, 'min_child_weight': 4.0478106371276725, 'max_depth': 17} 
        params = hyperopt_test.params_preprocessing(params, model_name)

        _, prediction_xgb, scr = hyperopt_test.train_model(X_train, X_test, y_train, params=params, model_type='xgb', name='xgb', 
                                                           n_estimators=n_estimators, n_fold=n_fold, X_valid=X_valid, y_valid=y_valid)
        submission['time_to_failure'] = prediction_xgb
        print("prediction_xgb:")
        print(submission.head())
        print("")
        if context.on_cloud:
            submission.to_csv(context.saving_path+'submission_master2_xgb_bestparams_fold'+str(n_fold)+n_est+'_oncloud.csv', line_terminator=os.linesep)
        else:
            submission.to_csv(context.saving_path+'submission_master2_xgb_bestparams_foold'+str(n_fold)+n_est+'.csv')


    if model_name=='cat' or model_name==None:
        submission = pd.read_csv(context.sample_submission_path+'sample_submission.csv', index_col='seg_id')
        params = {'learning_rate': 0.09159686068779108, 'fold_len_multiplier': 2.058197813635219, 'min_data_in_leaf': 8, 
                  'od_type': 'Iter', 'objective': 'MAE', 'bootstrap_type': 'Bernoulli', 'l2_leaf_reg': 2.558997618660658, 
                  'random_strength': 0.034282019666114015, 'verbose': 0, 'od_wait': 25, 'max_bin': 134, 'grow_policy': 'SymmetricTree', 
                  'score_function': 'Correlation', 'task_type': 'GPU', 'depth': 8, 'eval_metric': 'MAE', 
                  'leaf_estimation_backtracking': 'AnyImprovement'}
        params = hyperopt_test.params_preprocessing(params, model_name)
        _, prediction_xgb, scr = hyperopt_test.train_model(X_train, X_test, y_train, params=params, n_fold=n_fold, model_type='cat',
                                                           name='cat', n_estimators=n_estimators, #output_result_csv=True, 
                                                           sample_threshold=3, X_valid=X_valid, y_valid=y_valid)
        submission['time_to_failure'] = prediction_xgb
        print("prediction_cat:")
        print(submission.head())
        print("")
        if context.on_cloud:
            submission.to_csv(context.saving_path+'submission_master2_cat_bestparams_fold'+str(n_fold)+n_est+'_oncloud.csv', 
                                line_terminator=os.linesep)
        else:
            submission.to_csv(context.saving_path+'submission_master2_cat_bestparams_fold'+str(n_fold)+n_est+'.csv')

    X_train = pd.read_csv(os.path.join(OUTPUT_DIR, 'scaled_no_lap_train_X.csv'))
    X_test = pd.read_csv(os.path.join(OUTPUT_DIR, 'scaled_test_X.csv'))
    y_train = pd.read_csv(os.path.join(OUTPUT_DIR, 'no_lap_train_y.csv'))
    X_valid = []
    y_valid = []

    if model_name=='lgb' or model_name==None:
        # output a csv
        submission = pd.read_csv(context.sample_submission_path+'sample_submission.csv', index_col='seg_id')
        params = {'max_depth': 20, 'learning_rate': 0.037491213086246415, 'num_leaves': 1352, 'min_data_in_leaf': 1,
                  'metric': 'rmse', 'lambda_l2': 3.0472684549095574, 'boosting': 'gbdt', 'bagging_fraction': 0.53, 
                  'min_gain_to_split': 1.01, 'lambda_l1': 0.38083001367253844, 'min_data_in_bin': 251, 'objective': 'gamma', 
                  'feature_fraction': 0.99, 'max_bin': 215, 'subsample': 0.6587682468738089}
        params = hyperopt_test.params_preprocessing(params, model_name)
        _, prediction_lgb, scr = hyperopt_test.train_model(X_train, X_test, y_train, params=params, X_valid=X_valid, y_valid=y_valid, 
                                                           n_estimators=n_estimators, n_fold=n_fold)
        submission['time_to_failure'] = prediction_lgb
        print("prediction_lgb:")
        print(submission.head())
        print("")
        if context.on_cloud:
            submission.to_csv(context.saving_path+'submission_master2_lgb_bestparams_fold'+str(n_fold)+'_nooverlap'+n_est+'_oncloud.csv', line_terminator=os.linesep)
        else:
            submission.to_csv(context.saving_path+'submission_master2_lgb_bestparams_fold'+str(n_fold)+'_nooverlap'+n_est+'.csv')

    if model_name=='xgb' or model_name==None:
        # output a csv
        submission = pd.read_csv(context.sample_submission_path+'sample_submission.csv', index_col='seg_id')
        '''
        params = {'boosting': 'gbtree', 'eval_metric' : 'rmse', 'reg_alpha': 0.37824240549600363, 'max_depth': 4, 
                  'colsample_bylevel': 0.26, 'subsample': 0.9500000000000001, 'reg_lambda': 3.2599423926177638, 
                  'objective': 'reg:linear', 'min_child_weight': 2.8828969938855473, 'gamma': 0.2543374108559115, 
                  'colsample_bytree': 0.86, 'colsample_bynode': 0.44, 'tree_method': 'approx', #'learning_rate': 0.00199865978032934}
                  'learning_rate': 0.19986597803293443}
        '''
        params = {'eval_metric': 'mae', 'objective': 'reg:linear', 'subsample': 0.9500000000000001, 'colsample_bytree': 0.72, 
                  'colsample_bynode': 0.9400000000000001, 'reg_lambda': 1.3521810608413789e-05, 'boosting': 'gblinear', 
                  'gamma': 0.46930956951472047, 'colsample_bylevel': 0.18, 'learning_rate': 0.01, 'tree_method': 'approx', 
                  'reg_alpha': 0.010103208706898292, 'min_child_weight': 4.0478106371276725, 'max_depth': 17}
        params = hyperopt_test.params_preprocessing(params, model_name)
        _, prediction_xgb, scr = hyperopt_test.train_model(X_train, X_test, y_train, params=params, model_type='xgb', name='xgb',
                                                           n_estimators=n_estimators, n_fold=n_fold, X_valid=X_valid, y_valid=y_valid)
        submission['time_to_failure'] = prediction_xgb
        print("prediction_xgb:")
        print(submission.head())
        print("")
        if context.on_cloud:
            submission.to_csv(context.saving_path+'submission_master2_xgb_bestparams_fold'+str(n_fold)+'_nooverlap'+n_est+'_oncloud.csv', 
                              line_terminator=os.linesep)
        else:
            submission.to_csv(context.saving_path+'submission_master2_xgb_bestparams_fold'+str(n_fold)+'_nooverlap'+n_est+'.csv')

    if model_name=='cat' or model_name==None:
        submission = pd.read_csv(context.sample_submission_path+'sample_submission.csv', index_col='seg_id')
        params = {'learning_rate': 0.09159686068779108, 'fold_len_multiplier': 2.058197813635219, 'min_data_in_leaf': 8, 
                  'od_type': 'Iter', 'objective': 'MAE', 'bootstrap_type': 'Bernoulli', 'l2_leaf_reg': 2.558997618660658, 
                  'random_strength': 0.034282019666114015, 'verbose': 0, 'od_wait': 25, 'max_bin': 134, 'grow_policy': 'SymmetricTree', 
                  'score_function': 'Correlation', 'task_type': 'GPU', 'depth': 8, 'eval_metric': 'MAE', 
                  'leaf_estimation_backtracking': 'AnyImprovement'}
        params = hyperopt_test.params_preprocessing(params, model_name)
        _, prediction_xgb, scr = hyperopt_test.train_model(X_train, X_test, y_train, params=params, n_fold=n_fold, model_type='cat',
                                                           name='cat', n_estimators=n_estimators, X_valid=X_valid, y_valid=y_valid)
        submission['time_to_failure'] = prediction_xgb
        print("prediction_cat:")
        print(submission.head())
        print("")
        if context.on_cloud:
            submission.to_csv(context.saving_path+'submission_master2_cat_bestparams_fold'+str(n_fold)+'_nooverlap'+n_est+'_oncloud.csv', 
                                line_terminator=os.linesep)
        else:
            submission.to_csv(context.saving_path+'submission_master2_cat_bestparams_fold'+str(n_fold)+'_nooverlap'+n_est+'.csv')

    '''
    # filter features
    print("feature number before filtering = "+str(len(X_train.columns)))
    feature_importance = pd.read_csv(context.saving_path+'feature_importance_master2.csv')
    X_train = feature_filter(X_train, feature_importance, 100)
    X_test = feature_filter(X_test, feature_importance, 100)
    print("feature number after filtering = "+str(len(X_train.columns)))

    import hyperopt
    import threading
    space = {'nu' : hyperopt.hp.uniform('nu', 0.00001, 1),
             'C' : hyperopt.hp.uniform('C', 0.1, 20),
             'kernel' : hyperopt.hp.choice('kernel', ['rbf', 'sigmoid']),
             'degree' : hyperopt.hp.choice('degree', [3, 5, 7, 9, 11]),
             'gamma' : hyperopt.hp.choice('gamma ', ['auto', 'scale']),
             'coef0' : hyperopt.hp.uniform('coef0', -100, 100),
             'shrinking' : hyperopt.hp.choice('shrinking ', [True, False]),
             'tol' : hyperopt.hp.loguniform('learning_rate', np.log(0.005), np.log(0.2))}
    trials = hyperopt.Trials()
    best = hyperopt.fmin(model3, space, hyperopt.tpe.suggest, 200, trials=trials)

    np.save(context.saving_path+'NuSVR_params.npy', best)
    with open(context.saving_path+'best_trials_NuSVR', "wb") as f:
        pickle.dump(trials, f)
    with open(context.saving_path+"logs.txt", 'a') as f:
        print("", file=f)
        print("best_NuSVR:", file=f)
        print(best, file=f)
        print("", file=f)
        print("best_trials_NuSVR:", file=f)
        print(trials, file=f)
        print("", file=f)
    
    params = {'loss_function':'MAE',
              'task_type':'GPU'}
    submission = pd.read_csv(context.sample_submission_path+'sample_submission.csv', index_col='seg_id')
    
    for i in range(3, 9):
        _, prediction, scr = hyperopt_test.train_model(X_train, X_test, y_train, params=params, n_fold=1, model_type='cat',
                                                       name='cat', n_estimators=i*1000, sample_threshold=5)
        print(scr)
        submission['time_to_failure'] = prediction
        print("prediction_cat:")
        print(submission.head())
        print("")
        submission.to_csv(context.saving_path+'master2_output/submission_master2_cat_samplefilte5_%d.csv'%(i*1000,), 
                          line_terminator=os.linesep)

   
    for i in range(3, 9):
        _, prediction, scr = hyperopt_test.train_model(X_train, X_test, y_train, params=params, n_fold=1, model_type='cat',
                                                       name='cat', n_estimators=i*1000, feature_threshold=100)
        print(scr)
        submission['time_to_failure'] = prediction
        print("prediction_cat:")
        print(submission.head())
        print("")
        submission.to_csv(context.saving_path+'master2_output/submission_master2_cat_featurefilte100_%d.csv'%(i*1000,), 
                          line_terminator=os.linesep)
   
    for i in range(3, 9):
        _, prediction, scr = hyperopt_test.train_model(more_X_train, X_test, more_y_train, params=params, n_fold=1, model_type='cat',
                                                       name='cat', n_estimators=i*1000)
        print(scr)
        submission['time_to_failure'] = prediction
        print("prediction_cat:")
        print(submission.head())
        print("")
        submission.to_csv(context.saving_path+'master2_output/submission_master2_cat_moredata_%d.csv'%(i*1000,), 
                          line_terminator=os.linesep)
    '''
    
    '''
    from keras import optimizers
    
    hidden_layer_num = 3  # <8
    params = {} 
    for i in range(hidden_layer_num+1):
        params['Dense'+str(i+1)+'_size'] = 2**(9-i)
        params['Activation'+str(i+1)] = 'relu'
        params['BatchNorm'+str(i+1)] = True
        params['Dropout_rate'+str(i+1)] = 0.5
    params['optimizer'] = optimizers.SGD
    params['learn_rate'] = 0.001
    
    global X_train_scaled, X_test_scaled, y_tr

    X_train_scaled = X_train
    X_test_scaled = X_test
    y_tr = y_train

    hidden_layer_num = 3  # <8
    space = {} 
    for i in range(hidden_layer_num+1):
        space['Dense'+str(i+1)+'_size'] = hyperopt.hp.choice('Dense'+str(i+1)+'_size', [2**(10-i), 2**(9-i)])
        space['Activation'+str(i+1)] = hyperopt.hp.choice('Activation'+str(i+1), ['linear', 'tanh', 'elu', 'relu'])
        space['BatchNorm'+str(i+1)] = hyperopt.hp.choice('BatchNorm'+str(i+1), [True, False])
        space['Dropout_rate'+str(i+1)] = hyperopt.hp.uniform('Dropout_rate'+str(i+1), 0.0, 0.5)
    space['optimizer'] = hyperopt.hp.choice('optimizer', [optimizers.Adam, optimizers.SGD, optimizers.RMSprop])
    space['learn_rate'] = hyperopt.hp.loguniform('learn_rate', np.log(0.0003), np.log(0.2))
    bestNN = hyperopt.fmin(model5, space, hyperopt.tpe.suggest, 200)
    
    with open(context.saving_path+'master2_logs/logs.txt', 'a') as f:
        print(bestNN, file=f)
    np.save(context.saving_path+'master2_logs/bestNN', bestNN)
    '''
    #feature_importance.to_csv(context.saving_path+'feature_importance_master2.csv', line_terminator=os.linesep)
    
    #lgb_base_model()
