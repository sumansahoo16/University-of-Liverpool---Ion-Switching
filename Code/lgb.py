"""
Created on Sat Mar 28 2020
@author: sumansahoo16
Public Score : 0.940 (Macro F1)
"""

import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats,signal
from functools import partial
from pykalman import KalmanFilter
from sklearn.model_selection import train_test_split
from sklearn.metrics import  f1_score
import lightgbm as lgb
import gc


def reduce_mem_usage(df, verbose=True):
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2  
    
    for col in df.columns:
        if col!='open_channels':
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)    
                        
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

target = pd.read_csv('/liverpool-ion-switching/train.csv',usecols=['open_channels'], dtype= {'open_channels' : np.int8})

# signal processing features
def calc_gradients(s, n_grads = 4):
    
    grads = pd.DataFrame()
    
    g = s.values
    for i in range(n_grads):
        g = np.gradient(g)
        grads['grad_' + str(i+1)] = g
        
    return grads

def calc_low_pass(s, n_filts=10):

    wns = np.logspace(-2, -0.3, n_filts)
    
    low_pass = pd.DataFrame()
    x = s.values
    for wn in wns:
        b, a = signal.butter(1, Wn=wn, btype='low')
        zi = signal.lfilter_zi(b, a)
        low_pass['lowpass_lf_' + str('%.4f' %wn)] = signal.lfilter(b, a, x, zi=zi*x[0])[0]
        low_pass['lowpass_ff_' + str('%.4f' %wn)] = signal.filtfilt(b, a, x)
        
    return low_pass

def calc_high_pass(s, n_filts=10):

    wns = np.logspace(-2, -0.1, n_filts)
    
    high_pass = pd.DataFrame()
    x = s.values
    for wn in wns:
        b, a = signal.butter(1, Wn=wn, btype='high')
        zi = signal.lfilter_zi(b, a)
        high_pass['highpass_lf_' + str('%.4f' %wn)] = signal.lfilter(b, a, x, zi=zi*x[0])[0]
        high_pass['highpass_ff_' + str('%.4f' %wn)] = signal.filtfilt(b, a, x)
        
    return high_pass

def calc_ewm(s, windows=[10, 50, 100, 500, 1000]):

    ewm = pd.DataFrame()
    for w in windows:
        ewm['ewm_mean_' + str(w)] = s.ewm(span=w, min_periods=1).mean()
        ewm['ewm_std_' + str(w)] = s.ewm(span=w, min_periods=1).std()
        
    # add zeros when na values (std)
    ewm = ewm.fillna(value=0)
        
    return ewm


def add_features(s):
    
    gradients = calc_gradients(s)
    low_pass = calc_low_pass(s)
    high_pass = calc_high_pass(s)
    ewm = calc_ewm(s)
    
    return pd.concat([s, gradients, low_pass, high_pass, ewm], axis=1)


def divide_and_add_features(s, signal_size=500000):
    # normalize
    s = s / 15.0
    
    ls = []
    for i in tqdm(range(int(s.shape[0]/signal_size))):
        sig = s[i*signal_size:(i+1)*signal_size].copy().reset_index(drop=True)
        sig_featured = add_features(sig)
        ls.append(sig_featured)
    
    return pd.concat(ls, axis=0)

# rolling and aggreagate batch features
def rolling_features(train, test):
    
    pre_train = train.copy()
    pre_test = test.copy()
    
        
    for df in [pre_train, pre_test]:
        
        df['lag_t1'] = df.groupby('batch')['signal'].transform(lambda x: x.shift(1))
        df['lag_t2'] = df.groupby('batch')['signal'].transform(lambda x: x.shift(2))
        df['lag_t3'] = df.groupby('batch')['signal'].transform(lambda x: x.shift(3))
        
        df['lead_t1'] = df.groupby('batch')['signal'].transform(lambda x: x.shift(-1))
        df['lead_t2'] = df.groupby('batch')['signal'].transform(lambda x: x.shift(-2))
        df['lead_t3'] = df.groupby('batch')['signal'].transform(lambda x: x.shift(-3))
                
        for window in [1000, 5000, 10000, 20000, 40000, 80000]:
            
            # roll backwards
            df['signalmean_t' + str(window)] = df.groupby(['batch'])['signal'].transform(lambda x: x.shift(1).rolling(window).mean())
            df['signalstd_t' + str(window)] = df.groupby(['batch'])['signal'].transform(lambda x: x.shift(1).rolling(window).std())
            df['signalvar_t' + str(window)] = df.groupby(['batch'])['signal'].transform(lambda x: x.shift(1).rolling(window).var())
            df['signalmin_t' + str(window)] = df.groupby(['batch'])['signal'].transform(lambda x: x.shift(1).rolling(window).min())
            df['signalmax_t' + str(window)] = df.groupby(['batch'])['signal'].transform(lambda x: x.shift(1).rolling(window).max())
            min_max = (df['signal'] - df['signalmin_t' + str(window)]) / (df['signalmax_t' + str(window)] - df['signalmin_t' + str(window)])
            df['norm_t' + str(window)] = min_max * (np.floor(df['signalmax_t' + str(window)]) - np.ceil(df['signalmin_t' + str(window)]))
            
            # roll forward
            df['signalmean_t' + str(window) + '_lead'] = df.groupby(['batch'])['signal'].transform(lambda x: x.shift(- window - 1).rolling(window).mean())
            df['signalstd_t' + str(window) + '_lead'] = df.groupby(['batch'])['signal'].transform(lambda x: x.shift(- window - 1).rolling(window).std())
            df['signalvar_t' + str(window) + '_lead'] = df.groupby(['batch'])['signal'].transform(lambda x: x.shift(- window - 1).rolling(window).var())
            df['signalmin_t' + str(window) + '_lead'] = df.groupby(['batch'])['signal'].transform(lambda x: x.shift(- window - 1).rolling(window).min())
            df['signalmax_t' + str(window) + '_lead'] = df.groupby(['batch'])['signal'].transform(lambda x: x.shift(- window - 1).rolling(window).max())   
            min_max = (df['signal'] - df['signalmin_t' + str(window) + '_lead']) / (df['signalmax_t' + str(window) + '_lead'] - df['signalmin_t' + str(window) + '_lead'])
            df['norm_t' + str(window) + '_lead'] = min_max * (np.floor(df['signalmax_t' + str(window) + '_lead']) - np.ceil(df['signalmin_t' + str(window) + '_lead']))
            
    del train, test, min_max
    
    return pre_train, pre_test

def static_batch_features(df, n):
    
    df = df.copy()
    df.drop('batch', inplace = True, axis = 1)
    df = df.sort_values(by=['time']).reset_index(drop=True)
    df.index = ((df.time * 10000) - 1).values
    df['batch_' + str(n)] = df.index // n
    df['batch_index_' + str(n)] = df.index  - (df['batch_' + str(n)] * n)
    df['batch_slices_' + str(n)] = df['batch_index_' + str(n)]  // (n / 10)
    df['batch_slices2_' + str(n)] = df.apply(lambda r: '_'.join([str(r['batch_' + str(n)]).zfill(3), str(r['batch_slices_' + str(n)]).zfill(3)]), axis=1)

    for c in ['batch_' + str(n), 'batch_slices2_' + str(n)]:
        d = {}
        d['mean' + c] = df.groupby([c])['signal'].mean()
        d['median' + c] = df.groupby([c])['signal'].median()
        d['max' + c] = df.groupby([c])['signal'].max()
        d['min' + c] = df.groupby([c])['signal'].min()
        d['std' + c] = df.groupby([c])['signal'].std()
        d['p10' + c] = df.groupby([c])['signal'].apply(lambda x: np.percentile(x, 10))
        d['p25' + c] = df.groupby([c])['signal'].apply(lambda x: np.percentile(x, 25))
        d['p75' + c] = df.groupby([c])['signal'].apply(lambda x: np.percentile(x, 75))
        d['p90' + c] = df.groupby([c])['signal'].apply(lambda x: np.percentile(x, 90))
        d['skew' + c] = df.groupby([c])['signal'].apply(lambda x: pd.Series(x).skew())
        d['kurtosis' + c] = df.groupby([c])['signal'].apply(lambda x: pd.Series(x).kurtosis())
        min_max = (d['mean' + c] - d['min' + c]) / (d['max' + c] - d['min' + c])
        d['norm' + c] = min_max * (np.floor(d['max' + c]) - np.ceil(d['min' + c]))
        d['mean_abs_chg' + c] = df.groupby([c])['signal'].apply(lambda x: np.mean(np.abs(np.diff(x))))
        d['abs_max' + c] = df.groupby([c])['signal'].apply(lambda x: np.max(np.abs(x)))
        d['abs_min' + c] = df.groupby([c])['signal'].apply(lambda x: np.min(np.abs(x)))
        d['range' + c] = d['max' + c] - d['min' + c]
        d['maxtomin' + c] = d['max' + c] / d['min' + c]
        d['abs_avg' + c] = (d['abs_min' + c] + d['abs_max' + c]) / 2
        for v in d:
            df[v] = df[c].map(d[v].to_dict())

    for c in [c1 for c1 in df.columns if c1 not in ['time', 'signal', 'open_channels', 'batch', 'batch_' + str(n), 
                                                    'batch_index_' + str(n), 'batch_slices_' + str(n), 
                                                    'batch_slices2_' + str(n)]]:
        df[c + '_msignal'] = df[c] - df['signal']
        
    df.reset_index(drop = True, inplace = True)
        
    return df

def Kalman1D(observations,damping=1):
    observation_covariance = damping
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = 0.1
    initial_value_guess
    kf = KalmanFilter(
            initial_state_mean=initial_value_guess,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix
        )
    pred_state, state_cov = kf.smooth(observations)
    return pred_state



train, test = read_data()


observation_covariance = .0015
train['signal'] = Kalman1D(train.signal.values,observation_covariance)
pre_train4 = divide_and_add_features(train['signal'])
pre_train4.drop(['signal'], inplace = True, axis = 1)
pre_train4.reset_index(inplace = True, drop = True)
pre_train4 = reduce_mem_usage(pre_train4)



train, test = get_batch(train, test)
pre_train1, pre_test1 = rolling_features(train, test)
pre_train1 = reduce_mem_usage(pre_train1)
pre_train2 = static_batch_features(train, 25000)
pre_train2 = reduce_mem_usage(pre_train2)

del train, test
gc.collect()


feat2 = [col for col in pre_train2.columns if col not in ['open_channels', 'signal', 'time', 'batch_25000', 
                                                          'batch_index_25000', 'batch_slices_25000', 'batch_slices2_25000']]

train = pd.concat([pre_train1, pre_train2[feat2], pre_train4], axis = 1)

del pre_train1, pre_train2, pre_train4

features = [col for col in train.columns if col not in ['open_channels', 'time', 'batch']]

print('Training with {} features'.format(len(features)))

def MacroF1Metric(preds, dtrain):
    labels = dtrain.get_label()
    preds = np.round(np.clip(preds, 0, 10)).astype(int)
    score = f1_score(labels, preds, average = 'macro')
    return ('MacroF1Metric', score, True)

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=16)


params = {'objective': 'regression',
          'metric': 'rmse',
          'learning_rate': 0.01,
          'lambda_l1': 6.708910037225561,
          'lambda_l2': 0.00035526711387904295,
          'num_leaves': 239,
          'feature_fraction': 0.7,
          'bagging_fraction': 1.0,
          'bagging_freq': 0,
          'min_child_samples': 20,
          'random_state': 16, 
          'n_jobs':-1 }

model = lgb.train(params,
                  train_set  = lgb.Dataset(X_train, y_train),
                  valid_sets = lgb.Dataset(X_test, y_test),
                  valid_names = ['test'],
                  num_boost_round  = 10000,
                  verbose_eval = 100,
                  early_stopping_rounds = 200,
                  feval = MacroF1Metric)

y_pred = model.predict(X_test, num_iteration= model.best_iteration)
f1_sc = f1_score(y_test,  np.round(np.clip(y_pred, 0, 10)), average='macro')
print(" f1_score : ", f1_sc)



class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _f1_loss(self, coef, X, y):
        
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        return -f1_score(y, X_p, average = 'macro')

    def fit(self, X, y):
        
        loss_partial = partial(self._f1_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def coefficients(self):
        return self.coef_['x']
    
    
train_preds = model.predict(train, num_iteration= model.best_iteration)
optR = OptimizedRounder()
optR.fit(train_preds.reshape(-1,), target)
coefficients = optR.coefficients()
print(coefficients)


def optimize_prediction(prediction):
    
    prediction[prediction <= coefficients[0]] = 0
    prediction[np.where(np.logical_and(prediction > coefficients[0], prediction <= coefficients[1]))] = 1
    prediction[np.where(np.logical_and(prediction > coefficients[1], prediction <= coefficients[2]))] = 2
    prediction[np.where(np.logical_and(prediction > coefficients[2], prediction <= coefficients[3]))] = 3
    prediction[np.where(np.logical_and(prediction > coefficients[3], prediction <= coefficients[4]))] = 4
    prediction[np.where(np.logical_and(prediction > coefficients[4], prediction <= coefficients[5]))] = 5
    prediction[np.where(np.logical_and(prediction > coefficients[5], prediction <= coefficients[6]))] = 6
    prediction[np.where(np.logical_and(prediction > coefficients[6], prediction <= coefficients[7]))] = 7
    prediction[np.where(np.logical_and(prediction > coefficients[7], prediction <= coefficients[8]))] = 8
    prediction[np.where(np.logical_and(prediction > coefficients[8], prediction <= coefficients[9]))] = 9
    prediction[prediction > coefficients[9]] = 10
    
    return prediction


test['signal'] = Kalman1D(test.signal.values,observation_covariance)
pre_train4 = divide_and_add_features(train['signal'])
pre_test4 = divide_and_add_features(test['signal'])
pre_train4.drop(['signal'], inplace = True, axis = 1)
pre_test4.drop(['signal'], inplace = True, axis = 1)
pre_train4.reset_index(inplace = True, drop = True)
pre_test4.reset_index(inplace = True, drop = True)
pre_train4 = reduce_mem_usage(pre_train4)
pre_test4 = reduce_mem_usage(pre_test4)



pre_test1 = reduce_mem_usage(pre_test1)
pre_test2 = static_batch_features(test, 25000)
pre_test2 = reduce_mem_usage(pre_test2)
test = pd.concat([pre_test1, pre_test2[feat2], pre_test4], axis = 1)
del pre_train1, pre_train2, pre_train4, pre_test1, pre_test2, pre_test4


y_pred = model.predict(test, num_iteration= model.best_iteration)
y_pred_op = optimize_prediction(y_pred).astype(int)


submission = pd.read_csv('/liverpool-ion-switching/sample_submission.csv')
submission['open_channels'] = y_pred_op
submission.to_csv('submission.csv', index=False, float_format='%.4f')
