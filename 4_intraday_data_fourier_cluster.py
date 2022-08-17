# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 12:55:20 2020

@author: yangl
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
import datetime
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.mixture import GaussianMixture

import itertools
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from pylab import rcParams
rcParams['figure.figsize'] = 12,8



path = r'C:/Users/yangl/Documents/Python/Spectrum_Analysis/'


# data = pd.read_csv(path+'000016_2007_2020.csv')
# data.date = data.date.apply(lambda x: pd.to_datetime(x).date())
# data.time = data.time.apply(lambda x: pd.to_datetime(x.split(' ')[-1]).time())
# data.to_pickle(path+'000016_2007_2020.pkl')


data = pd.read_pickle(path+'000016_2007_2020.pkl')
calendar = data.date.unique()
calendar = pd.Series(calendar)

aftn_open = np.array(data[data.time == datetime.time(13,1)]['close'])
aftn_close = np.array(data[data.time == datetime.time(15,0)]['close'])
log_Ret = np.log(aftn_close) - np.log(aftn_open)
log_Ret = pd.Series(data = log_Ret, index = calendar)

daily_close = pd.Series(data = aftn_close, index = calendar)


#%%
def sharpe_ratio(rets, adj = 1):
    rets = np.array(rets)
    return np.sqrt(250.0)* np.mean(rets) / ((np.std(rets)**2/adj)**(1/2))

def freqDomain(inputs, plotBool=False):
    # 傅里叶变换
    inputs = list(inputs)
    fft_res = fft(inputs)
    if plotBool:
        plt.figure()
        plt.grid()
        plt.plot(np.abs(fft_res)[1:]/len(inputs),'o-',markersize=4)
        plt.title('frequency domain')
    return fft_res

def freqFilter(freq_list, num):
    # 筛选前num个非零频率
    length = len(freq_list)
    remain_length_half = num
    drop_length = length-2*num-1
    
    return (list(freq_list[:remain_length_half+1])
            +list(np.zeros(drop_length))
            +list(freq_list[-remain_length_half:]))

def ave_corr(df):
    corr_matrix = df.corr()
    size = len(corr_matrix)
    
    s = np.sum(np.sum(corr_matrix))
    s -= size
    num = size*(size-1)
    
    return round(s/num, 3)

def matrix_calculate(df, target):
    
    res = np.zeros((len(df.columns), len(df.columns)))
    for i in range(len(df.columns)):
        for j in range(i,len(df.columns)):
            x = np.array(df.iloc[:,i])
            y = np.array(df.iloc[:,j])
            if target=='cosine':
                res[i][j] = x.dot(y)/np.sqrt((x.dot(x))*(y.dot(y)))
            elif target=='euclidean':
                res[i][j] = np.sum(np.abs(x-y))
                # res[i][j] = euclidean(x,y) # np.sqrt(np.sum((x-y)**2))
            elif target=='dtw':
                distance, path = fastdtw(x, y, dist=euclidean)
                res[i][j] = distance
            if i!=j:
                res[j][i] = res[i][j]
    return res

def ave_coef(df, target):
    
    matrix = matrix_calculate(df, target)
    size = len(matrix)
    
    s = np.sum(np.sum(matrix))
    s -= np.sum(np.diag(matrix))
    num = size*(size-1)
    
    return round(s/num, 3)

class clusterAnalysis:
    # 对某一个cluster
    def __init__(self, inputs, keys, keylabel, normalize_factor, const_factor, drift, eps, 
                 period, predict_date):
        self.data = inputs
        self.date_keys = keys
        self.label = keylabel
        self.normalize_factor = normalize_factor # 振幅的归一系数
        self.const_factor = const_factor
        self.drift = drift
        self.eps = eps
        self.period = period # train样本点数
        self.predict_date = predict_date
        self.scaledClosePrice = pd.DataFrame()
        self.scaledClosePrice_withDrift = pd.DataFrame()
    
    def scale(self):
        # 对cluster中的每一天的价格做归一化
        for k in self.date_keys:
            close_price = np.array(self.data[self.data['date']==k]['close'])
            abs_factor = self.normalize_factor[k]
            constant = self.const_factor[k]/self.period
            trend = self.drift[k]*np.arange(self.period*2)
            close_price_scale = (close_price - trend - constant)/abs_factor
            self.scaledClosePrice[k] = close_price_scale # 减去了上午的drift
            self.scaledClosePrice_withDrift[k] = (close_price - constant)/abs_factor # 保留drift
        
        self.morning_drift_scaled = self.drift[date_keys]/self.normalize_factor[date_keys] # 上午的drift归一
        
    def corr(self):
        # correlations of scaled close prices, morning and afternoon
        self.corr_train = ave_corr(self.scaledClosePrice.iloc[:self.period,:])
        self.corr_pred = ave_corr(self.scaledClosePrice.iloc[self.period:self.period*2,:])
    
    def euclid(self):
        # euclidean distances of scaled close prices, morning and afternoon
        self.euclid_train = ave_coef(self.scaledClosePrice.iloc[:self.period,:], 'euclidean')
        self.euclid_pred = ave_coef(self.scaledClosePrice.iloc[self.period:self.period*2,:], 'euclidean')
    
    def trending(self):
        
        # delta drift 下午-上午
        self.aftn_trend = (self.scaledClosePrice.iloc[self.period*2-1,:] 
                           - self.scaledClosePrice.iloc[self.period,:])/(self.period-1)
        
        if self.predict_date in self.aftn_trend.index:
            # print(self.aftn_trend.index)
            self.true_trend_normalize = self.aftn_trend[self.predict_date]
            self.aftn_trend.drop(self.predict_date, inplace=True)
            # print(self.aftn_trend.index)
        
        self.expectation = np.mean(self.aftn_trend)
        
    def decay(self, half_life):
        # decay
        self.half_life = half_life
        self.rate = 0.5**(1/self.half_life)
        self.aftn_trend_decay = self.aftn_trend.copy()
        for i in self.aftn_trend_decay.index:
            delta = (self.predict_date - i).days
            self.aftn_trend_decay[i] *= (self.rate)**delta
    
    def filter_aftn_trend(self, nums):
        # mean variance
        sampleM = np.mean(self.aftn_trend)
        sampleSTD = np.std(self.aftn_trend)
        lower = sampleM - nums*sampleSTD
        upper = sampleM + nums*sampleSTD
        self.aftn_trend_adj = self.aftn_trend.copy()[(self.aftn_trend > lower) & (self.aftn_trend < upper)]
    

    def plot(self, driftBool):
        plt.figure()
        for col in self.scaledClosePrice.columns:
            if driftBool==False:
                close_price_scale = self.scaledClosePrice[col]
            else:
                close_price_scale = self.scaledClosePrice_withDrift[col]
            if self.morning_drift_scaled[col] > 0:
                ls = '-'
            else:
                ls = '--'
            plt.plot(close_price_scale[:self.period*2], label='%s'%(col), linestyle=ls)
            plt.axhline(y=0, color='slategrey', linestyle='--')
            plt.axvline(x=self.period, color='slategrey', linestyle='--')
        plt.legend()
        plt.title('eps:%s  No.%s  corr:%s'%(self.eps, self.label, self.corr_train))
    
    
def freqRestore(freq_list_half, overall_len):
    len_zero = overall_len-2*len(freq_list_half)+1
    return (list(freq_list_half)+list(np.zeros(len_zero))
            +list(np.conjugate(freq_list_half[::-1])[:-1]))


def normalize(series, var):
    
    mean = np.mean(series)
    std = np.std(series)
    
    return (series-mean)/std*np.sqrt(var)


def transform(df_complex, drop_list, period):
    
    # 振幅归一
    
    df_trans = df_complex.drop(drop_list, axis=1)
    const = df_trans[0].apply(np.real)
    df_trans.drop([0], axis=1, inplace=True)
    
    abs_max = df_trans.apply(lambda x: np.max(np.abs(x)), axis=1) # 最大振幅
    df_trans = df_trans[abs_max!=0]
    abs_max = abs_max[df_trans.index]
    df_trans = df_trans.apply(lambda x: x/abs_max[x.name], axis=1)    
    
    freq_nums = len(df_trans.columns)
    
    df_real = pd.DataFrame(np.real(df_trans), index=df_trans.index, columns=np.arange(freq_nums))
    df_imag = pd.DataFrame(np.imag(df_trans), index=df_trans.index, columns=np.arange(freq_nums,freq_nums*2))
    df_trans = pd.merge(df_real, df_imag, left_index=True, right_index=True)
    
    return df_trans, const, abs_max    
    

def rescale(df_trans, times):
    
    # 改变高频分量权重
    # 线性
    
    nums = int(len(df_trans.columns)/2)
    step = (times-1)/(nums-1)
    for i in range(1, nums):
        df_trans.iloc[:,i] *= (1+step*i)
        df_trans.iloc[:,i+nums] *= (1+step*i)
    
    return df_trans

#%% Data

train_length = 250*4 # calendar day
start_date = datetime.date(2010,1,4)
start_index = calendar[calendar == start_date].index[0]
end_date = start_date + datetime.timedelta(days=train_length)

start_time = datetime.time(9,30) # exclusive
end_time = datetime.time(11,30) # inclusive
interval_points = int((datetime.datetime.combine(start_date, end_time)
                       - datetime.datetime.combine(start_date, start_time)).seconds/60)

freq_num = 7

state_seq = []
state_seq_adj = []
state_seq_adj2 = []
state_seq_adj3 = []
state_seq_decay = []

pred_date_list = []
df_complex = pd.DataFrame(columns=np.arange(freq_num+1).tolist()+['coeff'])

summary_col = ['date','true state','delta drift normalized','drift mrng normalized',
               'expectation','adj expectation','decay expectation','cluster samples',
               'corr train','aftn trend series','morning drift scaled series']
summary = pd.DataFrame(columns=summary_col)

#%%
for rolling in range(0, 1800):
    
    ### FFT
    
    if len(df_complex)==0:
    #     for num in range(train_length):
    #         # get data in one hour
    #         day = calendar[start_index + num]
    #         X = data[(data.date == day) & ((data.time > start_time) & (data.time <= end_time))]['close']
    #         X = np.array(X)
    #         if np.sum(np.abs(np.diff(X)[-10:]))==0:
    #             continue
    #         coeff = (X[-1]-X[0])/(len(X)-1)
    #         line_fit = [i*coeff for i in np.arange(len(X))]
    #         X_resid = np.array(X) - np.array(line_fit)
            
    #         # FFT
    #         fft_res = freqDomain(X_resid, plotBool = False)
    #         # fft_select = freqFilter(fft_res, 4)
    #         # plt.figure()
    #         # plt.plot(X_resid)
    #         # plt.plot(np.real(ifft(fft_select)))
            
    #         freq_pick = fft_res[:freq_num + 1] # plus one constant component
    #         df_complex = df_complex.append(pd.DataFrame([freq_pick.tolist()+[coeff]], index=[day], columns=np.arange(freq_num+1).tolist()+['coeff']),
    #                                         ignore_index=False)
        
        
        df_complex = pd.read_pickle(path+'2010_2016_freqPick7_unnormalize_drift.pkl')
        df_complex = df_complex[:train_length]
        drift_coeff = df_complex.copy().coeff
        df_complex.drop(['coeff'], axis=1, inplace=True)
        
    else:
        df_complex.drop(df_complex.index[0], inplace=True)
        drift_coeff = drift_coeff[1:]
        day = calendar[start_index + rolling + train_length-1]
        X = data[(data.date == day) & ((data.time > start_time) & (data.time <= end_time))]['close']
        X = np.array(X)
        
        if len(X)==0:
            print('No data for %s'%(day))
            continue
        
        if np.sum(np.abs(np.diff(X)[-10:]))==0:
            continue
        
        coeff = (X[-1]-X[0])/(len(X)-1)
        line_fit = [i*coeff for i in np.arange(len(X))]
        X_resid = np.array(X) - np.array(line_fit)
        
        # FFT
        fft_res = freqDomain(X_resid, plotBool = False)    
        freq_pick = fft_res[:freq_num + 1] # plus one constant component
        df_complex = df_complex.append(pd.DataFrame([freq_pick.tolist()], index=[day], columns=np.arange(freq_num+1).tolist()),
                                       ignore_index=False)
        drift_coeff = drift_coeff.append(pd.Series({day:coeff}))
    
    
    ### Transform
    
    df_trans, const, abs_max = transform(df_complex, [6, 7], interval_points) # 取前五个频率
    # df_trans = rescale(df_trans, 0.1) # 改变高频分量的权重
    pred_date = df_trans.index[-1]
    pred_date_list.append(pred_date)
    X = df_trans.copy()
    
    
    ### Cluster
    
    ## choose max_eps
    
    # eps_list = np.arange(0.7,1.0,0.1)
    # score_list = []
    # # eps = 0.4
    
    # for eps in eps_list:
    #     model = OPTICS(min_samples=4, max_eps=eps)
    #     model.fit(X)
    #     labels = model.labels_
    #     label_pred = labels[-1]
        
    #     counts = 0
    #     # expect = 0
    #     # expect_adj = 0
    #     # aftn_trend_base = []
        
    #     for label in pd.Series(pd.Series(labels).unique()).sort_values().tolist():
    #         if label==-1:
    #             continue
    #         date_keys = X.index[labels==label]
    
    #         clusterRes = clusterAnalysis(data, date_keys, label, abs_max, const, drift_coeff, eps, 
    #                                       interval_points, pred_date)
    #         clusterRes.scale()
    #         clusterRes.corr()
    #         # clusterRes.trending()
    #         # counts += (clusterRes.corr_train > 0.85)*len(date_keys)
    #         counts += (clusterRes.corr_train > 0.85)
    #         clusterRes.plot()
            
    #         # if label==label_pred:
    #         #     clusterRes.plot()
    #         #     clusterRes.filter_aftn_trend(1.5)
    #         #     expect = clusterRes.expectation
    #         #     expect_adj = np.mean(clusterRes.aftn_trend_adj)
    #         #     aftn_trend_base = clusterRes.aftn_trend
    #         #     print(clusterRes.aftn_trend)
    #         #     print('expectation:',expect)
    #         #     print('adj expectation:',expect_adj)
    #         #     print('drift normalized morning:',drift_coeff[pred_date]/abs_max[pred_date])
        
    #     score_list.append(counts/(len(pd.Series(labels).unique())-(-1 in pd.Series(labels).unique())))
    #     print('score:', score_list[-1])
        
    # eps_pick = eps_list[np.argmax(score_list)]
    
    eps_pick = 0.5
    
    ##
    
    model = OPTICS(min_samples=4, max_eps=eps_pick)
    model.fit(X)
    labels = model.labels_
    label_pred = labels[-1]
    # print('Date:%s  Label:%s  EPS:%s  Score:%s'%(pred_date, label_pred, eps_pick, np.max(score_list)))
    print('Date:%s  Label:%s  EPS:%s'%(pred_date, label_pred, eps_pick))
    
    
    if label_pred==-1:
        state = 0
        state_adj = 0
        state_adj2 = 0
        state_adj3 = 0
        state_decay = 0
    
    else:
        date_keys = X.index[labels==label_pred]
        clusterRes = clusterAnalysis(data, date_keys, label_pred, abs_max, const, drift_coeff, eps_pick, 
                                     interval_points, pred_date)
        clusterRes.scale()
        clusterRes.corr()
        clusterRes.trending()
        
        # 直接取mean
        expect = clusterRes.expectation
        aftn_trend_base = clusterRes.aftn_trend
        
        clusterRes.plot(True) # 作图，保留上午的drift
        # clusterRes.plot(False) # 去除上午的drift
        
        # filter
        clusterRes.filter_aftn_trend(1.5)
        expect_adj = np.mean(clusterRes.aftn_trend_adj)
        
        # decay
        clusterRes.decay(365*4)
        expect_decay = np.mean(clusterRes.aftn_trend_decay)
        
        print(clusterRes.aftn_trend)
        print('expectation:',expect)
        print('adj expectation:',expect_adj)
        print('expectation decay:',expect_decay)
        print('drift normalized morning:',drift_coeff[pred_date]/abs_max[pred_date])
    
        # mean, filter, decay
        state = np.sign(drift_coeff[pred_date]/abs_max[pred_date] + expect)
        state_adj = np.sign(drift_coeff[pred_date]/abs_max[pred_date] + expect_adj)
        state_decay = np.sign(drift_coeff[pred_date]/abs_max[pred_date] + expect_decay)
        
        # quantile
        total_drift_list = aftn_trend_base + drift_coeff[pred_date]/abs_max[pred_date]
        if np.sum(total_drift_list > 0)/len(total_drift_list) > 0.7:
            state_adj2 = 1
        elif np.sum(total_drift_list < 0)/len(total_drift_list) > 0.7:
            state_adj2 = -1
        else:
            state_adj2 = 0
            
            
        # decay + quantile
        
        total_drift_list_decay = clusterRes.aftn_trend_decay + drift_coeff[pred_date]/abs_max[pred_date]
        if np.sum(total_drift_list_decay > 0)/len(total_drift_list_decay) > 0.7:
            state_adj3 = 1
        elif np.sum(total_drift_list_decay < 0)/len(total_drift_list_decay) > 0.7:
            state_adj3 = -1
        else:
            state_adj3 = 0
        
            
        
        print('state:%s  state_adj:%s  state_adj2:%s  state_adj3:%s  decay:%s'%(state, state_adj, state_adj2, state_adj3, state_decay))
        print('True State:',np.sign(log_Ret[pred_date]))
        print()
        
        summary = summary.append(pd.DataFrame([[pred_date, np.sign(log_Ret[pred_date]),
                                               clusterRes.true_trend_normalize, 
                                               drift_coeff[pred_date]/abs_max[pred_date],
                                               expect, expect_adj, expect_decay, 
                                               len(aftn_trend_base), clusterRes.corr_train, 
                                               aftn_trend_base, clusterRes.morning_drift_scaled]],columns=summary_col),ignore_index=True)
        
    state_seq.append(state)
    state_seq_adj.append(state_adj)
    state_seq_adj2.append(state_adj2)
    state_seq_adj3.append(state_adj3)
    state_seq_decay.append(state_decay)
    
    

#%%

real_price_ratio = daily_close[pred_date_list]
real_price_ratio = real_price_ratio/real_price_ratio[0]

#%%
daily_rtn = log_Ret[pred_date_list]*state_seq # mean
daily_rtn_adj = log_Ret[pred_date_list]*state_seq_adj # mean variance
daily_rtn_adj2 = log_Ret[pred_date_list]*state_seq_adj2 # quantile
daily_rtn_decay = log_Ret[pred_date_list]*state_seq_decay # decay
daily_rtn_adj3 = log_Ret[pred_date_list]*state_seq_adj3 # decay + quantile


plt.figure()
plt.grid()
plt.plot(np.exp(daily_rtn.cumsum()),'-',label='mean')
plt.plot(np.exp(daily_rtn_adj.cumsum()),'-',label='1.5 sigma')
plt.plot(np.exp(daily_rtn_adj2.cumsum()),'-',label='70% quantile')
plt.plot(np.exp(daily_rtn_adj3.cumsum()),'-',label='decay+70% quantile')#'+dropMaxMin')
plt.plot(np.exp(daily_rtn_decay.cumsum()),'-',label='decay')

# plt.plot(np.exp(daily_rtn.cumsum())[np.array(state_seq_adj)==1],'s',label='long')
# plt.plot(np.exp(daily_rtn.cumsum())[np.array(state_seq_adj)==-1],'o',label='short')
# # plt.plot(real_price_ratio,'--',label='market daily')
plt.axhline(1,linestyle='--',color='blue',linewidth=2)
plt.legend()
