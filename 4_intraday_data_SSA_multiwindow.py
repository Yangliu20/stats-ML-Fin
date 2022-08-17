# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:58:37 2020

@author: yangl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft


from pylab import rcParams
#rcParams['figure.figsize'] = 36,12
rcParams['figure.figsize'] = 12,8

import tushare as ts
#df = ts.get_k_data('sh',start='2000-01-01',end='2020-06-16')
#df = ts.get_k_data('sh',start='2000-01-01',end='2020-12-20')
df = ts.get_k_data('sh',start='2000-01-01',end='2020-12-04')
df.index = pd.to_datetime(df.date)
# df.head()
df_pick = df[df.index>pd.to_datetime('20100101')]


series = df_pick.close.tolist()
# series = series - np.mean(series)   # 中心化(不必要)
log_Ret_1 = np.diff(np.log(series))

#%%
df_pick = df_pick[df_pick.index.year != pd.to_datetime('20150101').year]

#%%
def sharpe_ratio(rets, adj = 1):
    rets = np.array(rets)
    return np.sqrt(250.0)* np.mean(rets) / ((np.std(rets)**2/adj)**(1/2))

def freqPlot(inputs):
    plt.figure()
    fft_res = fft(inputs)
    plt.plot(np.abs(fft_res)[1:]/len(inputs),'o-',markersize=4)
    plt.title('frequency domain')

def freqFilter(freq_list, num):
    length = len(freq_list)
    remain_length_half = num
    drop_length = length-2*num-1
    
    return (list(freq_list[:remain_length_half+1])
            +list(np.zeros(drop_length))
            +list(freq_list[-remain_length_half:]))

def reverseState(data, days):
    diff = []
    for i in range(0, days):
        diff.append(data[-i-1]-data[-i-2])
        if len(diff) > 1:
            if diff[-1]*diff[-2] < 0:
                if diff[-1] > 0:
                    return -1
                else:
                    return 1
    return 0

def reverseState_adj(data, days):
    diff = []
    for i in range(0, days):
        diff.append(data[-i-1]-data[-i-2])
    if diff[0]*diff[1] <= 0:
        if diff[0] > 0:
            return 1
        else:
            return -1
    return 0

def reverseState_adj2(data, days, first_order):
    
    stationary = 0
    
    diff_series = np.diff(data)
    
    if diff_series[-1]>diff_series[-2]: # 二阶导>0
        state = 1
    elif diff_series[-1]<diff_series[-2]: # 二阶导<0
        state = -1
    else:
        state = 0
    
    for i in range(1,1 + days):
        if diff_series[-i]*diff_series[-i-1] <= 0:
            stationary = 1
    
    if first_order == True:
        return stationary*state
    else:
        return state

def trade(state, full):
    
    action = 0
    
    if full == 0: 
        if state == 1: # buy
            action = 1
            rtn = 1
            full = 1
        else:
            rtn = 0
    else:
        if state != 1: # sell
            action = -1
            full = 0
            rtn = 0
        else: # hold
            rtn = 1
    
    return full, rtn, action


class SSA:
    def __init__(self, series_train):
        self.series_train = series_train
        self.seriesLen = len(self.series_train)     # 序列长度
    
    def reorder(self, windowLen):
        self.windowLen = windowLen
        self.K = self.seriesLen - self.windowLen + 1
        self.X = np.zeros((self.windowLen, self.K))
        # step1: 嵌入
        for m in range(self.K):
            self.X[:, m] = self.series_train[m:m + self.windowLen]
        
    def svd(self):
        # step2: svd分解， U和sigma已经按升序排序
        self.U, self.sigma, self.VT = np.linalg.svd(self.X, full_matrices=False)
        
    def reconstruct(self, sval_nums):
        # step3: 重组    
        self.sval_nums = sval_nums
        
        X_reconstruct = (self.U[:,0:self.sval_nums]).dot(np.diag(self.sigma[0:self.sval_nums])).dot(self.VT[0:self.sval_nums,:])
        self.X_reconstruct = pd.DataFrame(X_reconstruct)
        
        self.series_recon = np.zeros(self.seriesLen)
        
        for k in range(0,self.seriesLen):
            length = len(range(np.max([0,k-self.X.shape[0]+1]),np.min([self.X.shape[1]-1,k])+1))
            for i in range(np.max([0,k-self.X.shape[0]+1]),np.min([self.X.shape[1]-1,k])+1):
                self.series_recon[k] += self.X_reconstruct.iloc[k-i,i]/length
    
    def plot(self):
        plt.figure()
        plt.grid()
        plt.plot(self.series_train,'--')
        plt.plot(self.series_recon,label='%s components'%(self.sval_nums))
        plt.legend()
        

def trade_by_list(state_seq, log_rtn):
    
    full = 0
    
    state_seq = np.array(state_seq)
    log_rtn = np.array(log_rtn)
    
    rtn_list = []
    trade_list = []
    
    for state_i in state_seq:
        
        full, rtn, action = trade(state_i, full)
        rtn_list.append(rtn)
        trade_list.append(action)
    
    daily_rtn = rtn_list * log_rtn
    
    # return rtn_list, daily_rtn, trade_list
    return daily_rtn
    

#%% SSA

train_length = 300
roll_num = 300

#  1 -> buy
#  0 -> sell
# -1 -> sell

full = 0
daily_rtn = []
state_seq_multiple = []
real_price_ratio = []
trade_seq = []

# windowLen = [10,30,50,60,70]  # 嵌入窗口长度
windowLen = [20,30,40]
col_name = pd.Series(windowLen).apply(lambda x:'windowLength '+str(x))
state_df_single = pd.DataFrame(columns=col_name)
sval_nums = 1

#%%

train_length = 300
roll_num = 600

for num in range(0, roll_num):
    
    series_train = series[num : num + train_length]
    state_list = []
    
    for wl in windowLen:
        ## Special Spectrum Analysis
        model = SSA(series_train)
        model.reorder(wl)
        model.svd()
        model.reconstruct(sval_nums)
        # model.plot()
        
        series_recon = model.series_recon
        
        diff_train = series_train[-1] - series_train[-2]
        diff1 = series_recon[-1] - series_recon[-2]
        # diff2 = series_recon[-2]-series_recon[-3]
        
        # if diff1 > diff2:
        #     state_el = 1
        # elif diff1 < diff2:
        #     state_el = -1
        # else:
        #     state_el = 0
        
        # state_el = np.sign(diff1)
        state_el = np.sign(series_train[-1]-series_recon[-1])
        
        
        state_list.append(state_el)
    
    state_df_single = state_df_single.append(pd.DataFrame([state_list], columns=col_name),
                                             ignore_index=True)
    state = np.sign(np.sum(state_list))
    state_seq_multiple.append(state)
    
    
    real_price_ratio.append(round(series[train_length + num +1-1]/
                                  series[train_length-1],3))
    
    # full, rtn, action = trade(state, full)
    # rtn *= log_Ret_1[num+train_length]
    # trade_seq.append(action)
    # daily_rtn.append(rtn)

log_rtn_trade = log_Ret_1[train_length: train_length + roll_num] # 明天-今天
Date_pred = df_pick.index[train_length: train_length + roll_num]
close_price = df_pick.close[train_length: train_length + roll_num].reset_index()

# for col in state_df_single.columns:
#     plt.figure()
#     plt.plot(close_price.date, close_price.close)
#     plt.plot(close_price[state_df_single[col]==1].date, 
#              close_price[state_df_single[col]==1].close, 'o', label = 'up') # up
#     plt.plot(close_price[state_df_single[col]==-1].date,
#              close_price[state_df_single[col]==-1].close, 's', label = 'down') # down
#     plt.title(col)
#     plt.legend()

#%%

# daily_rtn_single = state_df_single.apply(lambda x: trade_by_list(x, log_rtn_trade))
# daily_rtn_multiple = trade_by_list(state_seq_multiple, log_rtn_trade)

daily_rtn_single = state_df_single.apply(lambda x: x*log_rtn_trade)
daily_rtn_multiple = state_seq_multiple*log_rtn_trade


for col in daily_rtn_single.columns:
    plt.plot(Date_pred, np.exp(daily_rtn_single[col].cumsum()), label=col)
plt.plot(Date_pred, np.exp(daily_rtn_multiple.cumsum()), label='multiple')
plt.plot(Date_pred, real_price_ratio,'--',label='market')
plt.legend()



# # trade result
# trade_seq.append(0)

# buy_point = (np.array(trade_seq)==1).tolist()
# sell_point = (np.array(trade_seq)==-1).tolist()

# plt.figure()
# plt.grid()
# plt.plot(np.arange(roll_num+1),np.exp(np.array(daily_rtn).cumsum()),'-')
# plt.plot(np.arange(roll_num+1)[buy_point],np.exp(np.array(daily_rtn).cumsum())[buy_point],'s',c='orange',label='buy')
# plt.plot(np.arange(roll_num+1)[sell_point],np.exp(np.array(daily_rtn).cumsum())[sell_point],'o',c='green',label='sell')
# plt.plot(np.arange(roll_num+1),real_price_ratio,'--')
# plt.legend()
