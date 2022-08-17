# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 09:03:38 2020

@author: yangl
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture
import tushare as ts
from pylab import rcParams
rcParams['figure.figsize'] = 12,8


#%%
def sharpe_ratio(rets, adj = 1):
    rets = np.array(rets)
    if len(rets[rets!=0])<10:
        return 0
    return np.sqrt(250.0)* np.mean(rets) / ((np.std(rets)**2/adj)**(1/2))

def sortino_ratio(rets, adj = 1):
    rets = np.array(rets)
    return np.sqrt(250.0)* np.mean(rets) / ((np.std(rets[rets <0 ])**2/adj)**(1/2))

def max_dd(cumpnl):
    cumpnl = np.array(cumpnl)
    max_dd = 0
    max_pnl = 0
    max_dd_temp = 0
    for x in cumpnl:
        if x >= max_pnl:
            max_pnl = x
        elif x < max_pnl:
            max_dd_temp = x/max_pnl - 1
            if max_dd_temp < max_dd:
                max_dd = max_dd_temp
    return max_dd

def annualized(rets, adj = 1):
    return np.sum(rets)*250.0/(len(rets)/adj)

def sharpe_ratio_adj(rets, adj = 1):
    rets = np.array(rets[rets!=0])
    if len(rets)==1:
        return 0
    return np.sqrt(250.0)* np.mean(rets) / ((np.std(rets)**2/adj)**(1/2))

def avg_daily_rt(rets):
    rets= np.array(rets)
    rets = rets[rets != 0]
    daily = sum(rets)/(len(rets))
    annual = 250* daily

    return annual

#%% get data
data = ts.get_k_data('sh',start='2000-01-01',end='2020-07-15')
data['date'] = data['date'].apply(pd.to_datetime)
# data = data[(data['date']<pd.to_datetime('20150101')) |
#             (data['date']>pd.to_datetime('20160101'))]
volume = data['volume']
close = data['close']


logDel = np.log(np.array(data['high'])) - np.log(np.array(data['low']))
logRet_1 = np.array(np.diff(np.log(close))) #这个作为后面计算收益使用
logRet_5 = np.log(np.array(close[5:])) - np.log(np.array(close[:-5]))
logVol_5 = np.log(np.array(volume[5:])) - np.log(np.array(volume[:-5]))


logDel = logDel[5:]
logRet_1 = logRet_1[4:]
close = np.array(close[5:])
Date = data.loc[5:,'date'].reset_index(drop=True)


A = np.column_stack([logRet_1,logRet_5,logDel,logVol_5])
df = pd.DataFrame(A,columns=['logRet_1','logRet_5','logDel','logVol_5'])
df = df.replace([np.inf, -np.inf], 0)
A = df.values


#%% class define
class GMM_fit:
    def __init__(self, minN, maxN, X_data, date, logRet):
        self.data = X_data
        self.date_list = date
        self.logRet_list = logRet
        self.model_list = []
        self.minN = np.max([down_limitN,np.min([minN,up_limitN-2])])
        self.maxN = np.min([up_limitN,np.max([maxN,down_limitN+2])])
        self.find = False
        
    def find_n_train(self):
        sh_ratio_adj_list = []
        
        for n in np.arange(self.minN, self.maxN+1, 1):
            model = GaussianMixture(n_components=n, covariance_type='full',
                                    max_iter=4000, tol=tolerance).fit(self.data)
            while (model.converged_==False):
                n_iter = model.n_iter_+1000
                model = GaussianMixture(n_components=n, covariance_type='full',
                                        max_iter=n_iter, tol=tolerance).fit(self.data)
    
            self.model_list.append(model)
            hidden_states = model.predict(self.data)
            res = pd.DataFrame({'Date':self.date_list,'logRet_1':self.logRet_list,
                                'state':hidden_states}).set_index('Date')
            rt = res.logRet_1
            ret_cum = res.copy()
            
            plt.figure()
            for i in range(model.n_components):
                pos = (hidden_states==i)
                pos = np.append(0,pos[:-1])
                res['state_ret%s'%i] = rt.multiply(pos)
                ret_cum['state_ret%s'%i] = res['state_ret%s'%i].cumsum()
                sh_ratio_adj_list.append(round(sharpe_ratio_adj(res['state_ret%s'%i]),3))
                plt.plot(ret_cum.index,ret_cum['state_ret%s'%i],label='state%s'%i)
            plt.legend(loc='lower left')
            
            ret_cum.drop(['logRet_1','state'],axis=1,inplace = True)
            ret_cum.columns = np.arange(0,model.n_components)
        
            max_point = np.argmax(ret_cum.iloc[-100:,:].apply(np.mean,axis=0))
            min_point = np.argmin(ret_cum.iloc[-100:,:].apply(np.mean,axis=0))
            
            ret_cum_copy = ret_cum.copy()
            ret_cum_copy1 = ret_cum.copy()
            
            for i in ret_cum.columns:
                ret_cum_copy.loc[:,i] = (ret_cum_copy.loc[:,i] < 
                                         (ret_cum.loc[:,max_point])+0.03)
                ret_cum_copy1.loc[:,i] = (ret_cum_copy1.loc[:,i] > 
                                         (ret_cum.loc[:,min_point]-0.03))
            max_per = (np.sum(ret_cum_copy.values)-len(ret_cum))/(len(ret_cum)*(model.n_components-1))
            min_per = (np.sum(ret_cum_copy1.values)-len(ret_cum))/(len(ret_cum)*(model.n_components-1))
            
            if ((max_per > 0.95) and (min_per > 0.95)):
                if ((sh_ratio_adj_list[max_point-n]>upper_thres) and (sh_ratio_adj_list[min_point-n]<lower_thres)):
                    self.find = True
                    self.n_pick = n
                    self.n_pick_loc = n-self.minN
    
                    self.model_pick = self.model_list[self.n_pick_loc]
                    hidden_states = self.model_pick.predict(self.data)
                    self.current_state = hidden_states[-1]
                    sh_ratio_adj = sh_ratio_adj_list[np.sum(np.arange(self.minN,self.n_pick,1)):
                                                     np.sum(np.arange(self.minN,self.n_pick,1))+self.n_pick] # 截取所选n对应的sharpe ratio list
                    
                    if (sh_ratio_adj[self.current_state] >= upper_thres):
                        self.buy_sell = 1
                    elif (sh_ratio_adj[self.current_state] <= lower_thres):
                        self.buy_sell = -1
                    else:
                        self.buy_sell = 0
                    
                    
                    self.sh_ratio_adj = sh_ratio_adj
                    
                    
                    ### Check result
                    print('n_pick_loc:',self.n_pick_loc)
                    print('n_pick:',self.n_pick)        
                    print('current state:',self.current_state)
                    print('max point:',max_point,'percent:',max_per)
                    print('min point:',min_point,'percent:',min_per)                
                    print('')
                    break
    
    def fix_n_train(self,n):
        self.n_pick = n
        
        model = GaussianMixture(n_components=n, covariance_type='full',
                                max_iter=4000, tol=tolerance).fit(self.data)
        while (model.converged_==False):
            n_iter = model.n_iter_+1000
            model = GaussianMixture(n_components=n, covariance_type='full',
                                    max_iter=n_iter, tol=tolerance).fit(self.data)
        
        hidden_states = model.predict(self.data)
        self.current_state = hidden_states[-1]
        res = pd.DataFrame({'Date':self.date_list,'logRet_1':self.logRet_list,
                            'state':hidden_states}).set_index('Date')
        rt = res.logRet_1
        self.sh_ratio_adj = []
        self.model_pick = model
        
        plt.figure()
        for i in range(model.n_components):
            pos = (hidden_states==i)
            pos = np.append(0,pos[:-1]) #第二天进行买入操作
            res['state_ret%s'%i] = rt.multiply(pos)
            plt.plot(Date_train, res['state_ret%s'%i].cumsum(),'-',label='state %s'%i)
            self.sh_ratio_adj.append(round(sharpe_ratio_adj(res['state_ret%s'%i]),3))
        plt.title('iter:%s tol:%s'%(model.n_iter_,tolerance))
        plt.legend(loc='lower left')
            
        sh_ratio_adj = pd.Series(self.sh_ratio_adj)
        long_state = np.array(sh_ratio_adj[sh_ratio_adj>upper_thres].index)
        short_state = np.array(sh_ratio_adj[sh_ratio_adj<lower_thres].index)
        
    
        if self.current_state in long_state:
            self.buy_sell = 1
        elif self.current_state in short_state:
            self.buy_sell = -1
        else:
            self.buy_sell = 0

        print('n_pick:',self.n_pick)        
        print('current state:',self.current_state)
        print('')

    def find_n_merge_train(self):
        sh_ratio_adj_list = []
        sh_ratio_merge_len = []
        
        for n in np.arange(self.minN, self.maxN+1, 1):
            model = GaussianMixture(n_components=n, covariance_type='full',
                                    max_iter=4000, tol=tolerance).fit(self.data)
            while (model.converged_==False):
                n_iter = model.n_iter_+1000
                model = GaussianMixture(n_components=n, covariance_type='full',
                                        max_iter=n_iter, tol=tolerance).fit(self.data)
    
            self.model_list.append(model)
            hidden_states = model.predict(self.data)
            res = pd.DataFrame({'Date':self.date_list,'logRet_1':self.logRet_list,
                                'state':hidden_states}).set_index('Date')
            rt = res.logRet_1
            
            # plt.figure()
            long_ret = pd.Series(np.zeros(len(hidden_states)))
            short_ret = pd.Series(np.zeros(len(hidden_states)))
            other_ret = pd.Series(np.zeros(len(hidden_states)))
            
            for i in range(model.n_components):
                pos = (hidden_states==i)
                pos = np.append(0,pos[:-1])
                res['state_ret%s'%i] = rt.multiply(pos)
                ratio = round(sharpe_ratio_adj(res['state_ret%s'%i]),3)
                if ratio > upper_thres:
                    long_ret = long_ret.add(np.array(res['state_ret%s'%i]))
                elif ratio < lower_thres:
                    short_ret = short_ret.add(np.array(res['state_ret%s'%i]))
                else:
                    other_ret = other_ret.add(np.array(res['state_ret%s'%i]))
                sh_ratio_adj_list.append(ratio)
            #     plt.plot(res.index,res['state_ret%s'%i].cumsum(),label='state%s'%i)
            # plt.title('iteration:',model.n_iter_)
            # plt.legend(loc='lower left')
            
            
            # plt.figure()
            # plt.plot(res.index,long_ret.cumsum(),label='long')
            # plt.plot(res.index,short_ret.cumsum(),label='short')
            # plt.plot(res.index,other_ret.cumsum(),label='other')
            # plt.legend()
            
            sh_ratio_long_merge = sharpe_ratio_adj(long_ret)
            sh_ratio_short_merge = sharpe_ratio_adj(short_ret)
            sh_ratio_merge_len.append(sh_ratio_long_merge-sh_ratio_short_merge)
        
            
        sh_ratio_merge_len = pd.Series(sh_ratio_merge_len).fillna(0)
        
        self.n_pick_loc = np.argmax(sh_ratio_merge_len)
        self.n_pick = np.arange(self.minN, self.maxN+1, 1)[self.n_pick_loc]
        
        self.model_pick = self.model_list[self.n_pick_loc]
        hidden_states = self.model_pick.predict(self.data)
        self.current_state = hidden_states[-1]
        
        sh_ratio_adj = sh_ratio_adj_list[np.sum(np.arange(self.minN,self.n_pick,1)):
                                         np.sum(np.arange(self.minN,self.n_pick,1))+self.n_pick] # 截取所选n对应的sharpe ratio list
        
        
        self.current_sh_ratio_adj = sh_ratio_adj[self.current_state]
        
        
        if (sh_ratio_adj[self.current_state] >= upper_thres):
            self.buy_sell = 1
        elif (sh_ratio_adj[self.current_state] <= lower_thres):
            self.buy_sell = -1
        else:
            self.buy_sell = 0
        
        
        print('n_pick:',self.n_pick)        
        print('current state:',self.current_state)
        print(list(sh_ratio_merge_len))

#%% rolling
upper_thres = 1
lower_thres = -1
roll_num = 1700
tolerance = 10**-21
train_length = 3200

up_limitN = 18
down_limitN = 9

daily_rtn1 = []
daily_rtn2 = []
daily_rtn3 = []
state_seq = []
current_sh_ratio_adj_list = []
last_signal = 0
full = 0


start_index = 0
real_price_ratio = []
Date_pred = Date[train_length + start_index : train_length + start_index + roll_num]
close_pred = close[train_length + start_index : train_length + start_index + roll_num]

n_pick = 14  # initial condition
n_pick_list = []


for num in range(0,roll_num):
    X_train = df.iloc[num + start_index : train_length + start_index + num]
    Date_train = Date[num + start_index : train_length + start_index + num]
    logRet_1_train = logRet_1[num + start_index : train_length + start_index + num]
    real_price_ratio.append(round(close[train_length + start_index + num]/
                                  close[train_length + start_index],3))
    
    
    
    # find best n
    gmm = GMM_fit(n_pick-1, n_pick+1, X_train, Date_train, logRet_1_train)
    gmm.find_n_merge_train()
    

    n_pick = gmm.n_pick
    n_pick_list.append(n_pick)
    buy_sell = gmm.buy_sell
    state_seq.append(buy_sell)
    current_sh_ratio_adj = gmm.current_sh_ratio_adj
    current_sh_ratio_adj_list.append(current_sh_ratio_adj)
    

    # strategy1 做空,0->平仓
    rtn1 = buy_sell * logRet_1[train_length + start_index + num]
    daily_rtn1.append(rtn1)

    
    # strategy2 不做空,0->sell
    if last_signal != 1:
        if buy_sell == 1: # buy
            rtn2 = logRet_1[train_length + start_index + num]
            last_signal = 1
        else:
            rtn2 = 0
    else: 
        if buy_sell != 1: # sell
            last_signal = buy_sell
            rtn2 = 0
        else: # hold
            rtn2 = logRet_1[train_length + start_index + num]
    
    daily_rtn2.append(rtn2)
    
    
    # strategy3 不做空,0->hold
    if full == 0: 
        if buy_sell == 1: # buy
            rtn3 = logRet_1[train_length + start_index + num]
            full = 1
        else:
            rtn3 = 0
    else:
        if buy_sell == -1: # sell
            full = 0
            rtn3 = 0
        else: # hold
            rtn3 = logRet_1[train_length + start_index + num]
    
    daily_rtn3.append(rtn3)




#%% plot
plt.figure()
plt.grid()
plt.plot(Date_pred,np.exp(np.array(daily_rtn1).cumsum()),'b-',label='strategy')
buy_point = (np.array(state_seq)==1)
sell_point = (np.array(state_seq)==-1)
plt.plot(Date_pred[buy_point],np.exp(np.array(daily_rtn1).cumsum())[buy_point],'s',c='orange',label='long')
plt.plot(Date_pred[sell_point],np.exp(np.array(daily_rtn1).cumsum())[sell_point],'o',c='green',label='short')
plt.plot(Date_pred,real_price_ratio,'r--',label='stock market')
plt.legend(loc='best')
plt.title('strategy 1, thresholds:%s & %s, tol:%s'%(upper_thres,lower_thres,tolerance))



plt.figure()
plt.grid()
plt.plot(Date_pred,np.exp(np.array(daily_rtn2).cumsum()),'b-',label='strategy')
plt.plot(Date_pred,real_price_ratio,'r--',label='stock market')
plt.legend(loc='best')
plt.title('strategy 2, thresholds:%s & %s, tol:%s'%(upper_thres,lower_thres,tolerance))


plt.figure()
plt.grid()
plt.plot(Date_pred,np.exp(np.array(daily_rtn3).cumsum()),'b-',label='strategy')
plt.plot(Date_pred,real_price_ratio,'r--',label='stock market')
plt.legend(loc='best')
plt.title('strategy 3, thresholds:%s & %s, tol:%s'%(upper_thres,lower_thres,tolerance))
