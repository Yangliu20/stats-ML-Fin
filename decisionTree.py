# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 09:04:23 2020

@author: yangl
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report,confusion_matrix,
                             average_precision_score)
import sklearn.tree as tree
from pylab import rcParams

path = 'C:/Users/yangl/Documents/Python/OLS_&_tree/'


#%%

data = pd.read_pickle(path + 'regr_df_new.pkl')
vix = pd.read_csv(path + 'SSE510050VIX(1).csv', index_col=0)
vix.tradedate = vix.tradedate.apply(pd.to_datetime)
vix = vix[['tradedate', 'vix']]

data_new = pd.merge(vix, data, left_on = 'tradedate', right_on = 'date')
data_new.set_index('date', drop=True, inplace=True)
data_new.drop(['tradedate', 'iv'], axis=1, inplace=True)
data_new.to_pickle(path + 'vix_hv.pkl')


#%%

df = pd.read_pickle(path + 'vix_hv.pkl')
df.vix /= 100
X = df.apply(lambda row: np.log(row[0]/row[1:]), axis=1)[:-1] # log(vix)-log(hv)
y = np.log(df.vix).rolling(2).apply(lambda x:np.sign(x[1]-x[0])).shift(-1).dropna()
y.replace(0, -1, inplace=True)

col = 'cchv22 cchv250 cchv44 cchv5 gkhv10 gkhv22 gkhv44 gkhv5 gkyzhv5 gkyzhv66 rshv66 yzhv22 yzhv5 yzhv66'.split(' ')

#%% feature selection
X_train, X_test, Y_train, Y_test = train_test_split(X[col], y, test_size = 0.2,
                                                    shuffle=False)

clf = DecisionTreeClassifier(max_depth = 7, criterion='gini')
clf.fit(X_train, Y_train)
importances = clf.feature_importances_
feature = pd.DataFrame({'features':X_train.columns,'importances':importances})
feature = feature[feature['importances']!=0].sort_values(by='importances', 
                 ascending=False).reset_index()

rcParams['figure.figsize'] = 12,8
plt.title("Feature importances")
plt.bar(feature['features'], feature['importances'],
        color="blue", alpha = 0.3, align="center")
plt.show()

print('number of features: ',len(feature))
feature_select = feature['features']

#%% depth

X_train, X_test, Y_train, Y_test = train_test_split(X[feature_select], y, test_size = 0.2,
                                                    shuffle=False)

depth_list = range(3,8)
acc_list = []
precision_list = []

for depth in depth_list:
    clf = DecisionTreeClassifier(max_depth = depth, criterion='gini')
    clf.fit(X_train, Y_train)
    y_score = clf.predict(X_test)
    
    acc = (sum(np.array(Y_test) == y_score)/len(Y_test))
    average_precision = average_precision_score(np.array(Y_test), y_score)    
    
    acc_list.append(acc)
    precision_list.append(average_precision)

print(acc_list)
print(precision_list)

#depth = depth_list[np.argmax(acc_list)]
depth = depth_list[np.argmax(precision_list)]


clf = DecisionTreeClassifier(max_depth = depth, criterion='gini')

clf.fit(X_train, Y_train)
acc= (sum(Y_test == clf.predict(X_test))/len(Y_test))
print('depth:   ',depth)
print('accuracy:',acc)

#%%

importances1 = clf.feature_importances_
indices1 = np.argsort(importances1)[::-1]

feature1 = pd.DataFrame({'features':feature_select,'importances':importances1})
feature1 = feature1[feature1['importances']!=0].sort_values(by='importances', 
                 ascending=False).reset_index()

rcParams['figure.figsize'] = 12,8
plt.title("Feature importances")
plt.bar(feature_select[indices1], importances1[indices1],
        color="blue", alpha = 0.3, align="center")
plt.show()
print('number of features: ', len(feature1))

#%% plot

cn=['sell vol', 'long vol']

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (12,4), dpi = 400) 

tree.plot_tree(clf,
               feature_names = feature_select, 
               #proportion = True,
               filled = True)

plt.savefig('fignew.png')