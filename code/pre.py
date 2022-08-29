#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

import seaborn as sns
import scipy.stats as st
import os
import re

warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


# In[2]:


data = pd.read_csv('fea_data_1207.csv', sep=',')
# data.head()


# In[3]:


# 压缩内存
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    for col in df.columns:
        col_type = df[col].dtype
        
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
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


# In[4]:


df = reduce_mem_usage(data)


# In[5]:


# nan可视化
missing = df.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()


# In[6]:


# missing


# In[7]:


df.info()


# In[8]:


col_list = df.columns.tolist()
col_list.remove('qty')
col_list.remove('original')
col_list.remove('date_start')
col_list.remove('unit')
col_list.remove('geography')
col_list.remove('product')
col_list.remove('ts')
col_list.remove('monthofyear')
col_list.remove('last_monthofyear')
col_list.remove('date')


# In[9]:


df['unit'] = df['unit'].astype('category')
df['geography'] = df['geography'].astype('category')
df['product'] = df['product'].astype('category')
df['date_start'] = df['date_start'].astype('category')
df['monthofyear'] = df['monthofyear'].astype('category')


# In[10]:


used_features = col_list


# In[14]:


#df['unit'] = df['unit'].astype('str')


# In[15]:


#df['geography'] = df['geography'].astype('str')
#df['product'] = df['product'].astype('str')


# In[11]:


cate_cols = ['date_start', 'unit', 'geography', 'product', 'monthofyear']
#cate_cols=''


# In[13]:


from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import gc

drop_col = ['ts', 'original', 'date', 'last_monthofyear', 'qty']

train = df[df['original'] == 'train']
labels = np.array(train['qty'].values.tolist())
train.drop(drop_col, axis=1, inplace=True)
test = df[df['original'] == 'test']
test_label = test['qty'].values.tolist()
test.drop(drop_col, axis=1, inplace=True)

used_features = used_features
ts_folds = TimeSeriesSplit(n_splits = 5)
N_round = 20000
Verbose = 500
Early_Stopping_Rounds = 100
target = 'qty'

params = {
    'objective': 'regression',
    'boosting': 'gbdt',
    'learning_rate': 0.001,
    'num_leaves': 2 ** 5,
    'bagging_fraction': 0.95,
    'bagging_freq': 1,
    'bagging_seed': 66,
    'feature_fraction': 0.7,
    'feature_fraction_seed': 66,
    'max_bin': 100,
    'max_depth': 10,
    'metric': {'l2', 'l1'},
    'verbose': -1
}

for fold_n, (train_index, valid_index) in enumerate(ts_folds.split(train)):
    if fold_n in [0, 1, 2, 3]:  
        continue  
  
    print('Training with validation') 
    trn_data = lgb.Dataset(train.iloc[train_index], label=labels[train_index],
                          categorical_feature=cate_cols)  
    val_data = lgb.Dataset(train.iloc[valid_index], label=labels[valid_index],
                          categorical_feature=cate_cols)  
    clf = lgb.train(params, trn_data, num_boost_round=N_round, valid_sets=[trn_data, val_data], verbose_eval=Verbose,  
    early_stopping_rounds=Early_Stopping_Rounds)
    val = clf.predict(train.iloc[valid_index])   
    mae_ = mean_absolute_error(labels[valid_index], val)  
  
    print('MAE: {}'.format(mae_))  
  
    print("ReTraining on all data")  
    gc.enable()  
    del trn_data, val_data  
    gc.collect()  
    Best_iteration = clf.best_iteration  
    print("Best_iteration: ", Best_iteration)  
    trn_data = lgb.Dataset(train, label=labels, categorical_feature=cate_cols)  
    clf = lgb.train(params, trn_data, num_boost_round=int(Best_iteration * 1.2))
  #valid_sets=[trn_data], verbose_eval=Verbose)  
  #pred = clf.predict(test[used_features])


# In[14]:


pred = clf.predict(test)
mae_test = mean_absolute_error(test_label, pred) 
mse_test = mean_squared_error(test_label, pred) 
print('测试集MAE: {}'.format(mae_test))
print('测试集MSE: {}'.format(mse_test))


# In[15]:


def feature_importance(gbm):
    importance = gbm.feature_importance(importance_type='gain')
    names = gbm.feature_name()
    print("-" * 10 + 'feature_importance:')
    no_weight_cols = []
    name_lis = []
    score_lis = []
    for name, score in sorted(zip(names, importance), key=lambda x: x[1], reverse=True):
        if score <= 1e-8:
            no_weight_cols.append(name)
        else:
            print('{}: {}'.format(name, score))
            name_lis.append(name)
            score_lis.append(score)
    print("no weight columns: {}".format(no_weight_cols))
    return name_lis, score_lis


# In[16]:


name_lis, score_lis = feature_importance(clf)


# In[20]:


# importance_fea = name_lis[:55]
# name_lis[66]


# all_fea = data.columns.tolist()[:6] + importance_fea + ['date_start']
# tree_data = data[all_fea]
# tree_data.head()

# tree_data.to_pickle('tree_data.pkl')

# In[17]:


plt.figure(figsize=(15, 30))
sns.barplot(x=name_lis, y=score_lis)
plt.xticks(rotation=90)


# mean_squared_error(mse_qty_diff['qty_fiff'][1:], mse_qty_diff['qty_pre_fiff'][1:]) 

# In[21]:


used_features = name_lis[:67]


# In[ ]:


drop_col = ['ts', 'original', 'date', 'last_monthofyear', 'qty']

train = df[df['original'] == 'train']
labels = np.array(train['qty'].values.tolist())
train.drop(drop_col, axis=1, inplace=True)
test = df[df['original'] == 'test']
test_label = test['qty'].values.tolist()
test.drop(drop_col, axis=1, inplace=True)

used_features = used_features
ts_folds = TimeSeriesSplit(n_splits = 5)
N_round = 20000
Verbose = 500
Early_Stopping_Rounds = 100
target = 'qty'

params = {
    'objective': 'regression',
    'boosting': 'gbdt',
    'learning_rate': 0.001,
    'num_leaves': 2 ** 5,
    'bagging_fraction': 0.95,
    'bagging_freq': 1,
    'bagging_seed': 66,
    'feature_fraction': 0.7,
    'feature_fraction_seed': 66,
    'max_bin': 100,
    'max_depth': 10,
    'metric': {'l2', 'l1'},
    'verbose': -1
}

for fold_n, (train_index, valid_index) in enumerate(ts_folds.split(train)):
    if fold_n in [0, 1, 2, 3]:  
        continue  
  
    print('Training with validation') 
    trn_data = lgb.Dataset(train.iloc[train_index], label=labels[train_index],
                          categorical_feature=cate_cols)  
    val_data = lgb.Dataset(train.iloc[valid_index], label=labels[valid_index],
                          categorical_feature=cate_cols)  
    clf = lgb.train(params, trn_data, num_boost_round=N_round, valid_sets=[trn_data, val_data], verbose_eval=Verbose,  
    early_stopping_rounds=Early_Stopping_Rounds)
    val = clf.predict(train.iloc[valid_index])   
    mae_ = mean_absolute_error(labels[valid_index], val)  
  
    print('MAE: {}'.format(mae_))  
  
    print("ReTraining on all data")  
    gc.enable()  
    del trn_data, val_data  
    gc.collect()  
    Best_iteration = clf.best_iteration  
    print("Best_iteration: ", Best_iteration)  
    trn_data = lgb.Dataset(train, label=labels, categorical_feature=cate_cols)  
    clf = lgb.train(params, trn_data, num_boost_round=int(Best_iteration * 1.2))
    pred = clf.predict(test)


# In[ ]:


pre = pred.tolist()


# In[48]:


ss = df[df['original'] == 'test'][['unit', 'ts']]
ss['qty'] = pre
#ss.head()


# ss1 = ss[ss['unit']=='1305184b1a7634e62b1ea3dc7c5fa81d']
# ss2 = ss1[ss1['ts'] == '2021-03-02'].index.tolist()[0]
# ss2

# In[49]:


ss.to_csv('ss_lgb_cat1605.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




