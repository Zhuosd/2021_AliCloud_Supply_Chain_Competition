


#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import supply_chain_round1_baseline
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

trainA = pd.read_csv('../data/Dataset/demand_train_A.csv', sep=',', index_col=0)
testA = pd.read_csv('../data/Dataset/demand_test_A.csv', sep=',', index_col=0)
inventory_infoA = pd.read_csv('../data/Dataset/inventory_info_A.csv', sep=',', index_col=0)
geo_topo = pd.read_csv('../data/Dataset/geo_topo.csv', sep=',')
product_topo = pd.read_csv('../data/Dataset/product_topo.csv', sep=',')
weightA = pd.read_csv('../data/Dataset/weight_A.csv', sep=',', index_col=0)
trainB = pd.read_csv('../data/Dataset/demand_train_B.csv', sep=',', index_col=0)
testB = pd.read_csv('../data/Dataset/demand_test_B.csv', sep=',', index_col=0)
inventory_infoB = pd.read_csv('../data/Dataset/inventory_info_B.csv', sep=',', index_col=0)
weightB = pd.read_csv('../data/Dataset/weight_B.csv', sep=',', index_col=0)


# In[10]:


# 转换地理信息和产品信息的行列
geo = pd.melt(geo_topo, value_vars=list(geo_topo.columns),
             var_name='unit', value_name='geography_level')
product = pd.melt(product_topo, value_vars=list(product_topo.columns),
             var_name='unit', value_name='product_level')


# In[11]:


trainA['original'] = 'train'
testA['original'] = 'train'
trainB['original'] = 'train'
testB['original'] = 'test'


# In[12]:


# trainB['unit'].nunique(), testB['unit'].nunique()


# In[13]:


# testB.head()


# In[14]:


# 合并训练集和测试集
all_data = pd.concat([trainA, testA, trainB, testB])


# In[15]:


# all_data.tail()


# In[16]:


# 将权重合并进来
weight = pd.concat([weightA, weightB])
data_w = pd.merge(all_data, weight, on=['unit'])


# In[17]:


# 按照日期进行排序
data_w = data_w.sort_values(by=['ts', 'unit'])
data_w = data_w.reset_index().drop(columns='index')


# In[18]:


# data_w.head().append(data_w.tail())


# In[19]:


# 删除只有一个值的列
data_w.drop(['geography_level', 'product_level'], axis=1, inplace=True)


# In[20]:


# data_w['unit'].nunique() == 632


# In[21]:


# 将label改为qty的净增量
data_w['qty_diff'] = data_w['qty'].groupby(data_w['unit']).diff(periods=1)
data_w['qty'] = data_w['qty_diff']
data_w.dropna(axis=0, how='any', inplace=True)
data_w = data_w.reset_index(drop=True)
del data_w['qty_diff']


# In[22]:


# data_unit1 = data_w[data_w['unit'] == 'e527dedfec712d75834a2eacb23e51fc']


# In[23]:


#times=data_unit1['ts'].tolist()


# plt.figure(figsize=(12, 4))
# ticks=list(range(0,len(times),100))
# if ticks[-1]!=len(times)-1:
#     ticks.append(len(times)-1)
# labels=[times[i] for i in ticks]
# plt.plot(data_unit1['ts'], np.log1p(data_unit1['qty']))
# plt.xlabel('时间',fontsize=12)
# plt.ylabel('订单量',fontsize=12)
# plt.xlim(0,len(times)-1)
# plt.xticks(ticks, rotation=45)
# # plt.xtick(labels, rotation=45, horizontalalignment='right')
# plt.tick_params(labelsize=12)
# plt.title('qty分布',fontsize=12)

# In[24]:


# data_unit1['geography'].nunique()
# 每一个unit中只有一个product和geography


# data_unit2 = data_w[data_w['unit'] == 'ffddc0dbb7fa28b00a5b6ddc8e7e317c']
# plt.figure(figsize=(12, 4))
# ticks=list(range(0,len(times),20))
# if ticks[-1]!=len(times)-1:
#     ticks.append(len(times)-1)
# labels=[times[i] for i in ticks]
# plt.plot(data_unit2['ts'], np.log1p(data_unit2['qty']))
# plt.xlabel('时间',fontsize=12)
# plt.ylabel('订单量',fontsize=12)
# plt.xlim(0,len(times)-1)
# plt.xticks(ticks, rotation=45)
# # plt.xtick(labels, rotation=45, horizontalalignment='right')
# plt.tick_params(labelsize=12)
# plt.title('qty分布',fontsize=12)

# #绘图部分
# times=data_unit1['ts'].tolist()
# #分时间区间,保证最后一位纳入标签, 隔100个显示
# ticks=list(range(0,len(times),100))
# if ticks[-1]!=len(times)-1:
#     ticks.append(len(times)-1)
# labels=[times[i] for i in ticks]
# #中文和负号的正常显示
# plt.rcParams['font.sans-serif'] = ['Times New Roman']
# plt.rcParams['font.sans-serif'] = [u'SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# fig= plt.figure(figsize=(12, 4),dpi=100)
# #设置图形的显示风格
# plt.style.use('ggplot')
# ax1 = fig.add_subplot(111)
# ax1.plot(data_unit1['qty'],'-v',linewidth=1.5)
# #ax1.legend(loc='upper right', frameon=False,fontsize = 10)
# ax1.set_xlabel('时间',fontsize =10)
# ax1.set_ylabel('订单量',fontsize =10)
# ax1.set(ylim=[9000, 30000])
# ax1.set(xlim=[0,len(times)-1])
# ax1.set_xticks(ticks)
# ax1.set_xticklabels(labels, rotation=45, horizontalalignment='right')
# ax1.tick_params(labelsize=8)
# ax1.set_title('qty分布',fontsize =8)
# #ax1.legend(loc='upper right', frameon=False,fontsize = 10)
# #plt.savefig('./time_distribute.png',format='png', dpi=300)
# #plt.show()

# # 查看某正常数据单元的分布情况
# plt.figure(figsize=(10,8))
# x = data_w[data_w['unit']=='ff65120018e54439e4071446a1cf6b14']['ts']
# y = data_w[data_w['unit']=='ff65120018e54439e4071446a1cf6b14']['qty']
# plt.plot(x, y)
# # plt.xticks(x, x, rotation=45)
# # plt.show()
# plt.xticks([])  # 去掉横坐标值
# plt.xlabel('时间', fontsize=15)

# In[25]:


data_c = data_w.copy()


# In[26]:


# data_c.describe()


# ### 删除异常值

# In[27]:


# 这里异常值处理的代码
def outliers_proc(data, col_name, scale=3):
    """
    用于清洗异常值，默认用 box_plot（scale=3）进行清洗
    :param data: 接收 pandas 数据格式
    :param col_name: pandas 列名
    :param scale: 尺度
    :return:
    """

    def box_plot_outliers(data_ser, box_scale):
        """
        利用箱线图去除异常值
        :param data_ser: 接收 pandas.Series 数据格式
        :param box_scale: 箱线图尺度，
        :return:
        """
        iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
        val_low = data_ser.quantile(0.25) - iqr
        val_up = data_ser.quantile(0.75) + iqr
        rule_low = (data_ser < val_low)
        rule_up = (data_ser > val_up)
        return (rule_low, rule_up), (val_low, val_up)

    data_n = data.copy()
    data_series = data_n[col_name]
    rule, value = box_plot_outliers(data_series, box_scale=scale)
    index = np.arange(data_series.shape[0])[rule[0] | rule[1]]
    print("Delete number is: {}".format(len(index)))
    data_n = data_n.drop(index)
    data_n.reset_index(drop=True, inplace=True)
    print("Now column number is: {}".format(data_n.shape[0]))
    index_low = np.arange(data_series.shape[0])[rule[0]]
    outliers = data_series.iloc[index_low]
    print("Description of data less than the lower bound is:")
    print(pd.Series(outliers).describe())
    index_up = np.arange(data_series.shape[0])[rule[1]]
    outliers = data_series.iloc[index_up]
    print("Description of data larger than the upper bound is:")
    print(pd.Series(outliers).describe())

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    sns.boxplot(y=data[col_name], data=data, palette="Set1", ax=ax[0])
    sns.boxplot(y=data_n[col_name], data=data_n, palette="Set1", ax=ax[1])
    return data_n


# In[28]:


df_c = outliers_proc(data_c, 'qty', scale=3)


# plt.figure(figsize=(16, 6))
# sns.boxplot(data_c['qty'], orient='h', width=0.5)

# In[29]:


unit_list = data_c['unit'].value_counts().index.tolist()


# ### 查看每个unit中有几个product和geography

# In[30]:


def unit_geography_product(data, unit_list):
    product_list = []
    geography_list = []
    for unit in unit_list:
        pro_num, geo_num = data[data['unit']==unit]['product'].nunique(), data[data['unit']==unit]['geography'].nunique()
        if pro_num > 1:
            product_list.append([unit, pro_num])
        if geo_num > 1:
            geography_list.append([unit, geo_num])
    return product_list, geography_list



# In[31]:


# product_unit, geo_unit = unit_geography_product(data_c, unit_list)


# In[32]:


# 每一个单元都只有一个product和geography
# geo_unit, product_unit


# In[33]:


# data_w['geography'].nunique()==81
# data_w['product'].nunique()==19
# data_w['product'].value_counts().index.tolist()


# In[34]:


#data_c['product'].value_counts()


# In[35]:


#data_c.info()


# def product_fig(data, n):
#     # n:隔多少日期显示
#     # plt.figure(figsize=(10,6))
#
#     times = data['ts'].value_counts().index.tolist()
#     ticks=list(range(0,len(times), n))
#     if ticks[-1]!=len(times)-1:
#         ticks.append(len(times)-1)
#
#     unit_lis = data['unit'].value_counts().index.tolist()
#     for unit in unit_lis:
#         plt.plot(data[data['unit']==unit]['ts'],
#                  data[data['unit']==unit]['qty'],
#                 label=unit[-3:])
#
#     # plt.xlabel('时间')
#     # plt.ylabel('qty')
#
#     #plt.xlim(0,(len(times)))
#     #plt.xticks(ticks, rotation=45)
#     #plt.tick_params(labelsize=12)
#
#     # plt.legend()
#     # plt.title('某一product下的unit分布情况')

# #不同的product数量
# pro_n = data_c['product'].nunique()
# #不同product的集合
# product_lis = data_c['product'].value_counts().index.tolist()[-9:]
#
# plt.figure(figsize=(30, 30))
# plt.subplots_adjust(wspace=0.8, hspace=0.9)
# for i, product in enumerate(product_lis):
#     plt.subplot(3,3,i+1)
#     data_p = data_c[data_c['product']==product]
#     data_p.sort_values(by=['ts'])
#     data_p.reset_index().drop(columns='index')
#
#     product_fig(data_p, 50)

# def geography_fig(data, n):
#     # n:隔多少日期显示
#     plt.figure(figsize=(10,6))
#
#     times = data['ts'].value_counts().index.tolist()
#     ticks=list(range(0,len(times), n))
#     if ticks[-1]!=len(times)-1:
#         ticks.append(len(times)-1)
#
#     unit_lis = data['unit'].value_counts().index.tolist()
#     for unit in unit_lis:
#         plt.plot(data[data['unit']==unit]['ts'],
#                  data[data['unit']==unit]['qty'],
#                 label=unit[-3:])
#
#     plt.xlabel('时间')
#     plt.ylabel('qty')
#
#     plt.xlim(0,(len(times)))
#     plt.xticks(ticks, rotation=45)
#     plt.tick_params(labelsize=12)
#
#     plt.legend()
#     plt.title('某一geography下的unit分布情况')

# data_geo1 = data_c[data_c['geography']=='af04e3e5da488f5ca9b5d1d4ce04ebaa']
# data_geo1.sort_values(by='ts')
# data_geo1.reset_index().drop(columns='index');

# geography_fig(data_geo1, 50)

# data_c.info()

# ### 日期特征

# In[36]:


df = data_c.copy()


# In[37]:


df['datetime'] = pd.to_datetime(df['ts'],errors='coerce')   #先转化为datetime类型,默认format='%Y-%m-%d %H:%M:%S'
df['date'] = df['datetime'].dt.date   #转化提取年-月-日
df['year'] =df['datetime'].dt.year.fillna(0).astype("int")   #转化提取年 ,
#如果有NaN元素则默认转化float64型，要转换数据类型则需要先填充空值,在做数据类型转换
df['month'] = df['datetime'].dt.month.fillna(0).astype("int")  #转化提取月
df['monthofyear'] = df['year'].map(str) + '-' + df['month'].map(str) #转化获取年-月
df['day'] = df['datetime'].dt.day.fillna(0).astype("int")      #转化提取天
# df['hour'] = df['datetime'].dt.hour.fillna(0).astype("int")    #转化提取小时
# df['minute'] = df['datetime'].dt.minute.fillna(0).astype("int") #转化提取分钟
# df['second'] = df['datetime'].dt.second.fillna(0).astype("int") #转化提取秒
df['dayofyear'] = df['datetime'].dt.dayofyear.fillna(0).astype("int") #一年中的第n天
df['weekofyear'] = df['datetime'].dt.weekofyear.fillna(0).astype("int") #一年中的第n周
df['weekday'] = df['datetime'].dt.weekday.fillna(0).astype("int") #周几，一周里的第几天，Monday=0, Sunday=6
df['quarter'] = df['datetime'].dt.quarter.fillna(0).astype("int")  #季度
df['is_wknd'] = df['datetime'].dt.dayofweek // 4
df['is_month_start'] = df['datetime'].dt.is_month_start.astype(int)
df['is_month_end'] = df['datetime'].dt.is_month_end.astype(int)



# lag特征
# 表示同一元素历史时间点的值(例如在这个赛题中，同一个unit昨天、前天、上周对应的使用量)

# In[38]:


df['yesterday_qty'] = df.groupby('unit')['qty'].shift(1).fillna(method='ffill').reset_index().sort_index().set_index('index')
df['before_yesterday_qty'] = df.groupby('unit')['qty'].shift(2).fillna(method='ffill').reset_index().sort_index().set_index('index')
df['last_week_qty'] = df.groupby('unit')['qty'].shift(7).fillna(method='ffill').reset_index().sort_index().set_index('index')
df['last_14day_qty'] = df.groupby('unit')['qty'].shift(14).fillna(method='ffill').reset_index().sort_index().set_index('index')
df['last_21day_qty'] = df.groupby('unit')['qty'].shift(21).fillna(method='ffill').reset_index().sort_index().set_index('index')


# 滑动窗口统计特征：历史时间窗口内的统计值

# In[39]:


def qty_rolling(df, window, val, keys):
    df['qty_rolling'+str(window)+'_mean'] = df.groupby(keys)[val].transform(
              lambda x: x.shift(1).rolling(window=window, min_periods=3, win_type="triang").mean()).values.tolist()
    df['qty_rolling'+str(window)+'_max'] = df.groupby(keys)[val].transform(
              lambda x: x.shift(1).rolling(window=window, min_periods=3).max()).values.tolist()
    df['qty_rolling'+str(window)+'_min'] = df.groupby(keys)[val].transform(
              lambda x: x.shift(1).rolling(window=window, min_periods=3).min()).values.tolist()
    df['qty_rolling'+str(window)+'_std'] = df.groupby(keys)[val].transform(
              lambda x: x.shift(1).rolling(window=window, min_periods=3, win_type="triang").std()).values.tolist()
    df['qty_rolling'+str(window)+'_skew'] = df.groupby(keys)[val].transform(
              lambda x: x.shift(1).rolling(window=window, min_periods=3).skew()).values.tolist()
    df['qty_rolling'+str(window)+'_kurt'] = df.groupby(keys)[val].transform(
              lambda x: x.shift(1).rolling(window=window, min_periods=3).kurt()).values.tolist()
    #df['qty_rolling'+str(window)+'_quantile'] = df.groupby(keys)[val].transform(
              #lambda x: x.rolling(window=window, min_periods=3).quantile()).values.tolist()
    df['qty_rolling'+str(window)+'_corr'] = df.groupby(keys)[val].transform(
              lambda x: x.shift(1).rolling(window=window, min_periods=3).corr()).values.tolist()
    return df


# In[40]:


# 滚动7天和14天
keys = ['unit']
df = qty_rolling(df, 7, 'qty', keys)
df = qty_rolling(df, 14, 'qty', keys)
df = qty_rolling(df, 21, 'qty', keys)


# 指数加权移动平均

# In[41]:


def qty_ewm(df, alpha, val, keys):
    df['qty_ewm'+'_mean'] = df.groupby(keys)[val].transform(lambda x: x.shift(1).ewm(alpha=alpha).mean()).values.tolist()
    df['qty_ewm'+'_std'] = df.groupby(keys)[val].transform(lambda x: x.shift(1).ewm(alpha=alpha).std()).values.tolist()
    df['qty_ewm'+'_corr'] = df.groupby(keys)[val].transform(lambda x: x.shift(1).ewm(alpha=alpha).corr()).values.tolist()
    return df


# In[42]:


df = qty_ewm(df, 0.95, 'qty', keys)


# In[43]:


# 构造日期差
df_date = df.groupby(['unit'])['date'].agg({'min'}).reset_index()
df_date.rename(columns={'min':'date_start'}, inplace=True)


# In[44]:


df = pd.merge(df, df_date, on=['unit'])


# In[45]:


df['times'] = df['date'] - df['date_start']


# In[46]:


df['times'] = df['times'].map(lambda x:x.days)


# In[47]:


#df.head()


# ### product独热编码

# In[48]:


#product_dummies = pd.get_dummies(df['product'])


# df = df.join(product_dummies)
# df.head()

# #### 按照季度、月份、一周统计

# 按照年份-月份，求一个月的增量和
# - std方差也可以加上，（甚至可以考虑最大子序列和）

# In[49]:


def diff_max_min(x):
    return x.max() - x.min()


# df['quarter'].unique(), df['month'].unique(), df['weekofyear'].unique()

# In[50]:


df['last_quarter'] = df.groupby(['unit'])['quarter'].transform(lambda x:x.shift(1)).values.tolist()
df['last_monthofyear'] = df.groupby(['unit'])['monthofyear'].transform(lambda x:x.shift(1)).values.tolist()
df['last_weekofyear'] = df.groupby(['unit'])['weekofyear'].transform(lambda x:x.shift(1)).values.tolist()


# #不能用本季度，本月，本周的数据，向上做差分

# - 这里可以写个函数，构造统计量特征
# - 年需要构造么？14天需要构造么？

# In[51]:


# 按照季度
df_quarter = df.groupby(['unit', 'year', 'last_quarter'])['qty'].agg({'sum', 'min', 'max',
                                                                       'median', 'std', diff_max_min}).reset_index()


# df_quarter['year'] = df_quarter['year'].astype('str')
# df_quarter['quarter'] = df_quarter['quarter'].astype('str')
# df_quarter['year_quarter'] = df_quarter['year'] + '_' + df_quarter['quarter']

# In[52]:


df_quarter.rename(columns={'sum':'quarter_sum', 'max':'quarter_max', 'min':'quarter_min', 'median':'quarter_median',
                           'std':'quarter_std', 'diff_max_min':'quarter_min_max'}, inplace=True)


# In[53]:


df = pd.merge(df, df_quarter, on=['unit', 'year', 'last_quarter'])


# In[54]:


# 按照月份
df_month = df.groupby(['unit', 'year', 'last_monthofyear'])['qty'].agg({'sum', 'min', 'max',
                                                                       'median', 'std', diff_max_min}).reset_index()


# In[55]:


df_month.rename(columns={'sum':'month_sum', 'max':'month_max', 'min':'month_min', 'median':'month_median',
                           'std':'month_std', 'diff_max_min':'month_min_max'}, inplace=True)
df = pd.merge(df, df_month, on=['unit', 'year', 'last_monthofyear'])


# In[56]:


# 按照周
df_week = df.groupby(['unit', 'year', 'last_weekofyear'])['qty'].agg({'sum', 'min', 'max',
                                                                       'median', 'std', diff_max_min}).reset_index()
df_week.rename(columns={'sum':'week_sum', 'max':'week_max', 'min':'week_min', 'median':'week_median',
                           'std':'week_std', 'diff_max_min':'week_min_max'}, inplace=True)
df = pd.merge(df, df_week, on=['unit', 'year', 'last_weekofyear'])


# #### 类别特征编码

# In[57]:


import category_encoders as ce


# In[58]:


# 日期特征已经拆分过，故删除
del df['datetime']


# In[59]:


#df.info()


# In[60]:


# del df['monthofyear']


# In[61]:


df1 = df.select_dtypes(include='object')
object_lis = list(df1.columns)


# In[62]:


# object_lis


# In[63]:


# 删除只有一个值的geography_level和product_level
cat_features = ['unit', 'geography', 'product']


# In[64]:


# 拆分训练集和验证集
train_x, val_x = df.iloc[:int(df.shape[0]*0.8), :], df.iloc[int(df.shape[0]*0.8):, :]
train_y, val_y = df['qty'][:int(df.shape[0]*0.8)], df['qty'][int(df.shape[0]*0.8):]


# ##### 目标编码，按照label的均值填充
# - 由于目标编码用的均值，后续可以用统计量特征包含

# ##### 计数编码

# ##### 标签编码

# In[65]:


# 标签编码
from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()
for fea in cat_features:
    encoded = label_enc.fit_transform(df[fea])
    df[fea+'_label'] = encoded


# ##### 类别特征交叉组合

# In[66]:


# 构造geo和product中有多少unit
def nums_unit(x):
    return len(x)
df_pro = df.groupby(['product'])['unit'].agg({nums_unit}).reset_index()
df_pro.rename(columns={'nums_unit':'pro_unit_num'}, inplace=True)
df = pd.merge(df, df_pro, on=['product'])

df_geo = df.groupby(['geography'])['unit'].agg({nums_unit}).reset_index()
df_geo.rename(columns={'nums_unit':'geo_unit_num'}, inplace=True)
df = pd.merge(df, df_geo, on=['geography'])


# #删除只有只有一个值的特征
# def delete_single(df):
#     for col in df:
#     # print(col,len(df.loc[:,col].unique()))
#         if(len(df.loc[:,col].unique()) == 1):
#             df.pop(col)
#     return df

# In[67]:


# len(df.columns)


# ##### 构造原始特征unit, product, geography, weight的统计量特征

# In[68]:


# 按照unit
df_unit = df[df['original']=='train'].groupby(['unit'])['qty'].agg({'sum', 'min', 'max',
                                                                       'median', 'std', diff_max_min}).reset_index()
df_unit.rename(columns={'sum':'unit_sum', 'max':'unit_max', 'min':'unit_min', 'median':'unit_median',
                           'std':'unit_std', 'diff_max_min':'unit_min_max'}, inplace=True)
df = pd.merge(df, df_unit, on=['unit'])
na_col = ['unit_sum', 'unit_max', 'unit_min', 'unit_median', 'unit_std', 'unit_min_max']
df[na_col] = df[na_col].fillna(method='ffill')
#df.head()


# ### 权重和qty的乘积也可以作为特征，后续可以试试

# ##### 压缩内存

# In[70]:


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


# In[71]:


df = reduce_mem_usage(df)


df.to_csv('fea_data_1207.csv', index=None)

df.to_pickle('fea_data_1209.pkl')

data = pd.read_csv('fea_data_1207.csv', sep=',')


# data.head()


# In[3]:


# 压缩内存
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
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

    end_mem = df.memory_usage().sum() / 1024 ** 2
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


# df['unit'] = df['unit'].astype('str')


# In[15]:


# df['geography'] = df['geography'].astype('str')
# df['product'] = df['product'].astype('str')


# In[11]:


cate_cols = ['date_start', 'unit', 'geography', 'product', 'monthofyear']
# cate_cols=''


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
ts_folds = TimeSeriesSplit(n_splits=5)
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
# valid_sets=[trn_data], verbose_eval=Verbose)
# pred = clf.predict(test[used_features])


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
ts_folds = TimeSeriesSplit(n_splits=5)
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
# ss.head()


# ss1 = ss[ss['unit']=='1305184b1a7634e62b1ea3dc7c5fa81d']
# ss2 = ss1[ss1['ts'] == '2021-03-02'].index.tolist()[0]
# ss2

# In[49]:


ss.to_csv('ss_lgb_cat1605.csv', index=False)


supply_chain_round1_baseline.SupplyChainRound1Baseline.run()

