# -*- coding: utf-8 -*-
# @Time    : 2020/3/5 10:26
# @Author  : YYLin
# @Email   : 854280599@qq.com
# @File    : XGB.py
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import LabelEncoder
import datetime as dtm
import xgboost as xgb

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

# 显示所有列
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth',100)

path = 'data/'
train_sales = pd.read_csv(path + 'train_sales_data.csv')
train_search = pd.read_csv(path + 'train_search_data.csv')
train_user = pd.read_csv(path + 'train_user_reply_data.csv')
evaluation_public = pd.read_csv(path + 'evaluation_public.csv')

data = train_sales.copy()
data = data.merge(train_search, 'left', on=['province', 'adcode', 'model', 'regYear', 'regMonth'])
data = data.merge(train_user, 'left', on=['model', 'regYear', 'regMonth'])


def get_delta_yr(year, month):
    res_dict = {}
    new_year_date = {2015: '2015-02-18',
                     2016: '2016-02-08',
                     2017: '2017-01-28',
                     2018: '2018-02-16',
                     2019: '2019-02-05'}
    ny_date = new_year_date[year]
    ny_mt = int(ny_date[5:7]) + (int(ny_date[8:10]) / 30)
    d = month - ny_mt
    if d >= 0:
        res_dict['last'] = d
        ny_date = new_year_date[year + 1]
        res_dict['next'] = 12 - month + int(ny_date[5:7]) + (int(ny_date[8:10]) / 30)
    elif d < 0:
        res_dict['next'] = -d
        ny_date = new_year_date[year - 1]
        res_dict['last'] = 12 - month + int(ny_date[5:7]) + (int(ny_date[8:10]) / 30)
    else:
        sys.exit()
    return res_dict


'''
数据筛选 预处理
'''

# col, col2, col3 中 ，设1.5倍四分位距之外的数据为异常值，用上下四分位数的均值填充
# 设1.5倍四分位距之外的数据为异常值，用上下四分位数的均值填充
for col in ['popularity', 'carCommentVolum', 'newsReplyVolum']:
    col_per = np.percentile(data[col], (25, 75))
    diff = 1.5 * (col_per[1] - col_per[0])
    col_per_in = (data[col] >= col_per[0] - diff) & (data[col] <= col_per[1] + diff)
    data.loc[~col_per_in, col] = col_per.mean()

# 统计销量
data['bt_ry_mean'] = data.groupby(['bodyType', 'regYear'])['salesVolume'].transform('mean')
data['ad_ry_mean'] = data.groupby(['adcode', 'regYear'])['salesVolume'].transform('mean')
data['md_ry_mean'] = data.groupby(['model', 'regYear'])['salesVolume'].transform('mean')


# 新特征
data['model_0_3'] = data['model'].astype(str).apply(lambda x: x[0:3])
data['model_4_8'] = data['model'].astype(str).apply(lambda x: x[3:9])
data['model_8_16'] = data['model'].astype(str).apply(lambda x: x[9:])

data['model_1'] = data['model'].astype(str).apply(lambda x: x[0:1])
data['model_2'] = data['model'].astype(str).apply(lambda x: x[1:2])
data['model_3'] = data['model'].astype(str).apply(lambda x: x[2:3])


data['month_to_12'] = data['regMonth'].apply(lambda x: abs(12-x))
data['month_to_2'] = data['regMonth'].apply(lambda x: abs(2-x))
data['month_to_1'] = data['regMonth'].apply(lambda x: abs(1-x))

data = pd.concat([data, evaluation_public], ignore_index=True)

model_type_list = ['5f727f32393ade77', '9a390098bf87b814', '3c974920a76ac9c1',
                   '7aab7fca2470987e', '04e66e578f653ab9', '32d3069d17aa47c2', 'a9a43d1a7ecbe75d',
                   'a432c483b5beb856', 'b4be3a4917289c82', 'f5d69960089c3614']


def model_type(model):
    if model in model_type_list:
        return 1
    else:
        return 0


data['month_type_feature'] = data['model'].apply(model_type)


# 年底销量暴增（业绩狗）
def threshold(a):
    if a >= 1.5 or a <= 0.67:
        return 1
    else:
        return 0


df_7_12 = data[data['regMonth'] >= 7]
df_11_12 = data[data['regMonth'] >= 11]

df_7_12mean = df_7_12.groupby(['adcode', 'model'])['salesVolume'].agg({'mean7_12': 'mean'}).reset_index()
df_11_12mean = df_11_12.groupby(['adcode', 'model'])['salesVolume'].agg({'mean11_12': 'mean'}).reset_index()

df_gl = pd.merge(df_7_12mean, df_11_12mean, 'left', on=['adcode', 'model'])
df_gl['mean_bl'] = df_gl['mean11_12']/df_gl['mean7_12']

# kpi
df_gl['kpi_dog'] = df_gl['mean_bl'].apply(lambda x: threshold(x))
df_kpi = df_gl[['adcode', 'model', 'kpi_dog']]
data = data.merge(df_kpi, 'left', on=['adcode', 'model'])


data['label'] = data['salesVolume']
data['id'] = data['id'].fillna(0).astype(int)
data['bodyType'] = data['model'].map(train_sales.drop_duplicates('model').set_index('model')['bodyType'])

# LabelEncoder
for i in ['bodyType', 'model']:
    data[i] = data[i].map(dict(zip(data[i].unique(), range(data[i].nunique()))))
data['mt'] = (data['regYear'] - 2016) * 12 + data['regMonth']


data['date_'] = data['regYear']+data['regMonth']


def get_stat_feature(df_):
    df = df_.copy()
    stat_feat = []
    df['model_adcode'] = df['adcode'] + df['model']
    df['model_adcode_mt'] = df['model_adcode'] * 100 + df['mt']
    for col in ['label', 'popularity']:
        # shift
        for i in [1,2,3,4,5,6,7,8,9,10,11,12]:
            stat_feat.append('shift_model_adcode_mt_{}_{}'.format(col, i))
            df['model_adcode_mt_{}_{}'.format(col, i)] = df['model_adcode_mt'] + i
            df_last = df[~df[col].isnull()].set_index('model_adcode_mt_{}_{}'.format(col, i))
            df['shift_model_adcode_mt_{}_{}'.format(col, i)] = df['model_adcode_mt'].map(df_last[col])

    for col in ['carCommentVolum', 'newsReplyVolum', 'popularity', 'bt_ry_mean', 'ad_ry_mean', 'md_ry_mean']:
        lgb_col_na = pd.isnull(df[col])
        df[col] = df[col].replace(0, 1)
        df.loc[lgb_col_na, col] = \
            ((((df.loc[(df['regYear'].isin([2017])) & (df['regMonth'].isin([1, 2, 3, 4])), col].values /
                df.loc[(data['regYear'].isin([2016])) & (df['regMonth'].isin([1, 2, 3, 4])), col].values))) *
             df.loc[(data['regYear'].isin([2017])) & (df['regMonth'].isin([1, 2, 3, 4])), col].values * 1.03).round()

    # 每年的新年在第几月份（1是新年）,新年所在月份为低谷期
    df['happyNY'] = 0
    df.loc[(data['regYear'].isin([2016, 2018]) & df['regMonth'].isin([2])), 'happyNY'] = 1
    df.loc[(data['regYear'].isin([2017]) & df['regMonth'].isin([1])), 'happyNY'] = 1

    # 每年的新年前高峰期在第几月份,新年所在月份为低谷期
    df['happyNY-high'] = 0
    df.loc[(data['regYear'].isin([2016, 2018]) & df['regMonth'].isin([1])), 'happyNY-high'] = 1
    df.loc[(data['regYear'].isin([2017]) & df['regMonth'].isin([12])), 'happyNY-high'] = 1
    df.loc[(data['regYear'].isin([2016]) & df['regMonth'].isin([12])), 'happyNY-high'] = 1

    # 根据月份添加权重值
    a = 6
    b = 4
    df['weightMonth'] = df['regMonth'].map({1: a, 2: a, 3: a, 4: a,
                                                5: b, 6: b, 7: b, 8: b, 9: b, 10: b, 11: b, 12: b, })
    # 在上半年还是下半年
    df['half_year'] = 0
    df.loc[(df['regMonth'].isin([1, 2, 3, 4, 5, 6])), 'half_year'] = 0
    df.loc[(df['regMonth'].isin([7, 8, 9, 10, 11, 12])), 'half_year'] = 1

    # 季度
    df['quarter'] = 0
    df.loc[(df['regMonth'].isin([1, 2, 3])), 'quarter'] = 1
    df.loc[(df['regMonth'].isin([4, 5, 6, ])), 'quarter'] = 2
    df.loc[(df['regMonth'].isin([7, 8, 9])), 'quarter'] = 3
    df.loc[(df['regMonth'].isin([10, 11, 12])), 'quarter'] = 4

    # Month 2 last/next newyear:
    df['month2lastNY'] = df[['regYear', 'regMonth']].apply(lambda x: get_delta_yr(*x)['last'], axis=1)
    df['month2nextNY'] = df[['regYear', 'regMonth']].apply(lambda x: get_delta_yr(*x)['next'], axis=1)

    # 根据销量添加月份权重值

    df['sale_weightMonth'] = df['regMonth'].map({1: 5, 2: 1, 3: 2, 4: 2,
                                                5: 2, 6: 2, 7: 2, 8: 3, 9: 3, 10: 4, 11: 4, 12: 6, })

    del df['salesVolume']
    return df, stat_feat


def score(data, pred='pred_label', label='label', group='model'):
    data['pred_label'] = data['pred_label'].apply(lambda x: 0 if x < 0 else x).round().astype(int)
    data_agg = data.groupby(group).agg({
        pred:  list,
        label: [list, 'mean']
    }).reset_index()
    data_agg.columns = ['_'.join(col).strip() for col in data_agg.columns]
    nrmse_score = []
    for raw in data_agg[['{0}_list'.format(pred), '{0}_list'.format(label), '{0}_mean'.format(label)]].values:
        nrmse_score.append(
            mse(raw[0], raw[1]) ** 0.5 / raw[2]
        )
    print(1 - np.mean(nrmse_score))
    return 1 - np.mean(nrmse_score)


'''

    一：XGBoost：0.7340759427887523/0.59048219000 

'''

df_xgb = pd.DataFrame({'id': evaluation_public['id']})
score_account = 0
for col_add in ['ad_ry_mean', 'md_ry_mean', 'bt_ry_mean']:
    for month in [25, 26, 27, 28]:

        data_df, shift_feat = get_stat_feature(data)

        cate_feat = ['adcode', 'model', 'regMonth', 'bodyType','regYear','month_to_1','date_','kpi_dog','month_type_feature']
        # 对类别特征做Ecoding
        lbl = LabelEncoder()
        for i in cate_feat:
            data_df[i] = lbl.fit_transform(data_df[i].astype(str))

        # 特征选择
        features = shift_feat + cate_feat + [col_add] + \
                   ['weightMonth', 'happyNY', 'half_year', 'sale_weightMonth', 'carCommentVolum',
                    'newsReplyVolum', 'popularity']

        # 数据集划分
        st = 1
        all_idx = (data_df['mt'].between(st, month - 1))
        train_idx = (data_df['mt'].between(st, month - 5))
        valid_idx = (data_df['mt'].between(month - 4, month - 4))
        test_idx = (data_df['mt'].between(month, month))

        # 最终确认
        data_df['n_label'] = np.log1p(data_df['label'])  # 对label作log变换
        train_x = data_df[train_idx][features]
        train_y = data_df[train_idx]['n_label']

        valid_x = data_df[valid_idx][features]
        valid_y = data_df[valid_idx]['n_label']

        model = xgb.XGBRegressor(
            max_depth=7, learning_rate=0.05, n_estimators=5000,
            objective='reg:gamma', tree_method='hist', subsample=0.9,
            colsample_bytree=0.7, min_child_samples=20, eval_metric='rmse'
        )

        model.fit(train_x, train_y,
                  eval_set=[(train_x, train_y), (valid_x, valid_y)],
                  early_stopping_rounds=400, verbose=100)

        # offline
        data_df['pred_label'] = np.expm1(model.predict(data_df[features]))
        best_score = score(data_df[valid_idx])
        print('best_score:', best_score)

        # online
        model.n_estimators = model.best_iteration + 145
        model.fit(data_df[all_idx][features], data_df[all_idx]['n_label'])
        data_df['forecastVolum'] = np.expm1(model.predict(data_df[features]))

        # 阶段结果
        sub = data_df[test_idx][['id']]
        sub['forecastVolum'] = data_df[test_idx]['forecastVolum'].apply(lambda x: 0 if x < 0 else x).round().astype(int)
        score_account = score_account + best_score

        # 对结果进行微调
        if month == 25:
            data.loc[(data.regMonth == (month - 24)) & (data.regYear == 2018), 'salesVolume'] = sub['forecastVolum'].values*0.98
            data.loc[(data.regMonth == (month - 24)) & (data.regYear == 2018), 'label'] = sub['forecastVolum'].values*0.98
        elif month == 26:
            data.loc[(data.regMonth == (month - 24)) & (data.regYear == 2018), 'salesVolume'] = sub['forecastVolum'].values*1.02
            data.loc[(data.regMonth == (month - 24)) & (data.regYear == 2018), 'label'] = sub['forecastVolum'].values*1.02
        else:
            data.loc[(data.regMonth == (month - 24)) & (data.regYear == 2018), 'salesVolume'] = sub['forecastVolum'].values
            data.loc[(data.regMonth == (month - 24)) & (data.regYear == 2018), 'label'] = sub['forecastVolum'].values

        # 下面是用于保存验证集输出的结果
        sub_val = data_df[valid_idx][['id']]
        sub_val['forecastVolum'] = data_df[valid_idx]['forecastVolum'].apply(lambda x: 0 if x < 0 else x).round().astype(int)
        # 在原始数据集中增加新的列 forecastVolum 用于保存验证集的结果
        data.loc[(valid_idx), 'forecastVolum'] = sub_val['forecastVolum'].values

    sub = data.loc[(data.regMonth >= 1) & (data.regYear == 2018), ['id', 'salesVolume']]
    sub.columns = ['id', 'forecastVolum']
    df_xgb = df_xgb.merge(sub, 'left', on=['id'])
    df_xgb.rename(columns={'forecastVolum': 'forecastVolum_xgb_{}'.format(col_add)}, inplace=True)

    # 从结果之中提取自己的验证集 forecastVolum:为预测的结果  salesVolume:为实际的结果值
    valid_id = (data_df['mt'].between(21, 24))
    val_xgb = data.loc[(valid_id), ['mt', 'forecastVolum', 'salesVolume']]
    val_xgb.rename(columns={'forecastVolum': 'forecastVolum_xgb_{}'.format(col_add)}, inplace=True)

    print('saving the result for valid .......')
    val_xgb.to_csv('Proba_in_valid_xgb_%s.csv'%col_add , index=False)


score_avr_offline = score_account / 12

# 直接保存测试集对应的结果
print('saving the result for test .......')
df_xgb.to_csv('Proba_in_test_xgb.csv', index=False)

# 单模结果提交
nm_unique_str = dtm.datetime.now().strftime('%Y%m%d_%H%M%S')

df_xgb['forecastVolum'] = (df_xgb['forecastVolum_xgb_ad_ry_mean'] + df_xgb['forecastVolum_xgb_md_ry_mean'] + df_xgb[
    'forecastVolum_xgb_bt_ry_mean']) / 3
df_xgb[['id', 'forecastVolum']].round().astype(int).to_csv(f'result_in_xgb_{nm_unique_str}.csv', index=False)
print("***" * 40)
print('score_avr_offline:', score_avr_offline)
