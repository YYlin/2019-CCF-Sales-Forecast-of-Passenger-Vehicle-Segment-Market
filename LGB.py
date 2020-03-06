# -*- coding: utf-8 -*-
# @Time    : 2020/3/5 9:49
# @Author  : YYLin
# @Email   : 854280599@qq.com
# @File    : LGB.py
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import mean_squared_error as mse
import datetime as dtm
import lightgbm as lgb
import warnings
import sys

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

# 方便显示打印的数据
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)

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
        print('异常 需退出')
        sys.exit()
    return res_dict


# col, col2, col3 中 ，设1.5倍四分位距之外的数据为异常值，用上下四分位数的均值填充
# 设1.5倍四分位距之外的数据为异常值，用上下四分位数的均值填充
for col in ['popularity', 'carCommentVolum', 'newsReplyVolum']:
    col_per = np.percentile(data[col], (25, 75))
    diff = 1.5 * (col_per[1] - col_per[0])
    col_per_in = (data[col] >= col_per[0] - diff) & (data[col] <= col_per[1] + diff)
    data.loc[~col_per_in, col] = col_per.mean()

# 求bodyType adcode model 每年销售量的均值
data['bt_ry_mean'] = data.groupby(['bodyType', 'regYear'])['salesVolume'].transform('mean')
data['ad_ry_mean'] = data.groupby(['adcode', 'regYear'])['salesVolume'].transform('mean')
data['md_ry_mean'] = data.groupby(['model', 'regYear'])['salesVolume'].transform('mean')


# 参考车辆识别码每一位表示什么意思 https://zhuanlan.zhihu.com/p/37659200
data['model_0_3'] = data['model'].astype(str).apply(lambda x: x[0:3])
data['model_4_8'] = data['model'].astype(str).apply(lambda x: x[3:9])
data['model_8_16'] = data['model'].astype(str).apply(lambda x: x[9:])

data['model_1'] = data['model'].astype(str).apply(lambda x: x[0:1])
data['model_2'] = data['model'].astype(str).apply(lambda x: x[1:2])
data['model_3'] = data['model'].astype(str).apply(lambda x: x[2:3])
data['model_4'] = data['model'].astype(str).apply(lambda x: x[3:4])
data['model_9'] = data['model'].astype(str).apply(lambda x: x[9:10])
data['model_10'] = data['model'].astype(str).apply(lambda x: x[10:11])

# 统计月份到1，2，12的距离
data['month_to_12'] = data['regMonth'].apply(lambda x: abs(12 - x))
data['month_to_2'] = data['regMonth'].apply(lambda x: abs(2 - x))
data['month_to_1'] = data['regMonth'].apply(lambda x: abs(1 - x))

data = pd.concat([data, evaluation_public], ignore_index=True)
data['date_'] = data['regYear'] + data['regMonth']

data['label'] = data['salesVolume']
data['id'] = data['id'].fillna(0).astype(int)
data['bodyType'] = data['model'].map(train_sales.drop_duplicates('model').set_index('model')['bodyType'])

# LabelEncoder
for i in ['bodyType', 'model']:
    data[i] = data[i].map(dict(zip(data[i].unique(), range(data[i].nunique()))))
data['mt'] = (data['regYear'] - 2016) * 12 + data['regMonth']


def get_stat_feature(df_):
    df = df_.copy()
    stat_feat = []
    df['model_adcode'] = df['adcode'] + df['model']
    df['model_adcode_mt'] = df['model_adcode'] * 100 + df['mt']
    for col in tqdm(['label', 'popularity']):
        # shift
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
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

    f_mean = lambda x: x.rolling(window=3, min_periods=1).mean()
    f_std = lambda x: x.rolling(window=3, min_periods=1).std()

    function_list = [f_mean, f_std]
    function_name = ['mean', 'std']

    for i in range(len(function_list)):
        df[('popularity_%s' % function_name[i])] = df.sort_values('mt').groupby(['model'])[
            'popularity'].apply(function_list[i])

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

    # 在上半年还是下半年
    df['quarter'] = 0
    df.loc[(df['regMonth'].isin([1, 2, 3])), 'quarter'] = 1
    df.loc[(df['regMonth'].isin([4, 5, 6])), 'quarter'] = 2
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
        pred: list,
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


df_lgb = pd.DataFrame({'id': evaluation_public['id']})
score_account = 0
for col_add in ['ad_ry_mean', 'md_ry_mean', 'bt_ry_mean']:
    for month in [25, 26, 27, 28]:

        data_df, shift_feat = get_stat_feature(data)

        shift_feat.remove('shift_model_adcode_mt_popularity_7')
        shift_feat.remove('shift_model_adcode_mt_popularity_8')

        #    data_df=Encoder(data_df, 'model')
        # month_to_12:0.618  month_to_1:0.619      month_to_2:0.616
        cate_feat = ['adcode', 'model', 'regMonth', 'bodyType', 'model_0_3', 'model_4_8', 'model_8_16', 'quarter',
                     'model_1', 'model_2', 'model_3', 'model_4', 'model_9', 'model_10']

        # 类别特征转换
        for i in cate_feat:
            data_df[i] = data_df[i].astype('category')

        # 特征选择
        features = shift_feat + cate_feat + [col_add] + ['happyNY', 'sale_weightMonth', 'carCommentVolum',
                                                         'newsReplyVolum', 'popularity', 'regYear', 'date_',
                                                         'happyNY-high', 'month_to_1', 'month2lastNY', 'month2nextNY']

        # 数据集划分
        st = 1
        all_idx = (data_df['mt'].between(st, month - 1))
        train_idx = (data_df['mt'].between(st, month - 5))
        valid_idx = (data_df['mt'].between(month - 4, month - 4))
        test_idx = (data_df['mt'].between(month, month))

        # 最终确认   对label作log变换
        data_df['n_label'] = np.log(data_df['label'])
        train_x = data_df[train_idx][features]
        train_y = data_df[train_idx]['n_label']

        valid_x = data_df[valid_idx][features]
        valid_y = data_df[valid_idx]['n_label']

        # get model
        model = lgb.LGBMRegressor(
            num_leaves=68, reg_alpha=1, reg_lambda=0.2, objective='mse',
            max_depth=9, learning_rate=0.05, min_child_samples=50, random_state=2019,
            n_estimators=3000, subsample=0.9, colsample_bytree=0.7,
        )
        model.fit(train_x, train_y,
                  eval_set=[(train_x, train_y), (valid_x, valid_y)],
                  categorical_feature=cate_feat,
                  early_stopping_rounds=400, verbose=100)
        print("=======================================================")

        # offline
        data_df['pred_label'] = np.e ** model.predict(data_df[features])
        best_score = score(data_df[valid_idx])
        print('best_score:', best_score, 'col_add', col_add, 'month', month, 'model.best_iteration_',
              model.best_iteration_)
        print("=======================================================")

        # forecastVolum 是用于保存训练时预测的结果
        model.n_estimators = model.best_iteration_ + 149

        model.fit(data_df[all_idx][features], data_df[all_idx]['n_label'], categorical_feature=cate_feat)
        data_df['forecastVolum'] = np.e ** model.predict(data_df[features])

        # 用于保存测试集中预测的结果
        sub = data_df[test_idx][['id']]
        sub['forecastVolum'] = data_df[test_idx]['forecastVolum'].apply(lambda x: 0 if x < 0 else x).round().astype(int)
        score_account = score_account + best_score
        data.loc[(data.regMonth == (month - 24)) & (data.regYear == 2018), 'salesVolume'] = sub['forecastVolum'].values
        data.loc[(data.regMonth == (month - 24)) & (data.regYear == 2018), 'label'] = sub['forecastVolum'].values

        # 下面是用于保存验证集输出的结果
        sub_val = data_df[valid_idx][['id']]
        sub_val['forecastVolum'] = data_df[valid_idx]['forecastVolum'].apply(
            lambda x: 0 if x < 0 else x).round().astype(int)
        # 在原始数据集中增加新的列 forecastVolum 用于保存验证集的结果
        data.loc[(valid_idx), 'forecastVolum'] = sub_val['forecastVolum'].values

    # 从结果集合中输出自己的测试集
    sub = data.loc[(data.regMonth >= 1) & (data.regYear == 2018), ['id', 'salesVolume']]
    sub.columns = ['id', 'forecastVolum']
    df_lgb = df_lgb.merge(sub, 'left', on=['id'])
    df_lgb.rename(columns={'forecastVolum': 'forecastVolum_lgb_{}'.format(col_add)}, inplace=True)

    # 从结果之中提取自己的验证集 forecastVolum:为预测的结果  salesVolume:为实际的结果值
    valid_id = (data_df['mt'].between(21, 24))
    val_lgb = data.loc[(valid_id), ['mt', 'forecastVolum', 'salesVolume']]
    val_lgb.rename(columns={'forecastVolum': 'forecastVolum_lgb_{}'.format(col_add)}, inplace=True)

    print('saving the result for valid .......')
    val_lgb.to_csv('Proba_in_valid_lgb_%s.csv' % col_add, index=False)

score_avr_offline = score_account / 12

# 直接保存测试集对应的结果
nm_unique_str = dtm.datetime.now().strftime('%Y%m%d_%H%M%S')
print('saving the result for test .......')
df_lgb.to_csv('Proba_in_test_lgb.csv', index=False)
df_lgb['forecastVolum'] = (df_lgb['forecastVolum_lgb_ad_ry_mean'] + df_lgb['forecastVolum_lgb_md_ry_mean'] + df_lgb[
    'forecastVolum_lgb_bt_ry_mean']) / 3
df_lgb[['id', 'forecastVolum']].round().astype(int).to_csv(f'result_in_lgb_{nm_unique_str}.csv', index=False)
print("***" * 40)
print('score_avr_offline:', score_avr_offline)

