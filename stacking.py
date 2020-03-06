# -*- coding: utf-8 -*-
# @Time    : 2020/3/6 15:56
# @Author  : YYLin
# @Email   : 854280599@qq.com
# @File    : stacking.py
# 对lgb和xgb的结果进行线性的stacking
from sklearn.linear_model import LinearRegression
import pandas as pd
import warnings
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

# 显示所有列 显示所有行 设置value的显示长度为100，默认为50
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)

# 通过字节之间的变化 减少内存的使用
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
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
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# 读取lgb数据
lgb_ad_ry = pd.read_csv('Proba_in_valid_lgb_ad_ry_mean.csv')
lgb_bt_ry = pd.read_csv('Proba_in_valid_lgb_bt_ry_mean.csv')
lgb_md_ry = pd.read_csv('Proba_in_valid_lgb_md_ry_mean.csv')

# 读取xgb数据.
xgb_ad_ry = pd.read_csv('Proba_in_valid_xgb_ad_ry_mean.csv')
xgb_bt_ry = pd.read_csv('Proba_in_valid_xgb_bt_ry_mean.csv')
xgb_md_ry = pd.read_csv('Proba_in_valid_xgb_md_ry_mean.csv')

print('merge nowing .................')
data = lgb_ad_ry.merge(lgb_bt_ry, 'left', on=['salesVolume', 'mt'])\
    .merge(lgb_md_ry, 'left', on=['salesVolume', 'mt'])\
    .merge(xgb_ad_ry, 'left', on=['salesVolume', 'mt'])\
    .merge(xgb_bt_ry, 'left', on=['salesVolume', 'mt'])\
    .merge(xgb_md_ry, 'left', on=['salesVolume', 'mt'])
print('data:', data.columns)

data = reduce_mem_usage(data)
fea = ['forecastVolum_lgb_ad_ry_mean', 'forecastVolum_lgb_md_ry_mean', 'forecastVolum_lgb_bt_ry_mean',
       'forecastVolum_xgb_ad_ry_mean', 'forecastVolum_xgb_md_ry_mean', 'forecastVolum_xgb_bt_ry_mean']

# 读取测试集并合并
lgb_test = pd.read_csv('Proba_in_test_lgb.csv')
xgb_test = pd.read_csv('Proba_in_test_xgb.csv')
data_test = lgb_test.merge(xgb_test, 'left', on=['id'])
print('data_test:', data_test.columns)

ensemble_model = LinearRegression()
ensemble_model.fit(data[fea], data['salesVolume'])
submit = pd.DataFrame(data_test['id'], columns=['id'])

final_predictions = ensemble_model.predict(data_test[fea])
print('合并sale对应的系数矩阵是:', ensemble_model.coef_)

submit['forecastVolum'] = final_predictions.round().astype(int)
submit.to_csv('stacking_lgb_xgb.csv', index=False)


