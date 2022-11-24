import gc
import os.path
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from tqdm import tqdm


def read_processed_bundle_data(path: str,
                               to_path: str,
                               user: str,
                               time: str,
                               spare_features: list,
                               dense_features: list,
                               usecols: list = None,
                               check_cols: list = None,
                               sep=','):
    """
    :param dense_features: 连续特征
    :param spare_features: 类别特征
    :param path: data_path
    :param user: user_col_name
    :param time: time_col_name
    :param usecols: Use None to read all columns.
    :param check_cols: delete null rows
    :param sep:
    :return: Returns a DataFrameGroupy by uid.
    """
    if check_cols is None:
        check_cols = []
    df = pd.read_csv(path, sep=sep, usecols=usecols)
    print("loaded data of {} rows".format(df.shape[0]))
    for col in usecols:
        null_num = df[col].isnull().sum()
        if null_num > 0:
            print("There are {} nulls in {}.".format(null_num, col))
            df = df[~df[col].isnull()]
    print("buy:{}, unbuy:{}".format(df[df['impression_result'] == 1].shape[0], df[df['impression_result'] == 0].shape[0]))
    df['register_time'] = pd.to_datetime(df["register_time"]).apply(lambda x: int(x.timestamp()))
    df = reduce_mem(df)

    lbes = [LabelEncoder() for _ in range(len(spare_features))]
    for i in range(len(spare_features)):
        df[spare_features[i]] = lbes[i].fit_transform(df[spare_features[i]])
    with open(to_path+"lbes.pkl", 'wb') as file:
        pickle.dump(lbes, file)
    mms = MinMaxScaler()
    df[dense_features] = mms.fit_transform(df[dense_features])
    with open(to_path+"mms.pkl", 'wb') as file:
        pickle.dump(mms, file)

    grouped = df.sort_values(by=[time]).groupby(user)
    return grouped


def transform2sequences(grouped, default_col, sequence_col):
    """
    :param grouped:
    :param default_col:
    :param sequence_col: columns needed to generate sequences.
    :return: DataFrame
    """
    df = pd.DataFrame(
        data={
            "uid": list(grouped.groups.keys()),
            **{col_name: grouped[col_name].apply(lambda x: x.iloc[0]) for col_name in default_col[1:]},
            **{col_name: grouped[col_name].apply(list) for col_name in sequence_col},
        }
    )
    return df


def create_fixed_sequences(values: list, sequence_length):
    """padding with -1."""
    sequences = []
    seq = [-1 for _ in range(sequence_length)]
    for end_index in range(len(values)):
        valid_len = min(sequence_length, end_index + 1)
        seq[sequence_length - valid_len:sequence_length] = values[end_index + 1 - valid_len:end_index + 1]
        sequences.append(seq.copy())
    return sequences


def split_train_test_by_user(df: pd.DataFrame, user: str, time: str):
    """
    Use last user record as test data.
    :param df:
    :param user: user_col_name
    :param time: time_col_name
    :return: DataFrame
    """
    grouped = df.sort_values(by=time).groupby(user)
    train = grouped.apply(lambda x: x[:-1])
    test = grouped.apply(lambda x: x[-1:])
    return train, test


def split_train_test_by_time(df: pd.DataFrame, user: str, time: str):
    """
    Use last user record as test data.
    :param df:
    :param user: user_col_name
    :param time: time_col_name
    :return: DataFrame
    """
    df = df.sort_values(by=time)
    rows = df.shape[0]
    th = int(rows * 0.8)
    train = df[:th]
    test = df[th:]
    return train, test


def gen_bundle_data():
    path = "./bundle_time/"
    if not os.path.exists(path):
        os.mkdir(path)
    table_default_col = ['module_id_1', 'module_id_2', 'module_id_3', 'product_cnt'] + ['model_name']
    user_default_col = ['uid', 'register_country', 'register_time', 'is_visitor', 'register_device',
                        'register_device_brand', 'register_os', 'gender']
    user_changeable_col = ['ftime', 'bundle_id', 'bundle_price', 'impression_result', 'island_no', 'spin_stock',
                           'coin_stock', 'diamond_stock', 'island_complete_gap_coin',
                           'island_complete_gap_building_cnt', 'tournament_rank', 'login_cnt_1d', 'ads_watch_cnt_1d',
                           'register_country_arpu', 'life_time', 'star', 'battle_pass_exp', 'is_up_to_date_version',
                           'pet_heart_stock', 'pet_exp_stock', 'friend_cnt', 'social_friend_cnt', 'pay_amt', 'pay_cnt',
                           'pay_per_day', 'pay_mean']
    spare_features = ['uid', 'is_visitor', 'gender', 'is_up_to_date_version', 'register_country', 'register_device',
                      'register_device_brand', 'register_os', 'bundle_id']
    dense_features = ['register_time', 'bundle_price', 'island_no', 'spin_stock', 'coin_stock',
                      'diamond_stock', 'island_complete_gap_coin', 'island_complete_gap_building_cnt',
                      'tournament_rank', 'login_cnt_1d', 'ads_watch_cnt_1d', 'register_country_arpu',
                      'life_time', 'star', 'battle_pass_exp', 'pet_heart_stock', 'pet_exp_stock', 'friend_cnt',
                      'social_friend_cnt', 'pay_amt', 'pay_cnt', 'pay_per_day', 'pay_mean']

    grouped = read_processed_bundle_data(path="./raw_data/bundle/brs_daily_20211101_20211230.csv",
                                         user='uid',
                                         time='ftime',
                                         spare_features=spare_features,
                                         dense_features=dense_features,
                                         usecols=user_default_col + user_changeable_col,
                                         check_cols=['bundle_id'],
                                         to_path=path)

    df = transform2sequences(grouped, user_default_col, user_changeable_col)

    sequence_length = 8
    for col_name in user_changeable_col:
        df[col_name] = df[col_name].apply(lambda x: create_fixed_sequences(x, sequence_length=sequence_length))

    df = df.explode(column=user_changeable_col, ignore_index=True)

    train, test = split_train_test_by_time(df, 'uid', 'ftime')
    train.to_csv(path+"train_data.csv", index=False, sep=',')
    test.to_csv(path+"test_data.csv", index=False, sep=',')


def reduce_mem(df):
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in tqdm(df.columns):
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
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
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem, 100 * (start_mem - end_mem) / start_mem, (time.time() - starttime) / 60))
    gc.collect()
    return df


if __name__ == '__main__':
    gen_bundle_data()
    pass