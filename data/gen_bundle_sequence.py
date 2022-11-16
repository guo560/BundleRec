import pickle

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


def read_processed_bundle_data(path: str,
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

    lbes = [LabelEncoder() for _ in range(len(spare_features))]
    for i in range(len(spare_features)):
        df[spare_features[i]] = lbes[i].fit_transform(df[spare_features[i]])
    with open("bundle_time/lbes.pkl", 'wb') as file:
        pickle.dump(lbes, file)
    mms = MinMaxScaler()
    df[dense_features] = mms.fit_transform(df[dense_features])
    with open("bundle_time/mms.pkl", 'wb') as file:
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
    """padding with 0."""
    sequences = []
    seq = [0 for _ in range(sequence_length)]
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
    table_default_col = ['module_id_1', 'module_id_2', 'module_id_3', 'product_cnt'] + ['model_name']
    user_default_col = ['uid', 'register_country', 'register_time', 'is_visitor', 'register_device',
                        'register_device_brand', 'register_os', 'gender']
    user_changeable_col = ['ftime', 'bundle_id', 'bundle_price', 'impression_result', 'island_no', 'spin_stock',
                           'coin_stock', 'diamond_stock', 'island_complete_gap_coin',
                           'island_complete_gap_building_cnt', 'tournament_rank', 'login_cnt_1d', 'ads_watch_cnt_1d',
                           'register_country_arpu', 'life_time', 'star', 'battle_pass_exp', 'is_up_to_date_version',
                           'pet_heart_stock', 'pet_exp_stock', 'friend_cnt', 'social_friend_cnt', 'pay_amt', 'pay_cnt',
                           'pay_per_day', 'pay_mean']
    spare_features = ['uid', 'is_visitor', 'gender', 'impression_result', 'is_up_to_date_version',
                      'register_country', 'register_device', 'register_device_brand', 'register_os',
                      'bundle_id']
    dense_features = ['register_time', 'bundle_price', 'island_no', 'spin_stock', 'coin_stock',
                      'diamond_stock', 'island_complete_gap_coin', 'island_complete_gap_building_cnt',
                      'tournament_rank', 'login_cnt_1d', 'ads_watch_cnt_1d', 'register_country_arpu',
                      'life_time', 'star', 'battle_pass_exp', 'pet_heart_stock', 'pet_exp_stock', 'friend_cnt',
                      'social_friend_cnt', 'pay_amt', 'pay_cnt', 'pay_per_day', 'pay_mean']

    grouped = read_processed_bundle_data(path="F:/brs_data/brs_daily_20211101_20211230.csv",
                                         user='uid',
                                         time='ftime',
                                         spare_features=spare_features,
                                         dense_features=dense_features,
                                         usecols=user_default_col + user_changeable_col,
                                         check_cols=['bundle_id'])

    df = transform2sequences(grouped, user_default_col, user_changeable_col)

    sequence_length = 8
    for col_name in user_changeable_col:
        df[col_name] = df[col_name].apply(lambda x: create_fixed_sequences(x, sequence_length=sequence_length))

    df = df.explode(column=user_changeable_col, ignore_index=True)

    train, test = split_train_test_by_time(df, 'uid', 'ftime')
    train.to_csv("bundle_time/train_data.csv", index=False, sep=',')
    test.to_csv("bundle_time/test_data.csv", index=False, sep=',')


if __name__ == '__main__':
    gen_bundle_data()
    pass