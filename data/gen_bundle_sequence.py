import time

import pandas as pd


def read_processed_bundle_data(path: str, user: str, time: str, usecols: list = None, check_cols: list = None, sep=','):
    """
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


def split_train_test(df: pd.DataFrame, user: str, time: str):
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


def gen_bundle_data():
    cur = time.time()
    table_default_col = ['module_id_1', 'module_id_2', 'module_id_3', 'product_cnt'] + ['model_name']
    user_default_col = ['uid', 'register_country', 'register_time', 'is_visitor', 'register_device',
                        'register_device_brand', 'register_os', 'gender']
    user_changeable_col = ['ftime', 'bundle_id', 'bundle_price', 'impression_result', 'island_no', 'spin_stock',
                           'coin_stock', 'diamond_stock', 'island_complete_gap_coin',
                           'island_complete_gap_building_cnt', 'tournament_rank', 'login_cnt_1d', 'ads_watch_cnt_1d',
                           'register_country_arpu', 'life_time', 'star', 'battle_pass_exp', 'is_up_to_date_version',
                           'pet_heart_stock', 'pet_exp_stock', 'friend_cnt', 'social_friend_cnt', 'pay_amt', 'pay_cnt',
                           'pay_per_day', 'pay_mean']
    grouped = read_processed_bundle_data(path="F:/brs_data/brs_daily_20211101_20211230.csv",
                                         user='uid',
                                         time='ftime',
                                         usecols=user_default_col + user_changeable_col,
                                         check_cols=['bundle_id'])
    print(str(time.time() - cur) + "transfroming...")
    df = transform2sequences(grouped, user_default_col, user_changeable_col)
    sequence_length = 8
    print(str(time.time() - cur) + "creating fixed sequences...")
    for col_name in user_changeable_col:
        df[col_name] = df[col_name].apply(lambda x: create_fixed_sequences(x, sequence_length=sequence_length))
    df['register_time'] = pd.to_datetime(df["register_time"]).apply(lambda x: x.timestamp())
    print(str(time.time() - cur) + "exploding...")
    df = df.explode(column=user_changeable_col, ignore_index=True)
    train, test = split_train_test(df, 'uid', 'ftime')
    print(str(time.time() - cur) + "saving...")
    train.to_csv("bundle/train_data.csv", index=False, sep=',')
    test.to_csv("bundle/test_data.csv", index=False, sep=',')


if __name__ == '__main__':
    gen_bundle_data()
