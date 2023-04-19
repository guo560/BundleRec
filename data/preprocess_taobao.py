import os
import pickle
import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from data.gen_sequence import split_train_val_test_by_time


def gen_taobao_list_data():
    to_path = "taobao/"
    if not os.path.exists(to_path):
        os.mkdir(to_path)
    data_path = "./raw_data/taobao/UserBehavior.csv"
    df = pd.read_csv(data_path, sep=',', nrows=None,
                     names=['user_id', 'product_id', 'product_category_id', 'behavior_type', 'timestamp'])
    df = df[df['timestamp'] >= datetime.datetime.timestamp(datetime.datetime.strptime("2017-11-25 00:00:00.00", "%Y-%m-%d %H:%M:%S.%f"))]
    df = df[df['timestamp'] <= datetime.datetime.timestamp(datetime.datetime.strptime("2017-11-28 00:00:00.00", "%Y-%m-%d %H:%M:%S.%f"))]
    df['label'] = (df['behavior_type'] == 'buy').astype('int')
    lbes = dict()
    for key in ['user_id', 'product_ids', 'product_category_ids']:
        feature = key[:-1] if key[-1] == 's' else key
        lbes[key] = LabelEncoder()
        df[feature] = lbes[key].fit_transform(df[feature])
    with open(to_path+"lbes.pkl", 'wb') as file:
        pickle.dump(lbes, file)

    mms = MinMaxScaler()
    with open(to_path + "mms.pkl", 'wb') as file:
        pickle.dump(mms, file)

    data_group = df.sort_values(by=['timestamp']).groupby('user_id')

    data = pd.DataFrame(
        data={
            'user_id': list(data_group.groups.keys()),
            'product_ids': data_group['product_id'].apply(list),
            'product_category_ids': data_group['product_category_id'].apply(list),
            'timestamps': data_group['timestamp'].apply(list),
            'labels': data_group['label'].apply(list),
        }
    )

    data.to_csv(to_path+"list_data.csv", index=False, sep=',')


def gen_taobao_data():
    to_path = "taobao/"
    if not os.path.exists(to_path):
        os.mkdir(to_path)
    data = pd.read_csv("taobao/list_data.csv")
    sequence_length = 20
    step_size = 1

    def create_sequences(values, window_size, step_size):
        values = values[1:-1].split(',')
        values = list(map(int, values))
        sequences = []
        start_index = 0
        while True:
            end_index = start_index + window_size
            seq = values[start_index:end_index]
            if len(seq) < window_size:
                seq = values[-window_size:]
                if len(seq) == window_size:
                    sequences.append(seq)
                break
            sequences.append(seq)
            start_index += step_size
        return sequences

    for col in data.columns:
        if col[-1] == 's':
            data[col] = data[col].apply(
                lambda xs: create_sequences(xs, sequence_length, step_size)
            )
    data = data.explode([col for col in data.columns if col[-1] == 's'], ignore_index=True)
    data = data[data.isna().sum(axis=1) == 0]
    # data.to_csv(to_path+"data.csv", index=False, sep=',')

    data['label'] = data['labels'].apply(lambda x: x[-1])
    data.drop('labels', axis=1, inplace=True)

    data['cur_time'] = data['timestamps'].apply(lambda x: x[-1])
    train, val, test = split_train_val_test_by_time(data, 'user_id', 'cur_time')
    train.to_csv(to_path+"train_data.csv", index=False, sep=',')
    val.to_csv(to_path+"val_data.csv", index=False, sep=',')
    test.to_csv(to_path+"test_data.csv", index=False, sep=',')


if __name__ == '__main__':
    gen_taobao_list_data()
    gen_taobao_data()
