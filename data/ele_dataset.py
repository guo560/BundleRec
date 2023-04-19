from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class EleDataset(Dataset):
    def __init__(self, data_path, max_len, test=False):
        dnn_col = ['user_id', 'gender', 'visit_city', 'avg_price', 'is_super_vip', 'ctr_30', 'ord_30',
                   'total_amt_30', 'city_id', 'district_id', 'shop_geohash_12', 'rank_7', 'rank_30', 'rank_90',
                   'geohash12']
        transformer_col = ['shop_id_list', 'item_id_list', 'category_1_id_list', 'merge_standard_food_id_list',
                           'brand_id_list', 'shop_aoi_id_list', 'shop_geohash_6_list', 'timediff_list', 'hours_list',
                           'time_type_list', 'weekdays_list']
        target_col = ['label']
        self.df = pd.read_csv(data_path, sep=',', usecols=dnn_col + transformer_col + target_col, nrows=None)
        print(f"loaded {data_path}")
        self.test = test

        def str2ndarray(x: list):
            seq = np.zeros(max_len, dtype=np.float32)
            length = len(x)
            seq[:length] = x
            seq[length:] = -1
            return seq

        for col_name in transformer_col:
            self.df[col_name] = self.df[col_name].str.split(';').apply(str2ndarray)
        print("Transformed done")

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        data = self.df.iloc[index]
        return data.to_dict()


if __name__ == '__main__':
    bd = EleDataset("ele_time/train_data.csv", 51)
    pass