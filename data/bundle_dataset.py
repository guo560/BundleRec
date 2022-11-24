import gc
import time

import pandas
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

from model.utils import timeit


class BundleDataset(Dataset):
    def __init__(self, data_path, test=False):
        self.df = pd.read_csv(data_path, decimal=',')
        print(f"loaded {data_path}")
        self.test = test
        user_default_col = ['uid', 'register_country', 'register_time', 'is_visitor', 'register_device',
                                 'register_device_brand', 'register_os', 'gender']
        user_changeable_col = ['ftime', 'bundle_id', 'bundle_price', 'island_no', 'spin_stock', 'coin_stock',
                                    'diamond_stock', 'island_complete_gap_coin', 'island_complete_gap_building_cnt',
                                    'tournament_rank', 'login_cnt_1d', 'ads_watch_cnt_1d', 'register_country_arpu',
                                    'life_time', 'star', 'battle_pass_exp', 'is_up_to_date_version', 'pet_heart_stock',
                                    'pet_exp_stock', 'friend_cnt', 'social_friend_cnt', 'pay_amt', 'pay_cnt',
                                    'pay_per_day', 'pay_mean']
        target_col = ['impression_result']
        for col_name in user_changeable_col:
            self.df[col_name] = self.df[col_name].apply(lambda x: torch.FloatTensor([float(i) for i in x[1:-1].split(",")]))
        self.df['register_time'] = self.df['register_time'].apply(lambda x: float(x))
        self.df[target_col[0]] = self.df[target_col[0]].apply(lambda x: float(x[1:-1].split(",")[-1]))
        print("Transformed done")

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        data = self.df.iloc[index]
        return data.to_dict()


if __name__ == '__main__':
    bd = BundleDataset("bundle_time/test_data.csv")
    dl = DataLoader(bd, batch_size=2, shuffle=True, num_workers=0, pin_memory=False)
    print(next(iter(dl)))
