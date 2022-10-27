import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd


class BundleDataset(Dataset):
    def __init__(self, data_path, test=False):
        self.df = pd.read_csv(data_path, decimal=',')
        self.test = test
        self.user_default_col = ['uid', 'register_country', 'register_time', 'is_visitor', 'register_device',
                                 'register_device_brand', 'register_os', 'gender']
        self.user_changeable_col = ['ftime', 'bundle_id', 'bundle_price', 'island_no', 'spin_stock', 'coin_stock',
                                    'diamond_stock', 'island_complete_gap_coin', 'island_complete_gap_building_cnt',
                                    'tournament_rank', 'login_cnt_1d', 'ads_watch_cnt_1d', 'register_country_arpu',
                                    'life_time', 'star', 'battle_pass_exp', 'is_up_to_date_version', 'pet_heart_stock',
                                    'pet_exp_stock', 'friend_cnt', 'social_friend_cnt', 'pay_amt', 'pay_cnt',
                                    'pay_per_day', 'pay_mean']
        self.target_col = 'impression_result'
        self.default_features = ['uid', 'is_visitor', 'gender', 'ftime', 'impression_result', 'is_up_to_date_version']
        self.spare_features = ['register_country', 'register_device', 'register_device_brand', 'register_os',
                               'bundle_id']
        self.dense_features = ['register_time', 'bundle_price', 'island_no', 'spin_stock', 'coin_stock',
                               'diamond_stock', 'island_complete_gap_coin', 'island_complete_gap_building_cnt',
                               'tournament_rank', 'login_cnt_1d', 'ads_watch_cnt_1d', 'register_country_arpu',
                               'life_time', 'star', 'battle_pass_exp', 'pet_heart_stock', 'pet_exp_stock', 'friend_cnt',
                               'social_friend_cnt', 'pay_amt', 'pay_cnt', 'pay_per_day', 'pay_mean']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        data = self.df.iloc[index]
        item = dict()
        for col_name in self.user_default_col:
            item[col_name] = data[col_name]
        for col_name in self.user_changeable_col:
            item[col_name] = eval(data[col_name])
        item[self.target_col] = eval(data[self.target_col])
        target = item[self.target_col][-1]
        return item, target


if __name__ == '__main__':
    bd = BundleDataset("bundle/train_data.csv")
    print(next(iter(bd)))
