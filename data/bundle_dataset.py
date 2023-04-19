from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np


class BundleDataset(Dataset):
    def __init__(self, data_path, max_len, test=False):
        self.df = pd.read_csv(data_path, decimal=',', nrows=None)
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

        def str2ndarray(x: str):
            x = x[1:-1].split(",")
            seq = np.full(max_len, -1, dtype=np.float32)
            seq[-len(x):] = x
            return seq

        for col_name in user_changeable_col:
            self.df[col_name] = self.df[col_name].apply(str2ndarray)
        self.df['register_time'] = self.df['register_time'].apply(lambda x: np.float32(x))
        self.df[target_col[0]] = self.df[target_col[0]].apply(lambda x: np.float32(x[1:-1].split(",")[-1]))
        print("Transformed done")

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        data = self.df.iloc[index]
        return data.to_dict()


if __name__ == '__main__':
    bd = BundleDataset("bundle_new_time/train_data.csv", 8)
    dl = DataLoader(bd, batch_size=512, shuffle=False, num_workers=0)
    sampler = next(iter(bd))
    sampler2 = next(iter(dl))
    for sampler2 in dl:
        print(sampler2)
