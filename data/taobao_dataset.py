from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np


class TaobaoDataset(Dataset):
    def __init__(self, data_path, max_len, test=False):
        self.df = pd.read_csv(data_path, decimal=',', nrows=None)
        print(f"loaded {data_path}")
        self.test = test
        user_default_col = ['user_id']
        user_changeable_col = ['timestamps', 'product_ids', 'product_category_ids']
        target_col = ['label']

        def str2ndarray(x: str):
            x = x[1:-1].split(",")
            seq = np.full(max_len, -1, dtype=np.int32)
            seq[-len(x):] = x
            return seq

        for col_name in user_changeable_col:
            self.df[col_name] = self.df[col_name].apply(str2ndarray)
        print("Transformed done")

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        data = self.df.iloc[index]
        return data.to_dict()


if __name__ == '__main__':
    bd = TaobaoDataset("F:/Taobao/train_data.csv", 20)
    dl = DataLoader(bd, batch_size=512, shuffle=False, num_workers=0)
    sampler = next(iter(bd))
    sampler2 = next(iter(dl))
    for sampler2 in dl:
        print(sampler2)
