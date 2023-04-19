from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np


class MovieDataset(Dataset):
    def __init__(self, data_path, max_len, test=False):
        self.df = pd.read_csv(data_path, decimal=',')
        print(f"loaded {data_path}")
        self.test = test
        user_default_col = ['user_id', 'sex', 'age_group', 'occupation', 'zip_code']
        user_changeable_col = ['timestamps', 'movie_ids', 'years', 'Actions', 'Adventures', 'Animations', 'Childrens',
                               'Comedys', 'Crimes', 'Documentarys', 'Dramas', 'Fantasys', 'Film_Noirs', 'Horrors',
                               'Musicals', 'Mysterys', 'Romances', 'Sci_Fis', 'Thrillers', 'Wars', 'Westerns']
        target_col = ['label']

        def str2ndarray(x: str):
            x = x[1:-1].split(",")
            seq = np.full(max_len, -1, dtype=np.float32)
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
    bd = MovieDataset("movielens1m/train_data.csv", 8)
    dl = DataLoader(bd, batch_size=512, shuffle=False, num_workers=0)
    sampler = next(iter(bd))
    sampler2 = next(iter(dl))
    for sampler2 in dl:
        print(sampler2)
