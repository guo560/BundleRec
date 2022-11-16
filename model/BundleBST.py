import math
from argparse import ArgumentParser

from .Utils import get_best_confusion_matrix
import torch
import os, platform, pickle
import torch.utils.data as data
import pytorch_lightning as pl
import torchmetrics
from torch import nn
from data.bundle_dataset import BundleDataset


class PositionEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionEmbedding, self).__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).repeat(batch_size, 1).to(x.device)
        return self.pe(pos)


class TimeEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super(TimeEmbedding, self).__init__()
        self.te = nn.Embedding(max_len, d_model)

    def forward(self, timestamps):
        """
        :param timestamps: [batch_size, seq_len]
        :return: [batch_size, seq_len, d_model]
        """
        timestamps = torch.div(timestamps, 3600, rounding_mode='floor')
        seq_len = timestamps.shape[1]
        cur_time = timestamps[:, -1]
        delta_times = cur_time.repeat(seq_len, 1).transpose(0, 1) - timestamps
        deltas = torch.log(delta_times + 1)
        return self.te(torch.ceil(deltas))


class BundleBST(pl.LightningModule):
    def __init__(self, args=None):
        super().__init__()
        super(BundleBST, self).__init__()

        self.save_hyperparameters()

        self.args = args
        self.spare_features = ['uid', 'is_visitor', 'gender', 'impression_result', 'is_up_to_date_version',
                               'register_country', 'register_device', 'register_device_brand', 'register_os',
                               'bundle_id']
        self.dense_features = ['register_time', 'bundle_price', 'island_no', 'spin_stock', 'coin_stock',
                               'diamond_stock', 'island_complete_gap_coin', 'island_complete_gap_building_cnt',
                               'tournament_rank', 'login_cnt_1d', 'ads_watch_cnt_1d', 'register_country_arpu',
                               'life_time', 'star', 'battle_pass_exp', 'pet_heart_stock', 'pet_exp_stock', 'friend_cnt',
                               'social_friend_cnt', 'pay_amt', 'pay_cnt', 'pay_per_day', 'pay_mean']

        self.user_default_col = ['uid', 'register_country', 'register_time', 'is_visitor', 'register_device',
                                 'register_device_brand', 'register_os', 'gender']
        self.user_changeable_col = ['bundle_id', 'bundle_price', 'island_no', 'spin_stock', 'coin_stock',
                                    'diamond_stock', 'island_complete_gap_coin', 'island_complete_gap_building_cnt',
                                    'tournament_rank', 'login_cnt_1d', 'ads_watch_cnt_1d', 'register_country_arpu',
                                    'life_time', 'star', 'battle_pass_exp', 'is_up_to_date_version', 'pet_heart_stock',
                                    'pet_exp_stock', 'friend_cnt', 'social_friend_cnt', 'pay_amt', 'pay_cnt',
                                    'pay_per_day', 'pay_mean']
        self.target_col = 'impression_result'

        self.embedding_dict = nn.ModuleDict()
        with open(os.path.join(args.data_path, "lbes.pkl"), 'rb') as file:
            self.lbes = pickle.load(file)
        with open(os.path.join(args.data_path, "mms.pkl"), 'rb') as file:
            self.mms = pickle.load(file)
        for i in range(len(self.spare_features)):
            self.embedding_dict[self.spare_features[i]] = nn.Embedding(
                self.lbes[i].classes_.size + 1, int(math.sqrt(self.lbes[i].classes_.size)), padding_idx=-1
            )
        self.time_embedding = TimeEmbedding(100, 26)
        self.transformerlayer = nn.TransformerEncoderLayer(26, 2)  # TODO: 为什么d_model必须要整除num_heads
        self.linear = nn.Sequential(
            nn.Linear(
                727,
                1024,
            ),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 2),
        )
        self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 483]).to(self.device))
        self.softmax_func = nn.Softmax(dim=1)
        self.auc = torchmetrics.AUROC(num_classes=2)

    def padding(self, item, padding_num: int):
        for col in self.user_changeable_col:
            if col in self.spare_features:
                item[col] = torch.where(item[col] == padding_num, self.lbes[self.spare_features.index(col)].classes_.size, item[col])
            else:
                item[col] = torch.where(item[col] == padding_num, 0, item[col])
        return item

    def encode_input(self, batch):
        item, target = batch
        user_features = [self.embedding_dict[col](item[col]) if col in self.spare_features
                         else torch.unsqueeze(item[col].to(torch.float32), dim=1)
                         for col in self.user_default_col]
        user_features = torch.cat(user_features, dim=1)

        item = self.padding(item, padding_num=-1)
        transformer_features = [self.embedding_dict[col](torch.LongTensor(item[col].cpu().numpy()).to(self.device)) if col in self.spare_features
                                else torch.unsqueeze(item[col], dim=2)
                                for col in self.user_changeable_col]
        transformer_features = torch.cat(transformer_features, dim=2)
        return user_features, transformer_features, item['ftime'], target

    def forward(self, batch):
        user_features, transformer_features, timestamps, target = self.encode_input(batch)
        time_embeddings = self.time_embedding(timestamps)  # TODO: position encoding
        transformer_features = transformer_features + time_embeddings
        transformer_output = self.transformerlayer(transformer_features)
        transformer_output = torch.flatten(transformer_output, start_dim=1)

        features = torch.cat((user_features, transformer_output), dim=1)

        output = self.linear(features)
        return output, target

    def training_step(self, batch, batch_idx):
        output, target = self(batch)

        loss = self.criterion(output, target)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False)

        return {"loss": loss, "y_pre": output, "y": target}

    def training_epoch_end(self, outputs):
        y_pre = torch.cat([x["y_pre"] for x in outputs])
        y = torch.cat([x["y"] for x in outputs])
        auc = self.auc(self.softmax_func(y_pre), y)
        matrix, metrics = get_best_confusion_matrix(y.detach().cpu(), self.softmax_func(y_pre)[:, 1:].detach().cpu())
        self.print(matrix)
        self.log("train/auc", auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log_dict({f"train/{k}": v for k, v in metrics.items()}, on_step=False, on_epoch=True, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        output, target = self(batch)
        return {"y_pre": output, "y": target}

    def validation_epoch_end(self, outputs):
        y_pre = torch.cat([x["y_pre"] for x in outputs])
        y = torch.cat([x["y"] for x in outputs])
        loss = self.criterion(y_pre, y)
        auc = self.auc(self.softmax_func(y_pre), y)

        matrix, metrics = get_best_confusion_matrix(y.cpu(), self.softmax_func(y_pre)[:, 1:].cpu())
        self.print(matrix)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auc", auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict({f"val/{k}": v for k, v in metrics.items()}, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr)

    def setup(self, stage=None):
        self.train_dataset = BundleDataset(os.path.join(self.args.data_path, "train_data.csv"))
        self.val_dataset = BundleDataset(os.path.join(self.args.data_path, "test_data.csv"), True)
        self.test_dataset = BundleDataset(os.path.join(self.args.data_path, "test_data.csv"), True)
        print("Done")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0 if platform.system() == "Windows" else os.cpu_count()
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0 if platform.system() == "Windows" else os.cpu_count()
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0 if platform.system() == "Windows" else os.cpu_count()
        )

