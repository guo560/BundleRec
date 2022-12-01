import math
import torch
import os, platform, pickle
import torch.utils.data as data
import pytorch_lightning as pl
import torchmetrics
from .utils import get_best_confusion_matrix
from torch import nn


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
    def __init__(self, max_len, d_model, log_base):
        super(TimeEmbedding, self).__init__()
        self.te = nn.Embedding(max_len, d_model)
        self.log_base = log_base

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
        deltas = torch.div(deltas, math.log(self.log_base))
        return self.te(torch.ceil(deltas).long())


class BST(pl.LightningModule):
    def __init__(self, spare_features, dense_features, dnn_col, transformer_col, target_col, time_col,
                 seq_len, args):
        super().__init__()
        super(BST, self).__init__()

        self.spare_features = spare_features
        self.dense_features = dense_features
        self.dnn_col = dnn_col
        self.transformer_col = transformer_col
        self.target_col = target_col
        self.time_col = time_col
        self.seq_len = seq_len
        self.save_hyperparameters(args, logger=False)

        with open(os.path.join(self.hparams.data_path, "lbes.pkl"), 'rb') as file:
            self.lbes = pickle.load(file)
        with open(os.path.join(self.hparams.data_path, "mms.pkl"), 'rb') as file:
            self.mms = pickle.load(file)

        self.embedding_dict = nn.ModuleDict()
        for i in range(len(self.spare_features)):
            self.embedding_dict[self.spare_features[i]] = nn.Embedding(
                self.lbes[i].classes_.size + 1,
                int(math.sqrt(self.lbes[i].classes_.size)) if self.hparams.embedding == -1 else self.hparams.embedding,
                padding_idx=-1
            )
        self.d_transformer = sum([self.embedding_dict[col].embedding_dim if col in spare_features else 1 for col in transformer_col])
        self.d_dnn = sum([self.embedding_dict[col].embedding_dim if col in spare_features else 1 for col in dnn_col])
        self.time_embedding = TimeEmbedding(100, self.d_transformer, self.hparams.log_base)
        self.position_embedding = PositionEmbedding(8, self.d_transformer)
        self.transformerlayers = nn.ModuleList(
            [nn.TransformerEncoderLayer(self.d_transformer, self.hparams.num_head, batch_first=True).to(self.device) for _ in range(self.hparams.transformer_num)]
        )  # TODO: 为什么d_model必须要整除num_heads
        self.linear = nn.Sequential(
            nn.Linear(
                self.d_dnn + self.d_transformer * seq_len,
                1024,
            ),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 2),
        ).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.softmax_func = nn.Softmax(dim=1).to(self.device)
        self.auc = torchmetrics.AUROC(num_classes=2).to(self.device)

    def padding(self, item, padding_num: int):
        for col in self.transformer_col:
            if col in self.spare_features:
                item[col] = torch.where(item[col] == padding_num, self.lbes[self.spare_features.index(col)].classes_.size, item[col])
            else:
                item[col] = torch.where(item[col] == padding_num, 0, item[col])
        return item

    def gen_mask(self, item, padding_num: int):
        col = self.transformer_col[0]
        mask = torch.zeros(item[col].size()).bool().to(self.device)
        mask = torch.where(item[col] == padding_num, True, mask)
        return mask

    def encode_input(self, batch):
        item = batch
        target = item[self.target_col].long()
        mask = self.gen_mask(item, padding_num=-1)
        item = self.padding(item, padding_num=-1)
        for col in self.spare_features:
            item[col] = self.embedding_dict[col](item[col].long())
        for col in self.dense_features:
            item[col] = item[col].float().unsqueeze(dim=-1)
        dnn_input = torch.cat([item[col] for col in self.dnn_col], dim=-1)
        transformer_input = torch.cat([item[col] for col in self.transformer_col], dim=-1)
        return dnn_input, transformer_input, item[self.time_col], target, mask

    def forward(self, batch):
        dnn_input, transformer_input, timestamp, target, mask = self.encode_input(batch)

        if self.hparams.use_time:
            transformer_output = transformer_input + self.time_embedding(timestamp)
        else:
            transformer_output = transformer_input + self.position_embedding(transformer_input)

        for i in range(len(self.transformerlayers)):
            transformer_output = self.transformerlayers[i](transformer_output, src_key_padding_mask=mask)
        transformer_output = torch.flatten(transformer_output, start_dim=1)

        dnn_input = torch.cat((dnn_input, transformer_output), dim=1)

        output = self.linear(dnn_input)
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
        matrix, metrics = get_best_confusion_matrix(y.detach().cpu(), self.softmax_func(y_pre)[:, 1].detach().cpu())
        # self.print(matrix)
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

        matrix, metrics = get_best_confusion_matrix(y.cpu(), self.softmax_func(y_pre)[:, 1].cpu())
        # self.print(matrix)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/auc", auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log_dict({f"val/{k}": v for k, v in metrics.items()}, on_step=False, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

