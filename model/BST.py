import math
import torch
import os, pickle
import pytorch_lightning as pl
import torchmetrics
from .utils import get_best_confusion_matrix
from torch import nn
from .interacting_layer import InteractingLayer


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
        self.te = nn.Embedding(2001 if log_base == -2 else max_len, d_model, padding_idx=-1)
        self.log_base = log_base

    def forward(self, timestamps):
        """
        :param timestamps: [batch_size, seq_len]
        :return: [batch_size, seq_len, d_model]
        """
        timestamps = torch.div(timestamps, 3600, rounding_mode='floor')

        # timestamps为实际时间时使用(电影、九九、淘宝数据集中使用)
        seq_len = timestamps.shape[1]
        cur_time = timestamps.max(dim=1)[0]
        delta_times = cur_time.repeat(seq_len, 1).transpose(0, 1) - timestamps

        # timestamps为时间差时使用(饿了么数据集中使用)
        # delta_times = timestamps

        if self.log_base == -2:
            # 线性时间-位置转换函数
            delta_times = torch.where(delta_times > 2000, 2000, delta_times)
            delta_times = torch.where(delta_times == -1, 2000, delta_times)
            return self.te(delta_times.long())

        deltas = torch.log(delta_times + 1)

        if self.log_base == -1:
            seq_len = timestamps.shape[1]
            log_bases = (torch.where(delta_times <= 0, torch.inf, delta_times)).min(dim=1)[0]
            log_bases = torch.where(log_bases == torch.inf, 1, log_bases) + 1
            bases = torch.log(log_bases).repeat(seq_len, 1).transpose(0, 1)
            deltas = torch.div(deltas, bases)
        else:
            deltas = torch.div(deltas, math.log(self.log_base))
        deltas = torch.ceil(deltas).long()
        return self.te(deltas)


class BST(pl.LightningModule):
    def __init__(self, spare_features, dense_features, dnn_col, transformer_col, target_col, time_col, args):
        super().__init__()
        super(BST, self).__init__()

        self.spare_features = spare_features
        self.dense_features = dense_features
        self.dnn_col = dnn_col
        self.transformer_col = transformer_col
        self.target_col = target_col
        self.time_col = time_col
        self.save_hyperparameters(args, logger=False)

        with open(os.path.join(self.hparams.data_path, "lbes.pkl"), 'rb') as file:
            self.lbes = pickle.load(file)
        with open(os.path.join(self.hparams.data_path, "mms.pkl"), 'rb') as file:
            self.mms = pickle.load(file)

        self.embedding_dict = nn.ModuleDict()
        for feature in self.spare_features:
            self.embedding_dict[feature] = nn.Embedding(
                self.lbes[feature].classes_.size + 1,
                int(math.sqrt(self.lbes[feature].classes_.size)) if self.hparams.embedding == -1 else self.hparams.embedding,
                padding_idx=-1
            )
        if self.hparams.use_int:
            self.d_transformer = sum([self.embedding_dict[col].embedding_dim
                                      if col in spare_features
                                      else self.hparams.embedding
                                      for col in transformer_col])
            self.dense_embedding_col = list(set(transformer_col) & set(dense_features))
            self.dense_embedding_dict = nn.ModuleDict()
            for feature in self.dense_embedding_col:
                self.dense_embedding_dict[feature] = nn.Embedding(1, self.hparams.embedding)
        else:
            self.d_transformer = sum([self.embedding_dict[col].embedding_dim
                                      if col in spare_features
                                      else 1
                                      for col in transformer_col])
        self.d_dnn = sum([self.embedding_dict[col].embedding_dim if col in spare_features else 1 for col in dnn_col])
        if self.hparams.use_time:
            self.time_embedding = TimeEmbedding(50, self.d_transformer, self.hparams.log_base)
        else:
            self.position_embedding = PositionEmbedding(args.max_len, self.d_transformer)

        if self.hparams.use_int:
            self.int_layers = nn.ModuleList(
                [InteractingLayer(self.hparams.embedding, 1, device=self.device, use_res=(self.hparams.int_num > 0))
                 for _ in range(abs(self.hparams.int_num))]
            )

        self.transformerlayers = nn.ModuleList(
            [nn.TransformerEncoderLayer(self.d_transformer, self.hparams.num_head, batch_first=True).to(self.device) for _ in range(self.hparams.transformer_num)]
        )
        self.linear = nn.Sequential(
            nn.Linear(
                self.d_dnn + self.d_transformer * args.max_len,
                1024,
            ),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 2),
        ).to(self.device)
        for name, tensor in self.linear.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0.0, std=0.0001)

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.softmax_func = nn.Softmax(dim=1).to(self.device)
        self.auc = torchmetrics.AUROC(num_classes=2).to(self.device)

    def padding(self, item, padding_num: int):
        for col in self.transformer_col:
            if col in self.spare_features:
                item[col] = torch.where(item[col] == padding_num, self.lbes[col].classes_.size, item[col])
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
        if self.hparams.use_int:
            for col in self.dense_embedding_col:
                item[col] = item[col] * self.dense_embedding_dict[col].weight[0]
        dnn_input = torch.cat([item[col] for col in self.dnn_col], dim=-1)
        transformer_input = torch.cat([item[col] for col in self.transformer_col], dim=-1)
        return dnn_input, transformer_input, item[self.time_col], target, mask

    def forward(self, batch):
        dnn_input, transformer_input, timestamp, target, mask = self.encode_input(batch)

        if self.hparams.use_int:
            batch_size, seq_len, total_dim = transformer_input.shape
            embedding_dim = self.hparams.embedding
            assert total_dim % embedding_dim == 0
            transformer_input = transformer_input.view((batch_size * seq_len, total_dim // embedding_dim, embedding_dim))
            for layer in self.int_layers:
                transformer_input = layer(transformer_input)
            transformer_input = transformer_input.view((batch_size, seq_len, total_dim))

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

    def test_step(self, batch, batch_idx):
        output, target = self(batch)
        return {"y_pre": output, "y": target}

    def test_epoch_end(self, outputs):
        y_pre = torch.cat([x["y_pre"] for x in outputs])
        y = torch.cat([x["y"] for x in outputs])
        loss = self.criterion(y_pre, y)
        auc = self.auc(self.softmax_func(y_pre), y)

        matrix, metrics = get_best_confusion_matrix(y.cpu(), self.softmax_func(y_pre)[:, 1].cpu())
        # self.print(matrix)
        result = {"test/log_loss": loss,
                  "test/auc": auc,
                  **{f"test/{k}": v for k, v in metrics.items()}}
        self.logger.log_hyperparams(self.hparams, metrics=result)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

