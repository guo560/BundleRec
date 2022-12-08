import copy
import gc
import math
import os
import platform
import time

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser
from data.bundle_dataset import BundleDataset
from model.BST import BST


def bundle_main(args, v_num, train_dataset, val_dataset, test_dataset):
    spare_features = ['uid', 'is_visitor', 'gender', 'is_up_to_date_version', 'register_country', 'register_device',
                      'register_device_brand', 'register_os', 'bundle_id']
    dense_features = ['register_time', 'bundle_price', 'island_no', 'spin_stock', 'coin_stock',
                      'diamond_stock', 'island_complete_gap_coin', 'island_complete_gap_building_cnt',
                      'tournament_rank', 'login_cnt_1d', 'ads_watch_cnt_1d', 'register_country_arpu',
                      'life_time', 'star', 'battle_pass_exp', 'pet_heart_stock', 'pet_exp_stock', 'friend_cnt',
                      'social_friend_cnt', 'pay_amt', 'pay_cnt', 'pay_per_day', 'pay_mean']
    dnn_col = ['uid', 'register_country', 'register_time', 'is_visitor', 'register_device',
               'register_device_brand', 'register_os', 'gender']
    transformer_col = ['bundle_id', 'bundle_price', 'island_no', 'spin_stock', 'coin_stock',
                       'diamond_stock', 'island_complete_gap_coin', 'island_complete_gap_building_cnt',
                       'tournament_rank', 'login_cnt_1d', 'ads_watch_cnt_1d', 'register_country_arpu',
                       'life_time', 'star', 'battle_pass_exp', 'is_up_to_date_version', 'pet_heart_stock',
                       'pet_exp_stock', 'friend_cnt', 'social_friend_cnt', 'pay_amt', 'pay_cnt',
                       'pay_per_day', 'pay_mean']
    target_col = 'impression_result'
    time_col = 'ftime'
    pl.seed_everything(args.seed)
    model = BST(spare_features=spare_features,
                dense_features=dense_features,
                dnn_col=dnn_col,
                transformer_col=transformer_col,
                target_col=target_col,
                time_col=time_col,
                seq_len=8,
                args=args)
    logger = TensorBoardLogger(save_dir='./bundle_logs',
                               name='bundle_BST_tricked',
                               version=v_num)
    callback = ModelCheckpoint(monitor="val/auc",
                               mode="max",
                               save_top_k=1,
                               filename='epoch={epoch}-step={step}-val_auc={val/auc:.4f}-log_loss={val/loss:.4f}',
                               save_weights_only=True,
                               auto_insert_metric_name=False)
    callback2 = EarlyStopping(monitor="val/auc",
                              mode="max",
                              patience=5)
    trainer = pl.Trainer(accelerator='gpu',
                         devices=1,
                         callbacks=[callback, callback2],
                         # auto_scale_batch_size='binsearch',
                         # auto_lr_find=True,
                         val_check_interval=None,
                         max_epochs=args.max_epochs,
                         logger=logger,
                         log_every_n_steps=50,
                         num_sanity_val_steps=2,
                         fast_dev_run=False,
                         enable_progress_bar=True)
    model.to('cuda')
    trainer.fit(model=model,
                train_dataloaders=DataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=0
                                             ),
                val_dataloaders=DataLoader(val_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           num_workers=0
                                           ),
                )
    trainer.test(ckpt_path=callback.best_model_path,
                 dataloaders=DataLoader(test_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=0
                                        ),
                 )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--use_time', default=True, type=bool)
    parser.add_argument('--log_base', default=10, type=float)
    parser.add_argument("--transformer_num", default=2)
    parser.add_argument("--embedding", default=9)
    parser.add_argument("--num_head", default=8)
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--masked', default=True, type=bool)
    parser.add_argument('--data_path', default="./data/bundle_time", type=str)
    parser.set_defaults(max_epochs=50)
    args = parser.parse_args()

    train_dataset = BundleDataset(os.path.join(args.data_path, "train_data.csv"))
    val_dataset = BundleDataset(os.path.join(args.data_path, "val_data.csv"))
    test_dataset = BundleDataset(os.path.join(args.data_path, "test_data.csv"), True)

    v_num = -5
    # for seed in [2022, 0, 123456]:
    #     for use_tricked_embedding in [True, False]:
    #         for use_tricked_linear in [True, False]:
    #             for use_time in [True, False]:
    #                 args.use_tricked_embedding, args.use_tricked_linear, args.use_time, args.seed = \
    #                     use_tricked_embedding, use_tricked_linear, use_time, seed
    #                 bundle_main(args, v_num, train_dataset, val_dataset, test_dataset)
    #                 v_num += 1
    for batch_size in [512]:
        for use_time in [True, False]:
            args.batch_size, args.use_time = batch_size, use_time
            bundle_main(args, v_num, train_dataset, val_dataset, test_dataset)
            v_num -= 1
