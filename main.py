import copy
import gc
import os
import platform
import time

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from multiprocessing.pool import Pool
from multiprocessing import Manager
from argparse import ArgumentParser
from data.bundle_dataset import BundleDataset
from model.BST import BST


def bundle_main(args, v_num, train_dataset, test_dataset):
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
                               name='bundle_BST',
                               version=v_num)
    callback = ModelCheckpoint(monitor="val/auc",
                               mode="max",
                               save_top_k=1,
                               filename='epoch={epoch}-step={step}-val_auc={val/auc:.2f}',
                               auto_insert_metric_name=False)
    trainer = pl.Trainer(accelerator='gpu',
                         devices=1,
                         callbacks=[callback],
                         # auto_scale_batch_size='binsearch',
                         # auto_lr_find=True,
                         val_check_interval=0.5,
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
                val_dataloaders=DataLoader(test_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           num_workers=0
                                           )
                )
    logger.log_hyperparams(model.hparams, {"val_auc": callback.best_model_score})


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--use_time', default=True, type=bool)
    parser.add_argument("--transformer_num", default=3)
    parser.add_argument("--embedding", default=9)
    parser.add_argument("--num_head", default=8)
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--masked', default=True, type=bool)
    parser.add_argument('--data_path', default="./data/bundle_time", type=str)
    parser.set_defaults(max_epochs=5)
    args = parser.parse_args()

    # manager = Manager()
    # list = manager.list()
    # train_dataset = BundleDataset(os.path.join(args.data_path, "train_data.csv"))
    # val_dataset = BundleDataset(os.path.join(args.data_path, "test_data.csv"), True)
    # print("data loaded")
    # list.append(val_dataset)
    # del val_dataset
    # gc.collect()
    # list.append(train_dataset)
    # del train_dataset
    # gc.collect()

    # pool = Pool(2)

    # for batch_size in [256, 512, 1024]:
    #     for lr in [1e-2, 1e-3, 1e-4]:
    #         for use_time in [True, False]:
    #             for transformer_num in [1, 2, 3]:
    #                 for (embedding, num_head) in [(9, 8), (-1, 2), (-1, 13)]:
    #                     args.batch_size, args.lr, args.use_time, args.transformer_num, args.embedding, args.num_head = \
    #                         batch_size, lr, use_time, transformer_num, embedding, num_head
    #                     pool.apply_async(func=bundle_main, args=(copy.deepcopy(args), list, v_num))
    #                     v_num += 1
    # pool.close()
    # pool.join()

    v_num = 0
    train_dataset = BundleDataset(os.path.join(args.data_path, "train_data.csv"))
    test_dataset = BundleDataset(os.path.join(args.data_path, "test_data.csv"), True)
    for batch_size in [512, 1024]:
        for lr in [1e-3, 1e-4]:
            for use_time in [True, False]:
                for transformer_num in [1, 2, 3]:
                    for (embedding, num_head) in [(9, 8), (9, 1), (-1, 1)]:
                        args.batch_size, args.lr, args.use_time, args.transformer_num, args.embedding, args.num_head = \
                            batch_size, lr, use_time, transformer_num, embedding, num_head
                        if v_num >= 0:
                            bundle_main(args, v_num, train_dataset, test_dataset)
                        v_num += 1

    # bundle_main(args)
