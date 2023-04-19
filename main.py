import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser
from data.bundle_dataset import BundleDataset
from data.ele_dataset import EleDataset
from data.movie_dataset import MovieDataset
from data.taobao_dataset import TaobaoDataset
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
                args=args)
    logger = TensorBoardLogger(save_dir='./bundle_logs',
                               name='bundle_BST_final',
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


def ele_main(args, v_num, train_dataset, val_dataset, test_dataset):
    spare_features = ['user_id', 'gender', 'visit_city', 'is_super_vip', 'city_id', 'district_id', 'shop_geohash_12',
                      'geohash12', 'shop_id_list', 'item_id_list', 'category_1_id_list', 'merge_standard_food_id_list',
                      'brand_id_list', 'shop_aoi_id_list', 'shop_geohash_6_list', 'hours_list', 'time_type_list',
                      'weekdays_list']
    dense_features = ['avg_price', 'ctr_30', 'ord_30', 'total_amt_30', 'rank_7', 'rank_30', 'rank_90']
    dnn_col = ['user_id', 'gender', 'visit_city', 'avg_price', 'is_super_vip', 'ctr_30', 'ord_30',
               'total_amt_30', 'city_id', 'district_id', 'shop_geohash_12', 'rank_7', 'rank_30', 'rank_90',
               'geohash12']
    transformer_col = ['shop_id_list', 'item_id_list', 'category_1_id_list', 'merge_standard_food_id_list',
                       'brand_id_list', 'shop_aoi_id_list', 'shop_geohash_6_list', 'hours_list',
                       'time_type_list', 'weekdays_list']
    target_col = 'label'
    time_col = 'timediff_list'
    pl.seed_everything(args.seed)
    model = BST(spare_features=spare_features,
                dense_features=dense_features,
                dnn_col=dnn_col,
                transformer_col=transformer_col,
                target_col=target_col,
                time_col=time_col,
                args=args)
    logger = TensorBoardLogger(save_dir='./ele_logs/ele_BST',
                               name='fixed',
                               version=v_num)
    callback = ModelCheckpoint(monitor="val/auc",
                               mode="max",
                               save_top_k=1,
                               filename='epoch={epoch}-step={step}-val_auc={val/auc:.4f}-log_loss={val/loss:.4f}',
                               save_weights_only=True,
                               auto_insert_metric_name=False)
    callback2 = EarlyStopping(monitor="val/auc",
                              mode="max",
                              patience=1)
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


def movie_main(args, v_num, train_dataset, val_dataset, test_dataset):
    dnn_col = ['user_id', 'sex', 'age_group', 'occupation', 'zip_code']
    transformer_col = ['movie_ids', 'years', 'Actions', 'Adventures', 'Animations', 'Childrens',
                               'Comedys', 'Crimes', 'Documentarys', 'Dramas', 'Fantasys', 'Film_Noirs', 'Horrors',
                               'Musicals', 'Mysterys', 'Romances', 'Sci_Fis', 'Thrillers', 'Wars', 'Westerns']

    spare_features = dnn_col + transformer_col
    dense_features = []
    target_col = 'label'
    time_col = 'timestamps'
    pl.seed_everything(args.seed)
    model = BST(spare_features=spare_features,
                dense_features=dense_features,
                dnn_col=dnn_col,
                transformer_col=transformer_col,
                target_col=target_col,
                time_col=time_col,
                args=args)
    logger = TensorBoardLogger(save_dir='./movie_logs',
                               name='movie_BST',
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


def taobao_main(args, v_num, train_dataset, val_dataset, test_dataset):
    dnn_col = ['user_id']
    transformer_col = ['product_ids', 'product_category_ids']

    spare_features = dnn_col + transformer_col
    dense_features = []
    target_col = 'label'
    time_col = 'timestamps'
    pl.seed_everything(args.seed)
    model = BST(spare_features=spare_features,
                dense_features=dense_features,
                dnn_col=dnn_col,
                transformer_col=transformer_col,
                target_col=target_col,
                time_col=time_col,
                args=args)
    logger = TensorBoardLogger(save_dir='./taobao_logs',
                               name='taobao_BST',
                               version=v_num)
    callback = ModelCheckpoint(monitor="val/auc",
                               mode="max",
                               save_top_k=1,
                               filename='epoch={epoch}-step={step}-val_auc={val/auc:.4f}-log_loss={val/loss:.4f}',
                               save_weights_only=True,
                               auto_insert_metric_name=False)
    callback2 = EarlyStopping(monitor="val/auc",
                              mode="max",
                              patience=3)
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


def run_bundle_main():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--use_time', default=True, type=bool)
    parser.add_argument('--use_int', default=True, type=bool)
    parser.add_argument('--int_num', default=2, type=int)
    parser.add_argument('--log_base', default=10, type=float)
    parser.add_argument("--transformer_num", default=2)
    parser.add_argument("--embedding", default=8)
    parser.add_argument("--num_head", default=8)
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--data_path', default="./data/bundle_time", type=str)
    parser.add_argument('--max_len', default=8, type=int)
    parser.set_defaults(max_epochs=50)
    args = parser.parse_args()

    train_dataset = BundleDataset(os.path.join(args.data_path, "train_data.csv"), args.max_len)
    val_dataset = BundleDataset(os.path.join(args.data_path, "val_data.csv"), args.max_len)
    test_dataset = BundleDataset(os.path.join(args.data_path, "test_data.csv"), args.max_len, True)

    v_num = None
    # for seed in [0]:
    #     for use_time, log_base in [(True, 10)]:
    #         for use_int, int_num, embedding in [(True, 1, 8), (True, 2, 8), (True, 3, 8)]:
    #             args.seed, args.log_base, args.int_num, args.use_time, args.use_int, args.embedding \
    #                 = seed, log_base, int_num, use_time, use_int, embedding
    #             bundle_main(args, v_num, train_dataset, val_dataset, test_dataset)

    for seed in [0]:
        for use_time, log_base in [(True, 10)]:
            for use_int, int_num, embedding in [(True, -1, 8)]:
                args.seed, args.log_base, args.int_num, args.use_time, args.use_int, args.embedding \
                    = seed, log_base, int_num, use_time, use_int, embedding
                bundle_main(args, v_num, train_dataset, val_dataset, test_dataset)

    for seed in [0]:
        for use_time, log_base in [(True, -2), (True, -3)]:
            for use_int, int_num, embedding in [(False, 0, 9)]:
                args.seed, args.log_base, args.int_num, args.use_time, args.use_int, args.embedding \
                    = seed, log_base, int_num, use_time, use_int, embedding
                bundle_main(args, v_num, train_dataset, val_dataset, test_dataset)


def run_ele_main():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--use_time', default=True, type=bool)
    parser.add_argument('--use_int', default=False, type=bool)
    parser.add_argument('--int_num', default=2, type=int)
    parser.add_argument('--log_base', default=2, type=float)
    parser.add_argument("--transformer_num", default=1, type=int)
    parser.add_argument("--embedding", default=8)
    parser.add_argument("--num_head", default=8, type=int)
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--data_path', default="./data/ele_time", type=str)
    parser.add_argument('--max_len', default=51, type=int)
    parser.set_defaults(max_epochs=50)
    args = parser.parse_args()

    train_dataset = EleDataset(os.path.join(args.data_path, "train_data.csv"), args.max_len)
    val_dataset = EleDataset(os.path.join(args.data_path, "val_data.csv"), args.max_len)
    test_dataset = EleDataset(os.path.join(args.data_path, "test_data.csv"), args.max_len, True)

    v_num = None

    for seed in [0]:
        for transformer_num in [1]:
            for use_time, log_base in [(True, 5), (False, 0)]:
                for use_int, int_num in [(True, 1), (True, 2), (True, 3)]:
                    for lr in [1e-5]:
                        args.seed, args.transformer_num, args.log_base, args.int_num, args.use_time, args.use_int, args.lr \
                                = seed, transformer_num, log_base, int_num, use_time, use_int, lr
                        ele_main(args, f"lr=1e-5 mean_std=1 res=concat log_base={log_base} int_num={int_num}", train_dataset, val_dataset, test_dataset)


def run_movie_main():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--use_time', default=True, type=bool)
    parser.add_argument('--use_int', default=True, type=bool)
    parser.add_argument('--int_num', default=1, type=int)
    parser.add_argument('--log_base', default=10, type=float)
    parser.add_argument("--transformer_num", default=1)
    parser.add_argument("--embedding", default=8)
    parser.add_argument("--num_head", default=8)
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--data_path', default="./data/movielens1m", type=str)
    parser.add_argument('--max_len', default=8, type=int)
    parser.set_defaults(max_epochs=50)
    args = parser.parse_args()

    train_dataset = MovieDataset(os.path.join(args.data_path, "train_data.csv"), args.max_len)
    val_dataset = MovieDataset(os.path.join(args.data_path, "val_data.csv"), args.max_len)
    test_dataset = MovieDataset(os.path.join(args.data_path, "test_data.csv"), args.max_len, True)

    v_num = None

    for seed in [123456]:
        for transformer_num in [1]:
            for use_time, log_base in [(False, 0)]:
                for use_int, int_num in [(True, -1)]:
                    args.seed, args.transformer_num, args.log_base, args.int_num, args.use_time, args.use_int \
                        = seed, transformer_num, log_base, int_num, use_time, use_int
                    movie_main(args, v_num, train_dataset, val_dataset, test_dataset)
    for seed in [123456]:
        for transformer_num in [1]:
            for use_time, log_base in [(True, -2)]:
                for use_int, int_num in [(False, 0)]:
                    args.seed, args.transformer_num, args.log_base, args.int_num, args.use_time, args.use_int \
                        = seed, transformer_num, log_base, int_num, use_time, use_int
                    movie_main(args, v_num, train_dataset, val_dataset, test_dataset)


def run_taobao_main():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--use_time', default=True, type=bool)
    parser.add_argument('--use_int', default=False, type=bool)
    parser.add_argument('--int_num', default=0, type=int)
    parser.add_argument('--log_base', default=10, type=float)
    parser.add_argument("--transformer_num", default=1)
    parser.add_argument("--embedding", default=8)
    parser.add_argument("--num_head", default=8)
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--data_path', default="./data/taobao", type=str)
    parser.add_argument('--max_len', default=20, type=int)
    parser.set_defaults(max_epochs=50)
    args = parser.parse_args()

    train_dataset = TaobaoDataset(os.path.join(args.data_path, "train_data.csv"), args.max_len)
    val_dataset = TaobaoDataset(os.path.join(args.data_path, "val_data.csv"), args.max_len)
    test_dataset = TaobaoDataset(os.path.join(args.data_path, "test_data.csv"), args.max_len, True)
    v_num = None
    for seed in [0]:
        for use_time, log_base in [(True, -2)]:
            for use_int, int_num in [(False, 0)]:
                args.seed, args.log_base, args.use_time, args.use_int, args.int_num \
                    = seed, log_base, use_time, use_int, int_num
                taobao_main(args, v_num, train_dataset, val_dataset, test_dataset)


if __name__ == '__main__':
    run_bundle_main()
