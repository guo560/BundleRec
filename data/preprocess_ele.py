import os.path
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from gen_sequence import split_train_val_test_by_time


def gen_ele_data():
    path = "ele_time/"
    if not os.path.exists(path):
        os.mkdir(path)
    cols = ['label', 'user_id', 'gender', 'visit_city', 'avg_price', 'is_super_vip', 'ctr_30', 'ord_30', 'total_amt_30',

            'shop_id', 'item_id', 'city_id', 'district_id', 'shop_aoi_id', 'shop_geohash_6', 'shop_geohash_12',
            'brand_id', 'category_1_id', 'merge_standard_food_id', 'rank_7', 'rank_30', 'rank_90',

            'shop_id_list', 'item_id_list', 'category_1_id_list', 'merge_standard_food_id_list', 'brand_id_list',
            'price_list', 'shop_aoi_id_list', 'shop_geohash_6_list', 'timediff_list', 'hours_list', 'time_type_list',
            'weekdays_list',

            'times', 'hours', 'time_type', 'weekdays', 'geohash12']

    spare_features = ['user_id', 'gender', 'visit_city', 'is_super_vip',

                      'shop_id', 'item_id', 'city_id', 'district_id', 'shop_aoi_id', 'shop_geohash_6', 'shop_geohash_12',
                      'brand_id', 'category_1_id', 'merge_standard_food_id', 'hours', 'time_type', 'weekdays', 'geohash12']

    dense_features = ['avg_price', 'ctr_30', 'ord_30', 'total_amt_30', 'rank_7', 'rank_30', 'rank_90']

    list_features = ['shop_id_list', 'item_id_list', 'category_1_id_list', 'merge_standard_food_id_list', 'brand_id_list',
                     'price_list', 'shop_aoi_id_list', 'shop_geohash_6_list', 'timediff_list', 'hours_list', 'time_type_list',
                     'weekdays_list']

    df = pd.read_csv("./raw_data/Ele.me/D1_0.csv", names=cols)
    print(f"raw data rows: {df.shape[0]}")

    print("drop nan...")
    df = df[df.isna().sum(axis=1) == 0]
    print(f"remain rows: {df.shape[0]}")

    mms = MinMaxScaler()
    df[dense_features] = mms.fit_transform(df[dense_features])
    with open(path+"mms.pkl", 'wb') as file:
        pickle.dump(mms, file)

    lbes = {feature: LabelEncoder() for feature in spare_features }
    df.reset_index(inplace=True)
    df['index'] = df.index
    for feature in spare_features:
        if df[feature].dtype != np.dtype('O'):
            df[feature] = df[feature].apply(lambda x: str(int(x)))
        feature_list = feature + '_list'

        if feature_list in list_features:
            lbes[feature_list] = LabelEncoder()
            feature_full_col = df[[feature, 'index']]
            df[feature_list] = df[feature_list].str.split(";")
            # temp = df[[feature_list, 'index']].explode(feature_list)
            feature_full_col = feature_full_col.rename(columns={feature: feature_list})
            feature_full_col = pd.concat([feature_full_col, df[[feature_list, 'index']]])
            feature_full_col = feature_full_col.explode(feature_list)
            feature_full_col[feature_list] = lbes[feature_list].fit_transform(feature_full_col[feature_list])
            df[feature_list] = feature_full_col.groupby(by=['index']).agg(lambda x: ';'.join(x.apply(str)))
            lbes.pop(feature)
            df.drop(labels=feature, axis=1, inplace=True)
        else:
            df[feature] = lbes[feature].fit_transform(df[feature])
        print(f"{feature} unique count: {lbes[feature].classes_.size}")

    df['timediff_list'] = df['timediff_list'].apply(lambda x: '0;'+x)
    with open(path+"lbes.pkl", 'wb') as file:
        pickle.dump(lbes, file)

    df.drop(labels='index', axis=1, inplace=True)
    train, val, test = split_train_val_test_by_time(df, 'user_id', 'times')
    train.to_csv(path+"train_data.csv", index=False, sep=',')
    val.to_csv(path+"val_data.csv", index=False, sep=',')
    test.to_csv(path+"test_data.csv", index=False, sep=',')


if __name__ == '__main__':
    gen_ele_data()