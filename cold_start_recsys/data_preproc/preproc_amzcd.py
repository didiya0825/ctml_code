"""
Preprocess Amazon-CDs
"""

# ========================================================
# setting: train_split_cutoff=0.7, val_split_cutoff=0.8, min_count=30
# --------------------------------------------------------
# item_info_dat 516914
# ratings_dat 2213290
# start time 1997-09-13 08:00:00, train split time 2012-01-13 08:00:00, val split time 2013-03-05 08:00:00, end time 2014-07-23 08:00:00
# raw num train users: 731056, raw num val users: 145596, raw num test users: 275515
# preproc num train users: 3584, min count: 30, avg count: 84.33984375, max count: 2874
# preproc num val users: 202, min count: 30, avg count: 74.74752475247524,, max count: 525
# preproc num test users: 355, min count: 30, avg count: 69.09859154929578, max count: 418
# user_id_list 3876
# item_id_list 132209
# brand_list 49283
# cate_list 490
# user_dict: 3876
# item_dict: 132209
# train_df: 302274
# val_df: 15099
# test_df: 24530
# user_set: 3876
# train_user_set: 3584
# val_user_set: 202
# val_ws_user_set: 110
# val_cs_user_set: 92
# test_user_set: 355
# test_ws_user_set: 119
# test_cs_user_set: 236
# item_set: 132209
# train_item_set: 116371
# val_item_set: 13112
# test_item_set: 19790
# --------------------------------------------------------

from __future__ import print_function
from __future__ import division

import os
import datetime
import pickle
import json
import pandas as pd
import numpy as np


def load_amzcd(raw_path):
    item_info_path = os.path.join(raw_path, 'meta_cds.json')
    rating_path = os.path.join(raw_path, 'ratings_CDs_and_Vinyl.csv')

    item_info_dict = {}
    item_info_keys = ['asin', 'brand', 'category']
    with open(item_info_path, mode='r') as f:
        i = 0
        for item in f.readlines():
            record = json.loads(item)
            item_info_dict[i] = dict((k, record[k]) for k in item_info_keys)  # only keep required info
            i += 1
            if i % 100000 == 0:
                print('load item_info done row {}'.format(i))
    item_info = pd.DataFrame.from_dict(item_info_dict, orient='index')
    item_info = item_info.rename(columns={'asin': 'item_id'})
    item_info = item_info.dropna()

    with open(rating_path, mode='r') as f:
        ratings = pd.read_csv(f,
                              names=['user_id', 'item_id', 'rating', 'timestamp'],
                              sep=",", engine='python')
    ratings['time'] = ratings["timestamp"].map(lambda x: datetime.datetime.fromtimestamp(x))  # e.g., 2001-01-01 06:12:40
    ratings = ratings.drop(["timestamp"], axis=1)
    ratings = ratings[ratings['item_id'].isin(item_info.item_id.unique())]

    return item_info, ratings


def preproc_amzcd(train_split_cutoff, val_split_cutoff, min_sample_size):

    data_path = '../data/amazon_cds'

    item_info_dat, ratings_dat = load_amzcd(raw_path=data_path)
    print('item_info_dat', len(item_info_dat))
    print('ratings_dat', len(ratings_dat))

    # split train & val & test time
    ratings_sorted = ratings_dat.sort_values(by='time', ascending=True).reset_index(drop=True)
    start_time = ratings_sorted['time'][0]
    train_split_time = ratings_sorted['time'][round(train_split_cutoff * len(ratings_dat))]
    val_split_time = ratings_sorted['time'][round(val_split_cutoff * len(ratings_dat))]
    end_time = ratings_sorted['time'][len(ratings_dat) - 1]
    print('start time {}, train split time {}, val split time {}, end time {}'.format(start_time,
                                                                                      train_split_time,
                                                                                      val_split_time,
                                                                                      end_time))

    # split train & val & test dat
    train_dat = ratings_sorted[ratings_sorted['time'] <= train_split_time]
    val_dat = ratings_sorted[(ratings_sorted['time'] > train_split_time) & (ratings_sorted['time'] <= val_split_time)]
    test_dat = ratings_sorted[ratings_sorted['time'] > val_split_time]
    print('raw num train users: {}, raw num val users: {}, raw num test users: {}'.format(train_dat.user_id.nunique(),
                                                                                          val_dat.user_id.nunique(),
                                                                                          test_dat.user_id.nunique()))

    # preproc train_df
    train_df = pd.DataFrame()
    train_user_counts = []

    train_user_ids = train_dat.user_id.unique()
    train_gb = train_dat.groupby(['user_id'])
    for u_id in train_user_ids:
        u_ratings = train_gb.get_group(u_id).reset_index(drop=True)  # all samples for the train user. return dataframe
        u_count = len(u_ratings)  # actual sample size for the train user
        if u_count >= min_sample_size:
            train_df = train_df.append(u_ratings, ignore_index=True)
            train_user_counts.append(u_count)

    print('preproc num train users: {}, min count: {}, avg count: {}, max count: {}'.format(len(train_user_counts),
                                                                                            min(train_user_counts),
                                                                                            sum(train_user_counts) / len(train_user_counts),
                                                                                            max(train_user_counts)))

    # preproc val_df
    val_df = pd.DataFrame()
    val_user_counts = []

    val_user_ids = val_dat.user_id.unique()
    val_gb = val_dat.groupby(['user_id'])
    for u_id in val_user_ids:
        u_ratings = val_gb.get_group(u_id).reset_index(drop=True)  # all samples for the val user. return dataframe
        u_count = len(u_ratings)  # actual sample size for the val user
        if u_count >= min_sample_size:
            val_df = val_df.append(u_ratings, ignore_index=True)
            val_user_counts.append(u_count)

    print('preproc num val users: {}, min count: {}, avg count: {}, max count: {}'.format(len(val_user_counts),
                                                                                          min(val_user_counts),
                                                                                          sum(val_user_counts) / len(val_user_counts),
                                                                                          max(val_user_counts)))

    # preproc test_df
    test_df = pd.DataFrame()
    test_user_counts = []

    test_user_ids = test_dat.user_id.unique()
    test_gb = test_dat.groupby(['user_id'])
    for u_id in test_user_ids:
        u_ratings = test_gb.get_group(u_id).reset_index(drop=True)  # all samples for the test user. return dataframe
        u_count = len(u_ratings)  # actual sample size for the test user
        if u_count >= min_sample_size:
            test_df = test_df.append(u_ratings, ignore_index=True)
            test_user_counts.append(u_count)

    print('preproc num test users: {}, min count: {}, avg count: {}, max count: {}'.format(len(test_user_counts),
                                                                                           min(test_user_counts),
                                                                                           sum(test_user_counts) / len(test_user_counts),
                                                                                           max(test_user_counts)))

    # collect appeared users & items
    train_user_set = set(train_df.user_id.unique())
    val_user_set = set(val_df.user_id.unique())
    test_user_set = set(test_df.user_id.unique())
    user_set = train_user_set.union(val_user_set).union(test_user_set)

    train_item_set = set(train_df.item_id.unique())
    val_item_set = set(val_df.item_id.unique())
    test_item_set = set(test_df.item_id.unique())
    item_set = train_item_set.union(val_item_set).union(test_item_set)

    # keep only info of appeared users & items
    user_info_dat = pd.DataFrame({'user_id': list(user_set)})
    item_info_dat = item_info_dat[item_info_dat['item_id'].isin(item_set)]
    assert len(user_info_dat.user_id.unique()) == len(user_set)
    assert len(item_info_dat.item_id.unique()) == len(item_set)

    # create user & item map
    user_id_list = sorted(user_set)
    user_map = dict(zip(user_id_list, range(len(user_id_list))))  # create map, key is original id, value is mapped id starting from 0
    user_info_dat['user_id'] = user_info_dat['user_id'].map(lambda x: user_map[x])  # map key to value in df
    train_df['user_id'] = train_df['user_id'].map(lambda x: user_map[x])  # map key to value in df
    val_df['user_id'] = val_df['user_id'].map(lambda x: user_map[x])  # map key to value in df
    test_df['user_id'] = test_df['user_id'].map(lambda x: user_map[x])  # map key to value in df

    item_id_list = sorted(item_set)
    item_map = dict(zip(item_id_list, range(len(item_id_list))))  # create map, key is original id, value is mapped id starting from 0
    item_info_dat['item_id'] = item_info_dat['item_id'].map(lambda x: item_map[x])  # map key to value in df
    train_df['item_id'] = train_df['item_id'].map(lambda x: item_map[x])  # map key to value in df
    val_df['item_id'] = val_df['item_id'].map(lambda x: item_map[x])  # map key to value in df
    test_df['item_id'] = test_df['item_id'].map(lambda x: item_map[x])  # map key to value in df

    brand_list = sorted(item_info_dat.brand.unique())
    brand_map = dict(zip(brand_list, range(len(brand_list))))  # create map, key is original id, value is mapped id starting from 0
    item_info_dat['brand'] = item_info_dat['brand'].map(lambda x: brand_map[x])  # map key to value in df

    cate_list = sorted(set([i for sublist in item_info_dat.category.tolist() for i in sublist]))
    cate_map = dict(zip(cate_list, range(len(cate_list))))  # create map, key is original id, value is mapped id starting from 0
    item_info_dat['category'] = item_info_dat['category'].map(lambda x: [cate_map[i] for i in x])  # map key to value in df

    print('user_id_list', len(user_id_list))
    print('item_id_list', len(item_id_list))
    print('brand_list', len(brand_list))
    print('cate_list', len(cate_list))

    user_dict = {}
    for row in user_info_dat.itertuples():
        # one-class, use class index
        user_idx = np.array([row.user_id], dtype=np.int32)
        user_dict[row.user_id] = user_idx  # [1]

    item_dict = {}
    for row in item_info_dat.itertuples():
        # one-class, use class index
        item_idx = np.array([row.item_id], dtype=np.int32)
        brand_idx = np.array([row.brand], dtype=np.int32)
        # possibly multi-class, use multi-hot
        cate_idx = np.zeros(shape=len(cate_list), dtype=np.int32)  # [490]
        for i in row.category:
            cate_idx[i] = 1
        item_dict[row.item_id] = np.concatenate((item_idx, brand_idx, cate_idx))  # [492]

    with open(os.path.join(data_path, 'user_dict.pkl'), mode='wb') as fp:
        pickle.dump(user_dict, fp, protocol=2)  # edit protocol based on the python version for model training
    print('user_dict: {}'.format(len(user_dict.keys())))

    with open(os.path.join(data_path, 'item_dict.pkl'), mode='wb') as fp:
        pickle.dump(item_dict, fp, protocol=2)  # edit protocol based on the python version for model training
    print('item_dict: {}'.format(len(item_dict.keys())))

    train_df.to_csv('{}/train_df.csv'.format(data_path), index=False)
    print('train_df: {}'.format(len(train_df)))

    val_df.to_csv('{}/val_df.csv'.format(data_path), index=False)
    print('val_df: {}'.format(len(val_df)))

    test_df.to_csv('{}/test_df.csv'.format(data_path), index=False)
    print('test_df: {}'.format(len(test_df)))

    # collect user & item set dict
    train_user_set = set(train_df.user_id.unique())
    val_user_set = set(val_df.user_id.unique())
    test_user_set = set(test_df.user_id.unique())
    user_set = train_user_set.union(val_user_set).union(test_user_set)
    val_ws_user_set = val_user_set.intersection(train_user_set)
    val_cs_user_set = val_user_set - val_ws_user_set
    test_ws_user_set = test_user_set.intersection(train_user_set)
    test_cs_user_set = test_user_set - test_ws_user_set
    user_set_dict = {'user_set': user_set,
                     'train_user_set': train_user_set,
                     'val_user_set': val_user_set,
                     'val_ws_user_set': val_ws_user_set,
                     'val_cs_user_set': val_cs_user_set,
                     'test_user_set': test_user_set,
                     'test_ws_user_set': test_ws_user_set,
                     'test_cs_user_set': test_cs_user_set}
    for k, v in user_set_dict.items():
        print('{}: {}'.format(k, len(v)))

    train_item_set = set(train_df.item_id.unique())
    val_item_set = set(val_df.item_id.unique())
    test_item_set = set(test_df.item_id.unique())
    item_set = train_item_set.union(val_item_set).union(test_item_set)
    item_set_dict = {'item_set': item_set,
                     'train_item_set': train_item_set,
                     'val_item_set': val_item_set,
                     'test_item_set': test_item_set}
    for k, v in item_set_dict.items():
        print('{}: {}'.format(k, len(v)))

    with open(os.path.join(data_path, 'user_set_dict.pkl'), mode='wb') as fp:
        pickle.dump(user_set_dict, fp, protocol=2)  # edit protocol based on the python version for model training
    with open(os.path.join(data_path, 'item_set_dict.pkl'), mode='wb') as fp:
        pickle.dump(item_set_dict, fp, protocol=2)  # edit protocol based on the python version for model training


if __name__ == "__main__":
    preproc_amzcd(train_split_cutoff=0.7, val_split_cutoff=0.8, min_sample_size=30)
