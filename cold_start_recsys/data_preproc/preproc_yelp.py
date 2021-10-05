"""
Preprocess Yelp
"""

# ========================================================
# setting: train_split_cutoff=0.7, val_split_cutoff=0.8, min_count=30
# --------------------------------------------------------
# user_info_dat 1968703
# item_info_dat 208869
# ratings_dat 8018920
# start time 2004-10-12 10:13:32, train split time 2018-02-07 21:22:18, val split time 2018-09-04 14:59:12, end time 2019-12-13 15:51:19
# raw num train users: 1393449, raw num val users: 376527, raw num test users: 671026
# preproc num train users: 22822, min count: 30, avg count: 73.4722636053, max count: 3939
# preproc num val users: 1281, min count: 30, avg count: 48.5035128806, max count: 689
# preproc num test users: 3442, min count: 30, avg count: 53.3332364904, max count: 519
# user_id_list 25206
# item_id_list 159611
# city_list 1023
# cate_list 1317
# user_dict: 25206
# item_dict: 159611
# train_df: 1676784
# val_df: 62133
# test_df: 183573
# test_ws_user_set: 1441
# val_ws_user_set: 681
# train_user_set: 22822
# test_cs_user_set: 2001
# test_user_set: 3442
# val_cs_user_set: 600
# val_user_set: 1281
# user_set: 25206
# item_set: 159611
# test_item_set: 53517
# val_item_set: 29380
# train_item_set: 146816
# --------------------------------------------------------

from __future__ import print_function
from __future__ import division

import os
import pickle
import json
import pandas as pd
import numpy as np


def load_yelp(raw_path):
    user_info_path = os.path.join(raw_path, 'yelp_user.json')
    item_info_path = os.path.join(raw_path, 'yelp_business.json')
    rating_path = os.path.join(raw_path, 'yelp_review.json')

    user_info_dict = {}
    user_info_keys = ['user_id']
    with open(user_info_path, mode='r') as f:
        i = 0
        for item in f.readlines():
            record = json.loads(item)
            user_info_dict[i] = dict((k, record[k]) for k in user_info_keys)  # only keep required info
            i += 1
            if i % 100000 == 0:
                print('load user_info done row {}'.format(i))
    user_info = pd.DataFrame.from_dict(user_info_dict, orient='index')
    user_info = user_info.dropna()

    item_info_dict = {}
    item_info_keys = ['business_id', 'city', 'categories']
    with open(item_info_path, mode='r') as f:
        i = 0
        for item in f.readlines():
            record = json.loads(item)
            item_info_dict[i] = dict((k, record[k]) for k in item_info_keys)  # only keep required info
            i += 1
            if i % 100000 == 0:
                print('load item_info done row {}'.format(i))
    item_info = pd.DataFrame.from_dict(item_info_dict, orient='index')
    item_info = item_info.rename(columns={'business_id': 'item_id'})
    item_info = item_info.dropna()

    ratings_dict = {}
    ratings_keys = ['user_id', 'business_id', 'stars', 'date']
    with open(rating_path, mode='r') as f:
        i = 0
        for item in f.readlines():
            record = json.loads(item)
            ratings_dict[i] = dict((k, record[k]) for k in ratings_keys)  # only keep required info
            i += 1
            if i % 100000 == 0:
                print('load ratings done row {}'.format(i))
    ratings = pd.DataFrame.from_dict(ratings_dict, orient='index')
    ratings = ratings.rename(columns={'business_id': 'item_id', 'stars': 'rating', 'date': 'time'})
    ratings = ratings.dropna()
    ratings = ratings[(ratings['user_id'].isin(user_info.user_id.unique())) &
                      (ratings['item_id'].isin(item_info.item_id.unique()))]

    return user_info, item_info, ratings


def preproc_yelp(train_split_cutoff, val_split_cutoff, min_sample_size):

    data_path = '../data/yelp_dataset'

    user_info_dat, item_info_dat, ratings_dat = load_yelp(raw_path=data_path)
    print('user_info_dat', len(user_info_dat))
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
    user_info_dat = user_info_dat[user_info_dat['user_id'].isin(user_set)]
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

    city_list = sorted(item_info_dat.city.unique())
    city_map = dict(zip(city_list, range(len(city_list))))  # create map, key is original id, value is mapped id starting from 0
    item_info_dat['city'] = item_info_dat['city'].map(lambda x: city_map[x])  # map key to value in df

    cate_list = sorted(set([i for sub in item_info_dat.categories.tolist() for i in sub.split(', ')]))
    cate_map = dict(zip(cate_list, range(len(cate_list))))  # create map, key is original id, value is mapped id starting from 0
    item_info_dat['categories'] = item_info_dat['categories'].map(lambda x: [cate_map[i] for i in x.split(', ')])  # map key to value in df

    print('user_id_list', len(user_id_list))
    print('item_id_list', len(item_id_list))
    print('city_list', len(city_list))
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
        city_idx = np.array([row.city], dtype=np.int32)
        # possibly multi-class, use multi-hot
        cate_idx = np.zeros(shape=len(cate_list), dtype=np.int32)  # []
        for i in row.categories:
            cate_idx[i] = 1
        item_dict[row.item_id] = np.concatenate((item_idx, city_idx, cate_idx))  # [1338]

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
    preproc_yelp(train_split_cutoff=0.7, val_split_cutoff=0.8, min_sample_size=30)
