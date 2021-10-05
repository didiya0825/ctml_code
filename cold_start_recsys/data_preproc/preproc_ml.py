"""
Preprocess MovieLens-1M. We acquire the extra side information from MeLU.
"""

# ========================================================
# setting: train_split_cutoff=0.7, val_split_cutoff=0.8, min_count=30
# --------------------------------------------------------
# user_id_list: 6040
# item_id_list: 3704
# start time 2000-04-26 07:05:32, train split time 2000-11-22 11:06:26, val split time 2000-12-02 22:52:18, end time 2003-03-01 01:49:50
# raw num train users: 4870, raw num val users: 945, raw num test users: 1783
# preproc num train users: 4119, min count: 30, avg count: 166.01723719349357,, max count: 1849
# preproc num val users: 621, min count: 30, avg count: 155.20772946859904, max count: 1518
# preproc num test users: 1229, min count: 30, avg count: 156.5720097640358, max count: 1226
# user_set: 5255
# train_user_set: 4119
# val_user_set: 621
# val_ws_user_set: 122
# val_cs_user_set: 499
# test_user_set: 1229
# test_ws_user_set: 485
# test_cs_user_set: 744
# item_set: 3697
# train_item_set: 3626
# val_item_set: 3358
# test_item_set: 3502
# --------------------------------------------------------

from __future__ import print_function
from __future__ import division

import os
import re
import datetime
import pickle
import pandas as pd
import numpy as np


def load_list(fname):
    list_ = []
    with open(fname, mode='r') as f:
        for line in f.readlines():
            list_.append(line.strip())
    return list_


def load_movielens(raw_path):
    user_info_path = os.path.join(raw_path, 'ml-1m/users.dat')
    item_info_path = os.path.join(raw_path, 'extrainfo/movies_extrainfos.dat')
    rating_path = os.path.join(raw_path, 'ml-1m/ratings.dat')

    with open(user_info_path, mode='r') as f:
        user_info = pd.read_csv(f,
                                names=['user_id', 'gender', 'age', 'occupation', 'zipcode'],
                                sep="::", engine='python')

    with open(item_info_path, mode='r') as f:
        item_info = pd.read_csv(f,
                                names=['item_id', 'title', 'year', 'rate', 'released', 'genre', 'director', 'writer', 'actor', 'plot', 'poster'],
                                sep="::", engine='python', encoding="utf-8")
    item_info = item_info[['item_id', 'rate', 'genre', 'director', 'actor']]

    with open(rating_path, mode='r') as f:
        ratings = pd.read_csv(f,
                              names=['user_id', 'item_id', 'rating', 'timestamp'],
                              sep="::", engine='python')
    ratings['time'] = ratings["timestamp"].map(lambda x: datetime.datetime.fromtimestamp(x))  # e.g., 2001-01-01 06:12:40
    ratings = ratings.drop(["timestamp"], axis=1)
    ratings = ratings[(ratings['user_id'].isin(user_info.user_id.unique())) &
                      (ratings['item_id'].isin(item_info.item_id.unique()))]

    user_info = user_info[user_info['user_id'].isin(ratings.user_id.unique())]
    item_info = item_info[item_info['item_id'].isin(ratings.item_id.unique())]

    assert len(user_info.user_id.unique()) == len(ratings.user_id.unique())
    assert len(item_info.item_id.unique()) == len(ratings.item_id.unique())

    return user_info, item_info, ratings


def item_converting(row, item_id_list, rate_list, genre_list, director_list, actor_list):
    # one-class, use class index
    item_idx = np.array([item_id_list.index(row.at[0, 'item_id'])], dtype=np.int16)  # [1]
    rate_idx = np.array([rate_list.index(str(row.at[0, 'rate']))], dtype=np.int16)  # [1]

    # possibly multi-class, use multi-hot
    genre_idx = np.zeros(shape=25, dtype=np.int16)  # [25]
    for genre in str(row.at[0, 'genre']).split(", "):
        idx = genre_list.index(genre)
        genre_idx[idx] = 1

    director_idx = np.zeros(shape=2186, dtype=np.int16)  # [2186]
    for director in str(row.at[0, 'director']).split(", "):
        idx = director_list.index(re.sub(r'\([^()]*\)', '', director))  # what is the regex pattern trying to match ??
        director_idx[idx] = 1

    actor_idx = np.zeros(shape=8030, dtype=np.int16)  # [8030]
    for actor in str(row.at[0, 'actor']).split(", "):
        idx = actor_list.index(actor)
        actor_idx[idx] = 1

    return np.concatenate((item_idx, rate_idx, genre_idx, director_idx, actor_idx))  # [10243]


def user_converting(row, user_id_list, gender_list, age_list, occupation_list, zipcode_list):
    # one-class, use class index
    user_idx = np.array([user_id_list.index(row.at[0, 'user_id'])], dtype=np.int16)
    gender_idx = np.array([gender_list.index(str(row.at[0, 'gender']))], dtype=np.int16)
    age_idx = np.array([age_list.index(str(row.at[0, 'age']))], dtype=np.int16)
    occupation_idx = np.array([occupation_list.index(str(row.at[0, 'occupation']))], dtype=np.int16)
    zipcode_idx = np.array([zipcode_list.index(str(row.at[0, 'zipcode'][:5]))], dtype=np.int16)  # some zipcode tail with 'xxxxx-xxxx'
    return np.concatenate((user_idx, gender_idx, age_idx, occupation_idx, zipcode_idx))  # [5]


def preproc_ml(train_split_cutoff, val_split_cutoff, min_sample_size):

    data_path = '../data/movielens_1m'

    user_info_dat, item_info_dat, ratings_dat = load_movielens(raw_path=data_path)

    gender_list = load_list("{}/extrainfo/m_gender.txt".format(data_path))
    age_list = load_list("{}/extrainfo/m_age.txt".format(data_path))
    occupation_list = load_list("{}/extrainfo/m_occupation.txt".format(data_path))
    zipcode_list = load_list("{}/extrainfo/m_zipcode.txt".format(data_path))
    rate_list = load_list("{}/extrainfo/m_rate.txt".format(data_path))
    genre_list = load_list("{}/extrainfo/m_genre.txt".format(data_path))
    actor_list = load_list("{}/extrainfo/m_actor.txt".format(data_path))
    director_list = load_list("{}/extrainfo/m_director.txt".format(data_path))

    user_id_list = user_info_dat.user_id.unique().tolist()
    item_id_list = item_info_dat.item_id.unique().tolist()
    print('user_id_list: {}'.format(len(user_id_list)))
    print('item_id_list: {}'.format(len(item_id_list)))

    # preproc user_dict
    user_dict = {}  # {u_id: array[user_idx, gender_idx, age_idx, occupation_idx, zipcode_idx], ...}
    for u_id in user_id_list:
        u_row = user_info_dat[user_info_dat.user_id == u_id].reset_index(drop=True)  # info for the user. return dataframe
        feature_vector = user_converting(row=u_row,
                                         user_id_list=user_id_list,
                                         gender_list=gender_list,
                                         age_list=age_list,
                                         occupation_list=occupation_list,
                                         zipcode_list=zipcode_list)  # array of shape [5]
        user_dict[u_id] = feature_vector

    with open(os.path.join(data_path, 'user_dict.pkl'), mode='wb') as fp:
        pickle.dump(user_dict, fp, protocol=2)  # edit protocol based on the python version for model training

    # preproc item_dict
    item_dict = {}  # {i_id: array[item_idx, rate_idx, genre_idx(multi-hot), director_idx(multi-hot), actor_idx(multi-hot)], ...}
    for i_id in item_id_list:
        i_row = item_info_dat[item_info_dat.item_id == i_id].reset_index(drop=True)  # info for the item. return dataframe
        feature_vector = item_converting(row=i_row,
                                         item_id_list=item_id_list,
                                         rate_list=rate_list,
                                         genre_list=genre_list,
                                         director_list=director_list,
                                         actor_list=actor_list)  # array of shape [10243]
        item_dict[i_id] = feature_vector

    with open(os.path.join(data_path, 'item_dict.pkl'), mode='wb') as fp:
        pickle.dump(item_dict, fp, protocol=2)  # edit protocol based on the python version for model training

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
    for u_id in train_user_ids:
        u_ratings = train_dat[train_dat.user_id == u_id].reset_index(drop=True)  # all samples for the train user. return dataframe
        u_count = len(u_ratings)  # actual sample size for the train user
        if u_count >= min_sample_size:
            train_df = train_df.append(u_ratings, ignore_index=True)
            train_user_counts.append(u_count)

    print('preproc num train users: {}, min count: {}, avg count: {}, max count: {}'.format(len(train_user_counts),
                                                                                            min(train_user_counts),
                                                                                            sum(train_user_counts) / len(train_user_counts),
                                                                                            max(train_user_counts)))

    train_df.to_csv('{}/train_df.csv'.format(data_path), index=False)

    # preproc val_df
    val_df = pd.DataFrame()
    val_user_counts = []

    val_user_ids = val_dat.user_id.unique()
    for u_id in val_user_ids:
        u_ratings = val_dat[val_dat.user_id == u_id].reset_index(drop=True)  # all samples for the val user. return dataframe
        u_count = len(u_ratings)  # actual sample size for the val user
        if u_count >= min_sample_size:
            val_df = val_df.append(u_ratings, ignore_index=True)
            val_user_counts.append(u_count)

    print('preproc num val users: {}, min count: {}, avg count: {}, max count: {}'.format(len(val_user_counts),
                                                                                          min(val_user_counts),
                                                                                          sum(val_user_counts) / len(val_user_counts),
                                                                                          max(val_user_counts)))

    val_df.to_csv('{}/val_df.csv'.format(data_path), index=False)

    # preproc test_df
    test_df = pd.DataFrame()
    test_user_counts = []

    test_user_ids = test_dat.user_id.unique()
    for u_id in test_user_ids:
        u_ratings = test_dat[test_dat.user_id == u_id].reset_index(drop=True)  # all samples for the test user. return dataframe
        u_count = len(u_ratings)  # actual sample size for the test user
        if u_count >= min_sample_size:
            test_df = test_df.append(u_ratings, ignore_index=True)
            test_user_counts.append(u_count)

    print('preproc num test users: {}, min count: {}, avg count: {}, max count: {}'.format(len(test_user_counts),
                                                                                           min(test_user_counts),
                                                                                           sum(test_user_counts) / len(test_user_counts),
                                                                                           max(test_user_counts)))

    test_df.to_csv('{}/test_df.csv'.format(data_path), index=False)

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
    preproc_ml(train_split_cutoff=0.7, val_split_cutoff=0.8, min_sample_size=30)
