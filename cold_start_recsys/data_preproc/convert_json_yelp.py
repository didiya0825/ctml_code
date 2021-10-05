import json
import os

data_path = '../data/yelp_dataset'

user_info_keys = ['user_id']
with open(os.path.join(data_path, 'yelp_academic_dataset_user.json'), encoding='utf8') as source_f:
    i = 0
    for item in source_f:
        record = json.loads(item)
        dict_ = {}
        for k in user_info_keys:
            dict_[k] = record[k]
        with open(os.path.join(data_path, 'yelp_user.json'), 'a') as target_f:
            json.dump(dict_, target_f)
            target_f.write('\n')
        i += 1
        if i % 100000 == 0:
            print('preproc user_info done row {}'.format(i))

item_info_keys = ['business_id', 'city', 'categories']
with open(os.path.join(data_path, 'yelp_academic_dataset_business.json'), encoding='utf8') as source_f:
    i = 0
    for item in source_f:
        record = json.loads(item)
        dict_ = {}
        for k in item_info_keys:
            dict_[k] = record[k]
        with open(os.path.join(data_path, 'yelp_business.json'), 'a') as target_f:
            json.dump(dict_, target_f)
            target_f.write('\n')
        i += 1
        if i % 100000 == 0:
            print('preproc item_info done row {}'.format(i))

ratings_keys = ['user_id', 'business_id', 'stars', 'date']
with open(os.path.join(data_path, 'yelp_academic_dataset_review.json'), encoding='utf8') as source_f:
    i = 0
    for item in source_f:
        record = json.loads(item)
        dict_ = {}
        for k in ratings_keys:
            dict_[k] = record[k]
        with open(os.path.join(data_path, 'yelp_review.json'), 'a') as target_f:
            json.dump(dict_, target_f)
            target_f.write('\n')
        i += 1
        if i % 100000 == 0:
            print('preproc ratings done row {}'.format(i))
