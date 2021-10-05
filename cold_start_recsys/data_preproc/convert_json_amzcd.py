import gzip
import json
import os

data_path = '../data/amazon_cds'

item_info_keys = ['asin', 'brand', 'category']
with gzip.open(os.path.join(data_path, 'meta_CDs_and_Vinyl.json.gz'), 'r') as source_f:
    i = 0
    for item in source_f:
        record = json.loads(item)
        dict_ = {}
        for k in item_info_keys:
            dict_[k] = record[k]
        with open(os.path.join(data_path, 'meta_cds.json'), 'a') as target_f:
            json.dump(dict_, target_f)
            target_f.write('\n')
        i += 1
        if i % 100000 == 0:
            print('preproc item_info done row {}'.format(i))
