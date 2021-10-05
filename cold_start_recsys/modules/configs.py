"""
Dataset-specific config
"""

config_ml = {
    # item
    'num_item': 3697,
    'num_rate': 6,
    'num_genre': 25,
    'num_director': 2186,
    'num_actor': 8030,

    # user
    'num_user': 5255,
    'num_gender': 2,
    'num_age': 7,
    'num_occupation': 21,
    'num_zipcode': 3402,
}

config_yelp = {
    # item
    'num_item': 159611,
    'num_city': 1023,
    'num_cate': 1317,

    # user
    'num_user': 25206,
}

config_amzcd = {
    # item
    'num_item': 132209,
    'num_brand': 49283,
    'num_cate': 490,

    # user
    'num_user': 3876,
}
