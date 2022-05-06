# CTML
This is our experimental code for ICLR 2022 submission "Clustered Task-Aware Meta-Learning by Learning from Learning Paths".  



## Requirements

python 2.*  
pillow  
numpy  
pandas  
scipy  
tensorflow 1.10+



## Few-Shot Image Classification

Navigate to `few_shot_image` directory for few-shot image classification experiments:
> cd few_shot_image

### Data Preprocessing
#### Meta-Dataset
1. Download the pre-processed datasets of CUB-200-2011, FGVC-Aircraft, FGVCx-Fungi, and Describable Textures [here](https://drive.google.com/file/d/1IJk93N48X0rSL69nQ1Wr-49o8u0e75HM/view) released by the authors of [HSML](https://github.com/huaxiuyao/HSML). Extract under `data` directory, which will create a `meta-dataset` folder containing the four sub-datasets.
2. Download [images](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz) and [labels](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat) from VGG Flower [official website](https://www.robots.ox.ac.uk/~vgg/data/flowers/). Move under `data` directory.
3. Download [GTSRB_Final_Training_Images.zip](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip) from GTSRB [official website](https://benchmark.ini.rub.de/). Extract under `data` directory.
4. Preprocess VGG Flower and GTSRB datasets (resize image + split data) by running:
> cd data_preproc  
> python preproc_flower.py  
> python preproc_tsign.py
5. Convert image file to numpy array and store in dict (for faster I/O at training time) by running:
> python convert_dict_meta_dataset.py

#### Mini-Imagenet
1. Download Mini-Imagenet [here](https://drive.google.com/file/d/1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk/view). Extract under `data/miniimagenet` directory.
2. Preprocess Mini-Imagenet dataset (resize image + split data) by running:
> cd data_preproc  
> python preproc_miniimagenet.py
3. Convert image file to numpy array and store in dict (for faster I/O at training time) by running:
> python convert_dict_miniimagenet.py

#### Data Directory Structure
After pre-processing, `few_shot_image/data` directory should have the following structure.
```
few_shot_image/data
├── meta-dataset
│   ├── CUB_Bird
│   │   ├── train_dict.pkl
│   │   ├── val_dict.pkl
│   │   └── test_dict.pkl
│   ├── DTD_Texture
│   │   ├── train_dict.pkl
│   │   ├── val_dict.pkl
│   │   └── test_dict.pkl
│   ├── FGVC_Aircraft
│   │   ├── train_dict.pkl
│   │   ├── val_dict.pkl
│   │   └── test_dict.pkl
│   ├── FGVCx_Fungi
│   │   ├── train_dict.pkl
│   │   ├── val_dict.pkl
│   │   └── test_dict.pkl
│   ├── GTSRB_Tsign
│   │   ├── train_dict.pkl
│   │   ├── val_dict.pkl
│   │   └── test_dict.pkl
│   └── VGG_Flower
│   │   ├── train_dict.pkl
│   │   ├── val_dict.pkl
│   │   └── test_dict.pkl
└── miniimagenet
    ├── train_dict.pkl
    ├── val_dict.pkl
    └── test_dict.pkl
```
### Meta-Training
To train the proposed model and variants , run the following (here we take 5-way 5-shot on Meta-Dataset for example):

| Method                 | Command                                                      |
| ---------------------- | ------------------------------------------------------------ |
| CTML                   | ```python main.py --data meta_dataset --support_size 5 ```   |
| CTML-feat              | ```python main.py --data meta_dataset --support_size 5 --path_or_feat feat``` |
| CTML-path              | ```python main.py --data meta_dataset --support_size 5 --path_or_feat path``` |
| Linear Path Learner    | ```python main.py --data meta_dataset --support_size 5 --path_learner linear``` |
| FC Path Learner        | ```python main.py --data meta_dataset --support_size 5 --path_learner fc``` |
| Attention Path Learner | ```python main.py --data meta_dataset --support_size 5 --path_learner attention``` |

### Meta-Testing

To evaluate the model, run the following:

| Method        | Commend                                                      |
| ------------- | ------------------------------------------------------------ |
| CTML          | ```python main.py --eval --data meta_dataset --support_size 5 --test_iters 59000 58000 57000``` |
| CTML(approx.) | ```python main.py --eval --data meta_dataset --support_size 5 --use_shortcut_approx true --test_iters 59000 58000 57000``` |



## Cold-Start Recommendation

Navigate to `cold_start_recsys` directory for cold-start recommendation experiments:
> cd cold_start_recsys  

### Data Preprocessing
#### MovieLens-1M
1. Download [ml-1m.zip](https://files.grouplens.org/datasets/movielens/ml-1m.zip) from MovieLens datasets [official website](https://grouplens.org/datasets/movielens/). Extract under `data/movielens_1m` directory.
2. Preprocess MovieLens-1M dataset (construct samples with side information + split data) by running:
> cd data_preproc  
> python preproc_ml.py

#### Yelp
1. Download Yelp dataset from the [official website](https://www.yelp.com/dataset). Extract under `data` directory, which will create a `yelp_dataset` folder.
2. Extract only useful information from the raw JSON files (for faster preprocessing later) by running:
> cd data_preproc  
> python convert_json_yelp.py
2. Preprocess Yelp dataset (construct samples with side information + split data) by running:
> python preproc_yelp.py

#### Amazon-CDs
1. Download [ratings](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_CDs_and_Vinyl.csv) and [metadata](https://jmcauley.ucsd.edu/data/amazon/amazon_readme.txt) of Amazon CDs and Vinyl category from the [official website](https://jmcauley.ucsd.edu/data/amazon/). Move under `data/amazon_cds` directory.
2. Extract only useful information from the item metadata JSON file (for faster preprocessing later) by running:
> cd data_preproc  
> python convert_json_amzcd.py
2. Preprocess Amazon-CDs dataset (construct samples with side information + split data) by running:
> python preproc_amzcd.py

#### Data Directory Structure
After pre-processing, `cold_start_recsys/data` directory should have the following structure.
```
cold_start_recsys/data
├── movielens_1m
│   ├── train_df.csv
│   ├── val_df.csv
│   ├── test_df.csv
│   ├── user_dict.pkl  // user info in one-hot/multi-hot
│   ├── item_dict.pkl  // item info in one-hot/multi-hot
│   └── user_set_dict.pkl  // record warm start and cold start user set
├── yelp_dataset
│   ├── train_df.csv
│   ├── val_df.csv
│   ├── test_df.csv
│   ├── user_dict.pkl
│   ├── item_dict.pkl
│   └── user_set_dict.pkl
└── amazon_cds
    ├── train_df.csv
    ├── val_df.csv
    ├── test_df.csv
    ├── user_dict.pkl
    ├── item_dict.pkl
    └── user_set_dict.pkl
```

### Meta-Training & Meta-Testing
To train the proposed model and variants , run the following (here we take MovieLens-1M for example):

| Method | Command                                 |
| :----: | ---------------------------------------- |
|  CTML  | ```python main.py --data movielens_1m``` |
| CTML-feat | ```python main.py --data movielens_1m --path_or_feat feat``` |
| CTML-path | ```python main.py --data movielens_1m --path_or_feat path``` |
| Linear Path Learner | ```python main.py --data movielens_1m --path_learner linear``` |
| FC Path Learner | ```python main.py --data movielens_1m --path_learner fc``` |
| Attention Path Learner | ```python main.py --data movielens_1m --path_learner attention``` |

To test the effect of changing the number of clusters, run the following:

| Method                         | Command                                                      |
| ------------------------------ | ------------------------------------------------------------ |
| Set number of path clusters    | ```python main.py --data movielens_1m --path_num_cluster 16``` |
| Set number of feature clusters | ```python main.py --data movielens_1m --feat_num_cluster 8``` |

Meta-testing will be automatically performed after every epoch of meta-training. 

Note that if shortcut tunnel is applied, meta-testing with and without the shortcut approximation will both be conducted (i.e., the results of CTML and CTML(approx.) will be displayed together).



## Acknowledgement

This code is modified based on [ARML](https://github.com/huaxiuyao/ARML) (for few-shot image classification) and [MeLU](https://github.com/hoyeoplee/MeLU) (for cold-start recommendation). We thank the authors for their contributions.

