from __future__ import print_function
from __future__ import division

import random
import pickle
import numpy as np

from utils import get_images


class DataGenerator(object):
    def __init__(self, config, num_samples_per_class, meta_batch_size):
        self.config = config
        self.num_samples_per_class = num_samples_per_class
        self.batch_size = meta_batch_size
        self.num_class = config['num_class']

        if config['data'] == 'meta_dataset':
            self.img_size = config.get('img_size', (84, 84))
            self.dim_input = np.prod(self.img_size) * 3
            self.dim_output = self.num_class
            self.sub_datasets = ['CUB_Bird', 'FGVC_Aircraft', 'FGVCx_Fungi', 'VGG_Flower', 'DTD_Texture', 'GTSRB_Tsign']
            metatrain_dict_ls, metaeval_dict_ls = [], []
            for each_dataset in self.sub_datasets:
                with open('{0}/meta-dataset/{1}/train_dict.pkl'.format(config['datadir'], each_dataset), mode='rb') as fp:
                    metatrain_dict = pickle.load(fp)
                metatrain_dict_ls.append(metatrain_dict)
                if config['eval_test']:
                    with open('{0}/meta-dataset/{1}/test_dict.pkl'.format(config['datadir'], each_dataset), mode='rb') as fp:
                        metaeval_dict = pickle.load(fp)
                else:
                    with open('{0}/meta-dataset/{1}/val_dict.pkl'.format(config['datadir'], each_dataset), mode='rb') as fp:
                        metaeval_dict = pickle.load(fp)
                metaeval_dict_ls.append(metaeval_dict)
            self.metatrain_dict_ls = metatrain_dict_ls
            self.metaeval_dict_ls = metaeval_dict_ls

        elif config['data'] == 'miniimagenet':
            self.img_size = config.get('img_size', (84, 84))
            self.dim_input = np.prod(self.img_size) * 3
            self.dim_output = self.num_class
            with open('{}/miniimagenet/train_dict.pkl'.format(config['datadir']), mode='rb') as fp:
                metatrain_dict = pickle.load(fp)
            if config['eval_test']:
                with open('{}/miniimagenet/test_dict.pkl'.format(config['datadir']), mode='rb') as fp:
                    metaeval_dict = pickle.load(fp)
            else:
                with open('{}/miniimagenet/val_dict.pkl'.format(config['datadir']), mode='rb') as fp:
                    metaeval_dict = pickle.load(fp)
            print('Done loading data dict')
            self.metatrain_dict = metatrain_dict  # list of folders, each folder represents one class
            self.metaeval_dict = metaeval_dict

        else:
            raise ValueError('Unrecognized dataset')

    def make_input_meta_dataset(self, train=True):
        if train:
            dict_ls = self.metatrain_dict_ls
        else:
            dict_ls = self.metaeval_dict_ls
        all_images = []  # batch_size x no. of samples per task
        for _ in range(self.batch_size):
            if self.config['train'] or self.config['test_dataset'] == -1:  # test specific dataset
                sel = np.random.randint(6)
            else:
                sel = self.config['test_dataset']
            sampled_classes = random.sample(dict_ls[sel].keys(), self.num_class)  # construct a 5-way task
            labels_and_images = get_images(sampled_classes, dict_ls[sel], range(self.num_class),
                                           nb_samples=self.num_samples_per_class, shuffle=False)  # [(label in {0, 1, 2, 3, 4}, numpy_image)]
            # make sure the above isn't randomized order, shuffle=False
            # follow the order [(0, image_1), (0, image_2), ..., (1, image_6), (1, image_7), ..., (4, image_21), (4, image_22), ...]
            labels = [li[0] for li in labels_and_images]
            images = [li[1] for li in labels_and_images]
            all_images.extend(images)
        examples_per_batch = self.num_class * self.num_samples_per_class  # no. of samples per task
        all_image_batches, all_label_batches = [], []
        for i in range(self.batch_size):
            image_batch = all_images[i * examples_per_batch:(i + 1) * examples_per_batch]  # [no. of samples per task, 21168]
            label_batch = labels  # [no. of samples per task]
            new_list, new_label_list = [], []  # shuffled image_batch & label_batch
            for k in range(self.num_samples_per_class):
                class_idxs = list(range(5))
                random.shuffle(class_idxs)
                true_idxs = [class_idx * self.num_samples_per_class + k for class_idx in class_idxs]
                new_list.extend([image_batch[i] for i in true_idxs])  # np of [21168] x num_class
                new_label_list.extend([label_batch[i] for i in true_idxs])  # scalar x num_class
            new_list = np.stack(new_list, 0)  # np of [no. of samples per task, 21168]
            new_label_list = np.stack(new_label_list, 0)  # np of [no. of samples per task]
            all_image_batches.append(new_list)
            all_label_batches.append(new_label_list)
        all_image_batches = np.stack(all_image_batches, 0).astype(np.float32) / 255.0  # np of [4, no. of samples per task, 21168]
        all_label_batches = np.stack(all_label_batches, 0).astype(np.int32)  # np of [4, no. of samples per task]
        return all_image_batches, all_label_batches

    def make_input_miniimagenet(self, train=True):
        if train:
            dict_ = self.metatrain_dict
        else:
            dict_ = self.metaeval_dict
        all_images = []  # 4 x no. of samples per task
        for _ in range(self.batch_size):
            sampled_classes = random.sample(dict_.keys(), self.num_class)  # construct a 5-way task
            labels_and_images = get_images(sampled_classes, dict_, range(self.num_class),
                                           nb_samples=self.num_samples_per_class, shuffle=False)  # [(label in {0, 1, 2, 3, 4}, image_np)]
            # make sure the above isn't randomized order, shuffle=False
            # follow the order [(0, image_1), (0, image_2), ..., (1, image_6), (1, image_7), ..., (4, image_21), (4, image_22), ...]
            labels = [li[0] for li in labels_and_images]
            images = [li[1] for li in labels_and_images]
            all_images.extend(images)
        examples_per_batch = self.num_class * self.num_samples_per_class  # no. of samples per task
        all_image_batches, all_label_batches = [], []
        for i in range(self.batch_size):
            image_batch = all_images[i * examples_per_batch:(i + 1) * examples_per_batch]  # [no. of samples per task, 21168]
            label_batch = labels  # [no. of samples per task]
            new_list, new_label_list = [], []  # shuffled image_batch & label_batch
            for k in range(self.num_samples_per_class):
                class_idxs = list(range(5))
                random.shuffle(class_idxs)
                true_idxs = [class_idx * self.num_samples_per_class + k for class_idx in class_idxs]
                new_list.extend([image_batch[i] for i in true_idxs])  # np of [21168] x num_class
                new_label_list.extend([label_batch[i] for i in true_idxs])  # scalar x num_class
            new_list = np.stack(new_list, 0)  # np of [no. of samples per task, 21168]
            new_label_list = np.stack(new_label_list, 0)  # np of [no. of samples per task]
            all_image_batches.append(new_list)
            all_label_batches.append(new_label_list)
        all_image_batches = np.stack(all_image_batches, 0).astype(np.float32) / 255.0  # np of [4, no. of samples per task, 21168]
        all_label_batches = np.stack(all_label_batches, 0).astype(np.int32)  # np of [4, no. of samples per task]
        return all_image_batches, all_label_batches
