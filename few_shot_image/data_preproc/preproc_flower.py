"""
Preprocess VGG Flower
"""

import numpy as np
import os
import subprocess
import scipy.io
import glob
from PIL import Image


source_dir = '../data'
label_file = os.path.join(source_dir, 'imagelabels.mat')
image_folder = os.path.join(source_dir, 'jpg')

target_dir = '../data/meta-dataset/VGG_Flower'

mat_dict = scipy.io.loadmat(label_file)
labels = mat_dict['labels'][0]

num_train_class = 64
num_val_class = 16
num_test_class = 22
train_val_test = [num_train_class, num_val_class, num_test_class]

classes = np.unique(labels)
rs = np.random.RandomState(123)
rs.shuffle(classes)

class_split = {
    'train': classes[:num_train_class],
    'val': classes[num_train_class:num_train_class+num_val_class],
    'test': classes[num_train_class+num_val_class:]
}

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
for k in class_split.keys():
    target_dir_k = os.path.join(target_dir, k)
    if not os.path.exists(target_dir_k):
        os.makedirs(target_dir_k)
    for c in class_split[k]:
        target_dir_k_c = os.path.join(target_dir_k, str(c))
        if not os.path.exists(target_dir_k_c):
            os.makedirs(target_dir_k_c)

all_images = glob.glob(image_folder + '/*')

# Resize images
for i, image_file in enumerate(all_images):
    im = Image.open(image_file)
    im = im.resize((84, 84), resample=Image.LANCZOS)
    im.save(image_file)
    if (i + 1) % 500 == 0:
        print('done resize {} images'.format(i + 1))

# mv images
for image_file in all_images:
    idx = int(image_file.split('image_')[-1].split('.jpg')[0]) - 1
    image_class = labels[idx]
    image_k = list(class_split.keys())[np.argmax([image_class in class_split[k] for k in list(class_split.keys())])]  # check image_class in which classes.keys()
    image_target_path = os.path.join(target_dir, image_k, str(image_class))
    cmd = ['mv', image_file, image_target_path]
    subprocess.call(cmd)
    print('{}/{} {}'.format(idx, len(all_images), ' '.join(cmd)))
