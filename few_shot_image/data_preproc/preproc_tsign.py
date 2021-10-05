"""
Preprocess GTSRB Traffic Sign
"""

import numpy as np
import os
import subprocess
import glob
from PIL import Image


source_dir = '../data/GTSRB'
image_folder = os.path.join(source_dir, 'Final_Training/Images')

target_dir = '../data/meta-dataset/GTSRB_Tsign'

num_train_class = 26
num_val_class = 7
num_test_class = 10
train_val_test = [num_train_class, num_val_class, num_test_class]

labels = []
all_images = []
for label in os.listdir(image_folder):
    if os.path.isdir(os.path.join(image_folder, label)):
        labels.append(label)
        all_images.extend(glob.glob(os.path.join(image_folder, label, '*.ppm')))

classes = labels
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
        target_dir_k_c = os.path.join(target_dir_k, c)
        if not os.path.exists(target_dir_k_c):
            os.makedirs(target_dir_k_c)

# Resize images
for i, image_file in enumerate(all_images):
    im = Image.open(image_file)
    im = im.resize((84, 84), resample=Image.LANCZOS)
    im.save(image_file)
    if (i + 1) % 500 == 0:
        print('done resize {} images'.format(i + 1))

# mv images
for i, image_file in enumerate(all_images):
    image_class = image_file.split('/')[-1].split('_')[-2]
    image_k = list(class_split.keys())[np.argmax([image_class in class_split[k] for k in list(class_split.keys())])]  # check image_class in which classes.keys()
    image_target_path = os.path.join(target_dir, image_k, image_class)
    cmd = ['mv', image_file, image_target_path]
    subprocess.call(cmd)
    print('{}/{} {}'.format(i, len(all_images), ' '.join(cmd)))
