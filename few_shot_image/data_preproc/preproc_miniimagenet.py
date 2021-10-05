"""
Preprocess Mini-Imagenet. We acquire the train, val, and test csv files from Ravi & Larochelle '17.
"""

import csv
import glob
import os

from PIL import Image

source_dir = '../data/miniimagenet'
image_folder = os.path.join(source_dir, 'images')

target_dir = '../data/miniimagenet'

all_images = glob.glob(image_folder + '/*')

# Resize images
for i, image_file in enumerate(all_images):
    im = Image.open(image_file)
    im = im.resize((84, 84), resample=Image.LANCZOS)  # resampling filter that maps multiple input pixels to a single output pixel
    im.save(image_file)
    if (i + 1) % 500 == 0:
        print('done resize {} images'.format(i + 1))

# Put in correct directory
for datatype in ['train', 'val', 'test']:
    os.system('mkdir ' + target_dir + '/' + datatype)

    with open(source_dir + '/' + datatype + '.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        last_label = ''
        for i, row in enumerate(reader):
            if i == 0:  # skip the headers
                continue
            label = row[1]
            image_name = row[0]
            if label != last_label:
                cur_dir = target_dir + '/' + datatype + '/' + label + '/'
                os.system('mkdir ' + cur_dir)
                last_label = label
            os.system('mv ' + source_dir + '/images/' + image_name + ' ' + cur_dir)
