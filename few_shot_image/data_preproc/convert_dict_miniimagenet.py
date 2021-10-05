import os
import numpy as np
from PIL import Image
import pickle

target_dir = '../data/miniimagenet'

for split in ['train', 'val', 'test']:
    train_dict = {}
    for label in os.listdir(os.path.join(target_dir, split)):
        if os.path.isdir(os.path.join(target_dir, split, label)):
            label_folder = os.path.join(target_dir, split, label)
            train_dict[label] = []
            for image_name in os.listdir(label_folder):
                image_file = os.path.join(label_folder, image_name)
                pil_image = Image.open(image_file).convert("RGB")
                image = np.asarray(pil_image)
                image = np.reshape(image, (84 * 84 * 3))
                image = image.astype(np.uint8)
                train_dict[label].append(image)
    pickle.dump(train_dict, open(os.path.join(target_dir, '{}_dict.pkl'.format(split)), 'wb'), protocol=2)  # edit protocol based on the python version for model training
