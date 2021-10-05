from __future__ import print_function
from __future__ import division

import random
import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers


def get_images(keys, dict_, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)  # each class get nb_samples
    else:
        sampler = lambda x: x
    images = [(i, image) \
              for i, key in zip(labels, keys) \
              for image in sampler(dict_[key])]  # (label, numpy_image)
    if shuffle:
        random.shuffle(images)
    return images


def normalize(inp, activation, reuse, scope):
    return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)


def conv_block(inp, cweight, bweight, reuse, scope, activation=tf.nn.relu, max_pool_pad='VALID'):
    stride, no_stride = [1, 2, 2, 1], [1, 1, 1, 1]
    conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
    normed = normalize(conv_output, activation, reuse, scope)
    normed = tf.nn.max_pool(normed, stride, stride, max_pool_pad)

    return normed


def xent(pred, label, size):
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / size
