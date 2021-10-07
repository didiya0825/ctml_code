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


def student_t_similarity(inputs, centroids, alpha=1.):
    """
    :param inputs: [1, embed_dim]
    :param centroids: [num_cluster, embed_dim]
    :param alpha: degrees of freedom, scalar
    :return: soft assignment [1, num_cluster]
    """
    distance = tf.reduce_sum(tf.square(inputs - centroids), axis=-1)  # [num_cluster]
    distance = tf.pow(1 + distance / alpha, - (alpha + 1.) / 2.)  # [num_cluster]
    soft_assignment = tf.expand_dims(distance / tf.reduce_sum(distance), axis=0)  # [1, num_cluster]
    return soft_assignment


def kl_divergence(p, q):
    p = tf.reshape(p, [-1])
    q = tf.reshape(q, [-1])
    return tf.reduce_sum(p * tf.log(p / q) / tf.log(2.))


def js_divergence(p, q):
    p = tf.reshape(p, [-1])
    q = tf.reshape(q, [-1])
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred - label))

