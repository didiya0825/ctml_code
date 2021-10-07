from __future__ import print_function
from __future__ import division

import tensorflow as tf


# Loss functions
def mse(pred_y, ground_truth):
    pred_y = tf.reshape(pred_y, [-1])
    ground_truth = tf.reshape(ground_truth, [-1])
    return tf.reduce_mean(tf.square(pred_y - ground_truth))


# Metrics
def mae(pred_y, ground_truth):
    pred_y = tf.reshape(pred_y, [-1])
    ground_truth = tf.reshape(ground_truth, [-1])
    return tf.reduce_mean(tf.abs(pred_y - ground_truth))


def IDCG(ground_truth, topk):
    ranked_list, _ = tf.nn.top_k(ground_truth, k=tf.shape(ground_truth)[0], sorted=True)
    idcg = 0
    for i in range(topk):
        # numerator = tf.pow(2.0, ranked_list[i]) - 1
        numerator = ranked_list[i]
        denominator = tf.log(tf.constant(i + 2, dtype=tf.float32)) / tf.log(tf.constant(2, dtype=tf.float32))
        idcg += numerator / denominator
    return idcg


def nDCG(pred_y, ground_truth, topk):
    pred_y = tf.reshape(pred_y, [-1])
    ground_truth = tf.reshape(ground_truth, [-1])
    _, ranked_idx = tf.nn.top_k(pred_y, k=tf.shape(pred_y)[0], sorted=True)
    dcg = 0
    idcg = IDCG(ground_truth, topk)
    for i in range(topk):
        idx = ranked_idx[i]
        # numerator = tf.pow(2.0, ground_truth[idx]) - 1
        numerator = ground_truth[idx]
        denominator = tf.log(tf.constant(i + 2, dtype=tf.float32)) / tf.log(tf.constant(2, dtype=tf.float32))
        dcg += numerator / denominator
    return dcg / idcg


def DCG(scores, topk):
    dcg = 0
    for i in range(topk):
        numerator = scores[i]
        denominator = tf.log(tf.constant(i + 2, dtype=tf.float32)) / tf.log(tf.constant(2, dtype=tf.float32))
        dcg += numerator / denominator
    return dcg


def nDCG2(pred_y, ground_truth, topk):
    pred_y = tf.reshape(pred_y, [-1])
    ground_truth = tf.reshape(ground_truth, [-1])

    _, ranked_idx = tf.nn.top_k(ground_truth, k=tf.shape(ground_truth)[0], sorted=True)
    top_k_ranked_idx = ranked_idx[:topk]
    # top_k_ranked_idx, _ = tf.nn.top_k(top_k_ranked_idx, k=tf.shape(top_k_ranked_idx)[0], sorted=False)

    r_s_at_k = tf.gather(ground_truth, top_k_ranked_idx)
    p_s_at_k = tf.gather(pred_y, top_k_ranked_idx)

    r_s_at_k, _ = tf.nn.top_k(r_s_at_k, k=tf.shape(r_s_at_k)[0], sorted=True)
    idcg = DCG(r_s_at_k, topk)

    dcg = DCG(p_s_at_k, topk)

    return dcg / idcg


def cluster_aux_target(soft_assignments):
    """
    :param soft_assignments: of shape [meta_batch_size, num_cluster]
    :return:
    """
    cluster_frequencies = tf.reduce_sum(soft_assignments, axis=0, keepdims=True)  # [1, num_cluster]
    # cluster_frequencies = tf.tile(cluster_frequencies, [tf.shape(soft_assignments)[0], 1])  # [meta_batch_size, num_cluster]

    enhanced_soft_assignments = tf.square(soft_assignments) / cluster_frequencies  # [meta_batch_size, num_cluster]

    task_frequencies = tf.reduce_sum(enhanced_soft_assignments, axis=1, keepdims=True)  # [meta_batch_size, 1]
    # task_frequencies = tf.tile(task_frequencies, [1, tf.shape(soft_assignments)[1]])  # [meta_batch_size, num_cluster]

    normalized_soft_assignments = enhanced_soft_assignments / task_frequencies  # [meta_batch_size, num_cluster]

    return normalized_soft_assignments


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
