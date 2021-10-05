from __future__ import print_function
from __future__ import division

import tensorflow as tf


class Kmeans(object):
    def __init__(self, config):
        self.feat_emb_centroids = tf.get_variable(name='feat_emb_centroids', shape=(config['feat_num_cluster'], config['embed_dim']))
        self.path_emb_centroids = tf.get_variable(name='path_emb_centroids', shape=(config['path_num_cluster'], config['embed_dim']))

    def feat_model(self, inputs):
        """
        :param inputs: [1, feat_embed_dim]
        :return:
        """
        soft_assignment = tf.nn.softmax(tf.matmul(inputs, self.feat_emb_centroids, transpose_b=True))  # [1, feat_num_cluster]
        outputs = tf.matmul(soft_assignment, self.feat_emb_centroids)  # [1, feat_embed_dim]
        return soft_assignment, outputs

    def path_model(self, inputs):
        """
        :param inputs: [1, path_embed_dim]
        :return:
        """
        soft_assignment = tf.nn.softmax(tf.matmul(inputs, self.path_emb_centroids, transpose_b=True))  # [1, path_num_cluster]
        outputs = tf.matmul(soft_assignment, self.path_emb_centroids)  # [1, path_embed_dim]
        return soft_assignment, outputs


def gru_parallel(prev_h, cur_x, param_dim, input_dim, hidden_dim, output_dim):
    """
    perform one step of gru with tensor parallelization
    :param prev_h: [N, param_dim, 1, hidden_dim]
    :param cur_x: [N, param_dim, 1, input_dim]
    :param param_dim: dimension of (flattened) parameter
    :param input_dim: dimension of input x
    :param hidden_dim: dimension of gru hidden state
    :param output_dim: dimension of output y
    :return: cur_h, cur_y
    """
    N = tf.shape(prev_h)[0]

    # gate params
    u_z = tf.tile(tf.expand_dims(tf.get_variable(name='u_z', shape=[param_dim, input_dim, hidden_dim]), 0), [N, 1, 1, 1])
    w_z = tf.tile(tf.expand_dims(tf.get_variable(name='w_z', shape=[param_dim, hidden_dim, hidden_dim]), 0), [N, 1, 1, 1])
    z = tf.sigmoid(tf.matmul(cur_x, u_z) + tf.matmul(prev_h, w_z))  # [N, param_dim, 1, hidden_dim]

    u_r = tf.tile(tf.expand_dims(tf.get_variable(name='u_r', shape=[param_dim, input_dim, hidden_dim]), 0), [N, 1, 1, 1])
    w_r = tf.tile(tf.expand_dims(tf.get_variable(name='w_r', shape=[param_dim, hidden_dim, hidden_dim]), 0), [N, 1, 1, 1])
    r = tf.sigmoid(tf.matmul(cur_x, u_r) + tf.matmul(prev_h, w_r))  # [N, param_dim, 1, hidden_dim]

    u_g = tf.tile(tf.expand_dims(tf.get_variable(name='u_g', shape=[param_dim, input_dim, hidden_dim]), 0), [N, 1, 1, 1])
    w_g = tf.tile(tf.expand_dims(tf.get_variable(name='w_g', shape=[param_dim, hidden_dim, hidden_dim]), 0), [N, 1, 1, 1])
    cur_h_tilde = tf.tanh(tf.matmul(cur_x, u_g) + tf.matmul(prev_h * r, w_g))  # [N, param_dim, 1, hidden_dim]

    cur_h = (1 - z) * prev_h + z * cur_h_tilde  # [N, param_dim, 1, hidden_dim]

    # params to generate cur_y
    w_hy = tf.tile(tf.expand_dims(tf.get_variable(name='w_hy', shape=[param_dim, hidden_dim, output_dim]), 0), [N, 1, 1, 1])
    b_y = tf.tile(tf.expand_dims(tf.get_variable(name='b_y', shape=[param_dim, 1, output_dim]), 0), [N, 1, 1, 1])

    # cur_y = tf.tanh(tf.matmul(cur_h, w_hy) + b_y)  # [N, param_dim, 1, output_dim]
    cur_y = tf.matmul(cur_h, w_hy) + b_y  # [N, param_dim, 1, output_dim]

    return cur_h, cur_y


class PathLearner(object):
    def __init__(self, config):

        self.config = config

        # path_embed settings
        self.num_step = config['num_step']
        self.stop_grad = config['path_stop_grad']
        self.add_param = config['add_param']
        self.add_loss = config['add_loss']
        self.add_grad = config['add_grad']
        self.add_fisher = config['add_fisher']
        self.path_learner = config['path_learner']
        if self.path_learner == 'gru':
            self.gru_hidden_dim = config['gru_hidden_dim']
            self.gru_output_dim = config['gru_output_dim']
        self.path_embed_dim = config['embed_dim']
        self.path_emb = None

    def model(self, inputa, labela, forward, weights, loss_func):
        """
        :param inputa: support x
        :param labela: support y
        :param forward: forward network def
        :param weights: forward network weights dict
        :param loss_func: loss function
        :return: path_emb [1, path_embed_dim]
        """
        with tf.variable_scope('path_embed', reuse=tf.AUTO_REUSE):

            fast_weights = weights

            loss_ls, params_ls, gradients_ls, gradients_sq_ls = [], [], [], []

            # perform num_step rehearsed task learning
            for j in range(self.num_step):
                output = forward(inputa, fast_weights)
                loss = loss_func(output, labela)  # compute gradient on theta_0
                grads = tf.gradients(loss, list(fast_weights.values()))  # gradients of fast_weights
                gradients = dict(zip(fast_weights.keys(), grads))
                if self.stop_grad:  # whether to compute gradient on theta_0 (second-order gradient)
                    for k, v in gradients.items():
                        gradients[k] = tf.stop_gradient(v)
                else:
                    for k, v in gradients.items():
                        if 'emb_w' in k:
                            gradients[k] = tf.stop_gradient(v)

                if self.add_param:
                    fcn_params = {}  # updated params of fast_weights fcn
                    for k, v in fast_weights.items():
                        if 'fcn' in k:
                            fcn_params[k] = v
                    params_ls.append(fcn_params)

                if self.add_grad:
                    fcn_gradients = {}  # gradients of fast_weights fcn
                    for k, v in gradients.items():
                        if 'fcn' in k:
                            fcn_gradients[k] = v
                    gradients_ls.append(fcn_gradients)

                if self.add_fisher:
                    fcn_gradients_sq = {}  # gradients square of fast_weights fcn
                    for k, v in gradients.items():
                        if 'fcn' in k:
                            fcn_gradients_sq[k] = tf.square(v)
                    gradients_sq_ls.append(fcn_gradients_sq)

                if self.add_loss:
                    loss_ls.append(tf.reduce_mean(loss))

                fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.config['adapt_lr'] * gradients[key] for key in fast_weights.keys()]))

            # collect params (fcn only) + loss at each adaptation step
            stacked_vec_ls = []
            num_elem = None
            for j in range(self.num_step):

                if self.add_param:
                    param_ls = []
                    for k, v in params_ls[j].items():
                        param_ls.append(tf.reshape(v, [-1]))
                    param_vec = tf.concat(param_ls, -1)  # [param_size]

                if self.add_grad:
                    grad_ls = []
                    for k, v in gradients_ls[j].items():
                        grad_ls.append(tf.reshape(v, [-1]))
                    grad_vec = tf.concat(grad_ls, -1)  # [param_size]

                if self.add_fisher:
                    grad_sq_ls = []
                    for k, v in gradients_sq_ls[j].items():
                        grad_sq_ls.append(tf.reshape(v, [-1]))
                    grad_sq_vev = tf.concat(grad_sq_ls, -1)  # [param_size]

                if self.add_param and self.add_grad and self.add_fisher:
                    if self.add_loss:
                        loss_vec = tf.tile(tf.reshape(loss_ls[j], [-1]), tf.shape(grad_vec))  # [param_size]
                        stacked_vec = tf.stack([param_vec, grad_vec, grad_sq_vev, loss_vec], axis=0)  # [4, param_size]
                        num_elem = 4
                    else:
                        stacked_vec = tf.stack([param_vec, grad_vec, grad_sq_vev], axis=0)  # [3, param_size]
                        num_elem = 3

                elif self.add_grad and self.add_fisher:
                    if self.add_loss:
                        loss_vec = tf.tile(tf.reshape(loss_ls[j], [-1]), tf.shape(grad_vec))  # [param_size]
                        stacked_vec = tf.stack([grad_vec, grad_sq_vev, loss_vec], axis=0)  # [3, param_size]
                        num_elem = 3
                    else:
                        stacked_vec = tf.stack([grad_vec, grad_sq_vev], axis=0)  # [2, param_size]
                        num_elem = 2

                elif self.add_grad and self.add_param:
                    if self.add_loss:
                        loss_vec = tf.tile(tf.reshape(loss_ls[j], [-1]), tf.shape(grad_vec))  # [param_size]
                        stacked_vec = tf.stack([param_vec, grad_vec, loss_vec], axis=0)  # [3, param_size]
                        num_elem = 3
                    else:
                        stacked_vec = tf.stack([param_vec, grad_vec], axis=0)  # [2, param_size]
                        num_elem = 2

                elif self.add_fisher and self.add_param:
                    if self.add_loss:
                        loss_vec = tf.tile(tf.reshape(loss_ls[j], [-1]), tf.shape(grad_sq_vev))  # [param_size]
                        stacked_vec = tf.stack([param_vec, grad_sq_vev, loss_vec], axis=0)  # [3, param_size]
                        num_elem = 3
                    else:
                        stacked_vec = tf.stack([param_vec, grad_sq_vev], axis=0)  # [2, param_size]
                        num_elem = 2

                elif self.add_param:
                    if self.add_loss:
                        loss_vec = tf.tile(tf.reshape(loss_ls[j], [-1]), tf.shape(param_vec))  # [param_size]
                        stacked_vec = tf.stack([param_vec, loss_vec], axis=0)  # [2, param_size]
                        num_elem = 2
                    else:
                        stacked_vec = tf.stack([param_vec], axis=0)  # [1, param_size]
                        num_elem = 1

                elif self.add_grad:
                    if self.add_loss:
                        loss_vec = tf.tile(tf.reshape(loss_ls[j], [-1]), tf.shape(grad_vec))  # [param_size]
                        stacked_vec = tf.stack([grad_vec, loss_vec], axis=0)  # [2, param_size]
                        num_elem = 2
                    else:
                        stacked_vec = tf.stack([grad_vec], axis=0)  # [1, param_size]
                        num_elem = 1

                elif self.add_fisher:
                    if self.add_loss:
                        loss_vec = tf.tile(tf.reshape(loss_ls[j], [-1]), tf.shape(grad_sq_vev))  # [param_size]
                        stacked_vec = tf.stack([grad_sq_vev, loss_vec], axis=0)  # [2, param_size]
                        num_elem = 2
                    else:
                        stacked_vec = tf.stack([grad_sq_vev], axis=0)  # [1, param_size]
                        num_elem = 1

                stacked_vec_ls.append(stacked_vec)  # list of tensor with shapes [[num_elem, param_size]_1, ..., [num_elem, param_size]_num_step]

            # model inputs at different adaptation steps
            with tf.variable_scope(self.path_learner):

                param_size = stacked_vec_ls[0].get_shape().as_list()[-1]  # param_size

                if self.path_learner == 'gru':
                    prev_h = tf.zeros([1, param_size, 1, self.gru_hidden_dim], dtype=tf.float32)  # [1, param_size, 1, gru_hidden_dim]
                    for j in range(self.num_step):
                        cur_x = stacked_vec_ls[j]  # [num_elem, param_size]
                        cur_x = tf.expand_dims(tf.expand_dims(tf.transpose(cur_x), 0), -2)  # [1, param_size, 1, num_elem]
                        cur_h, cur_y = gru_parallel(prev_h=prev_h,
                                                    cur_x=cur_x,
                                                    param_dim=param_size,
                                                    input_dim=num_elem,
                                                    hidden_dim=self.gru_hidden_dim,
                                                    output_dim=self.gru_output_dim)  # [1, param_size, 1, gru_hidden_dim], [1, param_size, 1, gru_output_dim]
                        prev_h = cur_h
                    flat_vec = tf.reshape(cur_y, [1, param_size * self.gru_output_dim])  # [1, param_size x gru_output_dim]
                    self.path_emb = tf.layers.dense(flat_vec, units=self.path_embed_dim, activation=tf.nn.relu, name='last_fc')  # [1, path_embed_dim]

                else:  # self.path_learner == 'linear'
                    stacked_vecs = tf.stack(stacked_vec_ls, axis=0)  # [num_step, num_elem, param_size]
                    stacked_vecs = tf.reshape(stacked_vecs, [self.num_step * num_elem, param_size])  # [num_step x num_elem, param_size]
                    w = tf.get_variable(name='w', shape=[self.num_step * num_elem, param_size])  # [num_step x num_elem, param_size]
                    b = tf.get_variable(name='b', shape=[1, param_size])  # [1, param_size]
                    output = tf.reduce_sum(stacked_vecs * w, axis=0, keepdims=True) + b  # [1, param_size]
                    self.path_emb = tf.layers.dense(output, units=self.path_embed_dim, activation=tf.nn.relu, name='last_fc')  # [1, path_embed_dim]

        return self.path_emb, stacked_vec_ls
