from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell


class Kmeans(object):
    def __init__(self, config):
        with tf.variable_scope('kmeans', reuse=tf.AUTO_REUSE):
            self.path_emb_centroids = tf.get_variable(name='path_emb_centroids', shape=(config['path_num_cluster'], config['embed_dim']))

    def model(self, inputs):
        """
        :param inputs: [1, embed_dim]
        :return:
        """
        soft_assignment = tf.nn.softmax(tf.matmul(inputs, self.path_emb_centroids, transpose_b=True))  # [1, path_num_cluster]
        outputs = tf.matmul(soft_assignment, self.path_emb_centroids)  # [1, embed_dim]
        return soft_assignment, outputs


class LSTMAutoencoder(object):
    def __init__(self, config, hidden_num, input_num, cell=None, reverse=True, decode_without_input=False, name=None):
        self.config = config
        self.name = name
        if cell is None:
            self._enc_cell = GRUCell(hidden_num, name='encoder_cell_{}'.format(self.name))
            self._dec_cell = GRUCell(hidden_num, name='decoder_cell_{}'.format(self.name))
        else:
            self._enc_cell = cell
            self._dec_cell = cell
        self.reverse = reverse
        self.decode_without_input = decode_without_input
        self.hidden_num = hidden_num
        self.elem_num = input_num

        self.dec_weight = tf.Variable(tf.truncated_normal([self.hidden_num, self.elem_num], dtype=tf.float32), name='dec_weight_{}'.format(self.name))
        self.dec_bias = tf.Variable(tf.constant(0.1, shape=[self.elem_num], dtype=tf.float32), name='dec_bias_{}'.format(self.name))

    def model(self, inputs):

        inputs = tf.expand_dims(inputs, 0)

        inputs = tf.unstack(inputs, axis=1)

        self.batch_num = self.config['meta_batch_size']

        with tf.variable_scope('encoder_{}'.format(self.name)):
            (self.z_codes, self.enc_state) = tf.contrib.rnn.static_rnn(self._enc_cell, inputs, dtype=tf.float32)

        with tf.variable_scope('decoder_{}'.format(self.name)) as vs:

            if self.decode_without_input:
                dec_inputs = [tf.zeros(tf.shape(inputs[0]), dtype=tf.float32) for _ in range(len(inputs))]
                (dec_outputs, dec_state) = tf.contrib.rnn.static_rnn(self._dec_cell, dec_inputs,
                                                                     initial_state=self.enc_state,
                                                                     dtype=tf.float32)
                if self.reverse:
                    dec_outputs = dec_outputs[::-1]
                dec_output_ = tf.transpose(tf.stack(dec_outputs), [1, 0, 2])
                dec_weight_ = tf.tile(tf.expand_dims(self.dec_weight, 0), [self.batch_num, 1, 1])
                self.output_ = tf.matmul(dec_weight_, dec_output_) + self.dec_bias
            else:
                dec_state = self.enc_state
                dec_input_ = tf.zeros(tf.shape(inputs[0]),
                                      dtype=tf.float32)

                dec_outputs = []
                for step in range(len(inputs)):
                    if step > 0:
                        vs.reuse_variables()
                    (dec_input_, dec_state) = self._dec_cell(dec_input_, dec_state)
                    dec_input_ = tf.matmul(dec_input_, self.dec_weight) + self.dec_bias
                    dec_outputs.append(dec_input_)
                if self.reverse:
                    dec_outputs = dec_outputs[::-1]
                self.output_ = tf.transpose(tf.stack(dec_outputs), [1, 0, 2])

        self.input_ = tf.transpose(tf.stack(inputs), [1, 0, 2])
        self.loss = tf.reduce_mean(tf.square(self.input_ - self.output_))
        self.emb_all = tf.reduce_mean(self.z_codes, axis=0)

        return self.emb_all, self.loss


def gru_parallel(prev_h, cur_x, param_dim, input_dim, hidden_dim, output_dim):
    """
    perform one step of gru with tensor parallelization (faster at graph construction)
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
                loss = loss_func(output, labela, self.config['support_size'])  # compute gradient on theta_0
                grads = tf.gradients(loss, list(fast_weights.values()))  # gradients of fast_weights
                gradients = dict(zip(fast_weights.keys(), grads))
                if self.stop_grad:  # whether to compute gradient on theta_0 (second-order gradient)
                    for k, v in gradients.items():
                        gradients[k] = tf.stop_gradient(v)

                if self.add_param:
                    params_ls.append(fast_weights)

                if self.add_grad:
                    gradients_ls.append(gradients)

                if self.add_fisher:
                    gradients_sq = {}  # gradients square of fast_weights
                    for k, v in gradients.items():
                        gradients_sq[k] = tf.square(v)
                    gradients_sq_ls.append(gradients_sq)

                if self.add_loss:
                    loss_ls.append(tf.reduce_mean(loss))

                fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.config['adapt_lr'] * gradients[key] for key in fast_weights.keys()]))

            # collect params + loss at each adaptation step
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
