from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from utils import student_t_similarity


class Kmeans(object):
    def __init__(self, config):
        self.config = config
        with tf.variable_scope('kmeans', reuse=tf.AUTO_REUSE):
            self.path_emb_centroids = tf.get_variable(name='path_emb_centroids', shape=(config['path_num_cluster'], config['embed_dim']))
            self.feat_emb_centroids = tf.get_variable(name='feat_emb_centroids', shape=(config['feat_num_cluster'], config['embed_dim']))

    def path_model(self, inputs):
        """
        :param inputs: [1, path_embed_dim]
        :return:
        """
        if self.config['similarity_kernel'] == 'student_t':
            soft_assignment = student_t_similarity(inputs, self.path_emb_centroids)  # [1, path_num_cluster]
        else:  # self.config['similarity_kernel'] == 'dot_product'
            soft_assignment = tf.nn.softmax(tf.matmul(inputs, self.path_emb_centroids, transpose_b=True))  # [1, path_num_cluster]
        outputs = tf.matmul(soft_assignment, self.path_emb_centroids)  # [1, path_embed_dim]
        return soft_assignment, outputs

    def feat_model(self, inputs):
        """
        :param inputs: [1, feat_embed_dim]
        :return:
        """
        if self.config['similarity_kernel'] == 'student_t':
            soft_assignment = student_t_similarity(inputs, self.feat_emb_centroids)  # [1, feat_num_cluster]
        else:  # self.config['similarity_kernel'] == 'dot_product'
            soft_assignment = tf.nn.softmax(tf.matmul(inputs, self.feat_emb_centroids, transpose_b=True))  # [1, feat_num_cluster]
        outputs = tf.matmul(soft_assignment, self.feat_emb_centroids)  # [1, feat_embed_dim]
        return soft_assignment, outputs

    def path_model_shortcut(self, soft_assignment):
        """
        generate cluster-enhanced embedding based on some (actual or reconstructed) path soft assignment and path centroids
        :param soft_assignment: [1, path_num_cluster]
        :return:
        """
        outputs = tf.matmul(soft_assignment, self.path_emb_centroids)  # [1, path_embed_dim]
        return outputs


def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=1,
                        dropout_rate=0.5,
                        is_training=True,
                        name='multihead_att',
                        causality=False,
                        reuse=None,
                        with_qk=False,
                        with_res=False):
    """Applies multihead attention.

    Args:
    name: Optional scope for `variable_scope`.
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size. Default to be the size of queries C_q
      # dropout_rate: A floating point number.
      # is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      with_qk: whether to return projected Q and K together with outputs.
      with_res: whether to use residual link, can be True only if num_units == C_q

    Returns
      A 3d tensor with shape of (N, T_q, C), where C is num_units
    """
    with tf.variable_scope(name, reuse=reuse):
        # Set the fall back option for num_units, the hidden size of query
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        # Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
        # K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
        # V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)
        att_weights = outputs

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # (h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # The final linear layer and dropout.
        # outputs = tf.layers.dense(outputs, num_units)
        # outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)

        # Residual connection
        if with_res:
            outputs += queries  # only for num_units == C_q

        # Normalize
        # outputs = normalize(outputs)  # (N, T_q, C)

    if with_qk:
        return outputs, Q, K
    else:
        return att_weights, outputs


def _positional_encoding_sinusoid(inp):
    """
    PE(pos, 2i) = sin(pos / 10000^{2i/embed_size})
    PE(pos, 2i+1) = cos(pos / 10000^{2i/embed_size})
    """
    batch_size, seq_len, embed_size = inp.shape.as_list()

    with tf.variable_scope('positional_sinusoid'):
        # Copy [0, 1, ..., seq_len - 1] by `batch_size` times => matrix [batch_size, seq_len]
        pos_ind = tf.tile(tf.expand_dims(tf.range(seq_len), 0), [batch_size, 1])

        # Compute the arguments for sin and cos: pos / 10000^{2i/embed_size})
        # Each dimension is sin/cos wave, as a function of the position.
        pos_enc = np.array([
            [pos / np.power(10000., 2. * (i // 2) / embed_size) for i in range(embed_size)]
            for pos in range(seq_len)
        ])  # [seq_len, embed_size]

        # Apply the cosine to even columns and sin to odds.
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])  # dim 2i
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(pos_enc, dtype=tf.float32)  # [seq_len, embed_size]

        out = tf.nn.embedding_lookup(lookup_table, pos_ind)  # [batch, seq_len, embed_size]
        return out


def _positional_encoding_embedding(inp):
    batch_size, seq_len, embed_size = inp.shape.as_list()

    with tf.variable_scope('positional_embedding'):
        # Copy [0, 1, ..., `inp_size`] by `batch_size` times => matrix [batch_size, seq_len]
        pos_ind = tf.tile(tf.expand_dims(tf.range(seq_len), 0), [batch_size, 1])
        embed_lookup = tf.get_variable("embed_lookup", [seq_len, embed_size], tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        out = tf.nn.embedding_lookup(embed_lookup, pos_ind)  # [batch_size, seq_len, embed_size]
        return out


def positional_encoding(inp, pos_encoding_type):
    """
    generate positional encoding
    :param inp: [B, seq_len, embed_size]
    :param pos_encoding_type: 'sinusoid' or 'embedding'
    :return: pos_enc [B, seq_len, embed_size]
    """
    if pos_encoding_type == 'sinusoid':
        pos_enc = _positional_encoding_sinusoid(inp)
    else:
        pos_enc = _positional_encoding_embedding(inp)
    return pos_enc


def gru_parallel(prev_h, cur_x, param_dim, input_dim, hidden_dim, output_dim, share):
    """
    perform one step of gru with tensor parallelization
    :param prev_h: [N, param_dim, 1, hidden_dim]
    :param cur_x: [N, param_dim, 1, input_dim]
    :param param_dim: dimension of (flattened) parameter
    :param input_dim: dimension of input x
    :param hidden_dim: dimension of gru hidden state
    :param output_dim: dimension of output y
    :param share: whether to share across the second dimension
    :return: cur_h, cur_y
    """
    N = tf.shape(prev_h)[0]

    # gate params
    if share:
        u_z = tf.tile(tf.expand_dims(tf.get_variable(name='u_z', shape=[1, input_dim, hidden_dim]), 0), [N, param_dim, 1, 1])
        w_z = tf.tile(tf.expand_dims(tf.get_variable(name='w_z', shape=[1, hidden_dim, hidden_dim]), 0), [N, param_dim, 1, 1])
    else:
        u_z = tf.tile(tf.expand_dims(tf.get_variable(name='u_z', shape=[param_dim, input_dim, hidden_dim]), 0), [N, 1, 1, 1])
        w_z = tf.tile(tf.expand_dims(tf.get_variable(name='w_z', shape=[param_dim, hidden_dim, hidden_dim]), 0), [N, 1, 1, 1])
    z = tf.sigmoid(tf.matmul(cur_x, u_z) + tf.matmul(prev_h, w_z))  # [N, param_dim, 1, hidden_dim]

    if share:
        u_r = tf.tile(tf.expand_dims(tf.get_variable(name='u_r', shape=[1, input_dim, hidden_dim]), 0), [N, param_dim, 1, 1])
        w_r = tf.tile(tf.expand_dims(tf.get_variable(name='w_r', shape=[1, hidden_dim, hidden_dim]), 0), [N, param_dim, 1, 1])
    else:
        u_r = tf.tile(tf.expand_dims(tf.get_variable(name='u_r', shape=[param_dim, input_dim, hidden_dim]), 0), [N, 1, 1, 1])
        w_r = tf.tile(tf.expand_dims(tf.get_variable(name='w_r', shape=[param_dim, hidden_dim, hidden_dim]), 0), [N, 1, 1, 1])
    r = tf.sigmoid(tf.matmul(cur_x, u_r) + tf.matmul(prev_h, w_r))  # [N, param_dim, 1, hidden_dim]

    if share:
        u_g = tf.tile(tf.expand_dims(tf.get_variable(name='u_g', shape=[1, input_dim, hidden_dim]), 0), [N, param_dim, 1, 1])
        w_g = tf.tile(tf.expand_dims(tf.get_variable(name='w_g', shape=[1, hidden_dim, hidden_dim]), 0), [N, param_dim, 1, 1])
    else:
        u_g = tf.tile(tf.expand_dims(tf.get_variable(name='u_g', shape=[param_dim, input_dim, hidden_dim]), 0), [N, 1, 1, 1])
        w_g = tf.tile(tf.expand_dims(tf.get_variable(name='w_g', shape=[param_dim, hidden_dim, hidden_dim]), 0), [N, 1, 1, 1])
    cur_h_tilde = tf.tanh(tf.matmul(cur_x, u_g) + tf.matmul(prev_h * r, w_g))  # [N, param_dim, 1, hidden_dim]

    cur_h = (1 - z) * prev_h + z * cur_h_tilde  # [N, param_dim, 1, hidden_dim]

    # params to generate cur_y
    if share:
        w_hy = tf.tile(tf.expand_dims(tf.get_variable(name='w_hy', shape=[1, hidden_dim, output_dim]), 0), [N, param_dim, 1, 1])
        b_y = tf.tile(tf.expand_dims(tf.get_variable(name='b_y', shape=[1, 1, output_dim]), 0), [N, param_dim, 1, 1])
    else:
        w_hy = tf.tile(tf.expand_dims(tf.get_variable(name='w_hy', shape=[param_dim, hidden_dim, output_dim]), 0), [N, 1, 1, 1])
        b_y = tf.tile(tf.expand_dims(tf.get_variable(name='b_y', shape=[param_dim, 1, output_dim]), 0), [N, 1, 1, 1])

    # cur_y = tf.tanh(tf.matmul(cur_h, w_hy) + b_y)  # [N, param_dim, 1, output_dim]
    cur_y = tf.matmul(cur_h, w_hy) + b_y  # [N, param_dim, 1, output_dim]

    return cur_h, cur_y


class PathLearner(object):
    def __init__(self, config):

        self.config = config

        # path_embed settings
        self.num_step = config['rehearse_num_step']
        self.stop_grad = config['path_stop_grad']
        self.add_param = config['add_param']
        self.add_loss = config['add_loss']
        self.add_grad = config['add_grad']
        self.add_fisher = config['add_fisher']
        self.path_learner = config['path_learner']
        if self.path_learner == 'gru':
            self.gru_hidden_dim = config['gru_hidden_dim']
            self.gru_output_dim = config['gru_output_dim']
        elif self.path_learner == 'fc':
            self.fc_num_layer = config['fc_num_layer']
        elif self.path_learner == 'attention':
            self.att_num_layer = config['att_num_layer']
            self.att_hidden_dim = config['att_hidden_dim']
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

        # collect params (fcn only) + loss at each adaptation step
        stacked_vec_ls = []
        num_elem = None
        for j in range(self.num_step + 1):

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
                grad_sq_vec = tf.concat(grad_sq_ls, -1)  # [param_size]

            if self.add_param and self.add_grad and self.add_fisher:
                if self.add_loss:
                    loss_vec = tf.tile(tf.reshape(loss_ls[j], [-1]), tf.shape(grad_vec))  # [param_size]
                    stacked_vec = tf.stack([param_vec, grad_vec, grad_sq_vec, loss_vec], axis=0)  # [4, param_size]
                    num_elem = 4
                else:
                    stacked_vec = tf.stack([param_vec, grad_vec, grad_sq_vec], axis=0)  # [3, param_size]
                    num_elem = 3

            elif self.add_grad and self.add_fisher:
                if self.add_loss:
                    loss_vec = tf.tile(tf.reshape(loss_ls[j], [-1]), tf.shape(grad_vec))  # [param_size]
                    stacked_vec = tf.stack([grad_vec, grad_sq_vec, loss_vec], axis=0)  # [3, param_size]
                    num_elem = 3
                else:
                    stacked_vec = tf.stack([grad_vec, grad_sq_vec], axis=0)  # [2, param_size]
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
                    loss_vec = tf.tile(tf.reshape(loss_ls[j], [-1]), tf.shape(grad_sq_vec))  # [param_size]
                    stacked_vec = tf.stack([param_vec, grad_sq_vec, loss_vec], axis=0)  # [3, param_size]
                    num_elem = 3
                else:
                    stacked_vec = tf.stack([param_vec, grad_sq_vec], axis=0)  # [2, param_size]
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
                    loss_vec = tf.tile(tf.reshape(loss_ls[j], [-1]), tf.shape(grad_sq_vec))  # [param_size]
                    stacked_vec = tf.stack([grad_sq_vec, loss_vec], axis=0)  # [2, param_size]
                    num_elem = 2
                else:
                    stacked_vec = tf.stack([grad_sq_vec], axis=0)  # [1, param_size]
                    num_elem = 1

            stacked_vec_ls.append(stacked_vec)  # list of tensor with shapes [[num_elem, param_size]_1, ..., [num_elem, param_size]_(num_step + 1)]

        # model inputs at different adaptation steps
        with tf.variable_scope('{}_path_learner'.format(self.path_learner), reuse=tf.AUTO_REUSE):

            param_size = stacked_vec_ls[0].get_shape().as_list()[-1]  # param_size

            if self.path_learner == 'gru':
                prev_h = tf.zeros([1, param_size, 1, self.gru_hidden_dim], dtype=tf.float32)  # [1, param_size, 1, gru_hidden_dim]
                for j in range(self.num_step + 1):
                    cur_x = stacked_vec_ls[j]  # [num_elem, param_size]
                    cur_x = tf.expand_dims(tf.expand_dims(tf.transpose(cur_x), 0), -2)  # [1, param_size, 1, num_elem]
                    cur_h, cur_y = gru_parallel(prev_h=prev_h,
                                                cur_x=cur_x,
                                                param_dim=param_size,
                                                input_dim=num_elem,
                                                hidden_dim=self.gru_hidden_dim,
                                                output_dim=self.gru_output_dim,
                                                share=True)  # [1, param_size, 1, gru_hidden_dim], [1, param_size, 1, gru_output_dim]
                    prev_h = cur_h
                flat_vec = tf.reshape(cur_y, [1, param_size * self.gru_output_dim])  # [1, param_size x gru_output_dim]
                self.path_emb = tf.layers.dense(flat_vec, units=self.path_embed_dim, activation=tf.nn.relu, name='last_fc')  # [1, path_embed_dim]

            elif self.path_learner == 'linear':
                stacked_vecs = tf.stack(stacked_vec_ls, axis=0)  # [num_step + 1, num_elem, param_size]
                stacked_vecs = tf.transpose(stacked_vecs, [2, 0, 1])  # [param_size, num_step + 1, num_elem]
                stacked_vecs = tf.reshape(stacked_vecs, [param_size, -1])  # [param_size, (num_step + 1) x num_elem]
                w = tf.get_variable(name='w', shape=[(self.num_step + 1) * num_elem, 1])  # [(num_step + 1) x num_elem, 1]
                b = tf.get_variable(name='b', shape=[1, 1])  # [1, 1]
                output = tf.matmul(stacked_vecs, w) + b  # [param_size, 1]
                output = tf.transpose(output)  # [1, param_size]
                self.path_emb = tf.layers.dense(output, units=self.path_embed_dim, activation=tf.nn.relu, name='last_fc')  # [1, path_embed_dim]

            elif self.path_learner == 'fc':
                stacked_vecs = tf.stack(stacked_vec_ls, axis=0)  # [num_step + 1, num_elem, param_size]
                stacked_vecs = tf.transpose(stacked_vecs, [2, 0, 1])  # [param_size, num_step + 1, num_elem]
                stacked_vecs = tf.reshape(stacked_vecs, [param_size, -1])  # [param_size, (num_step + 1) x num_elem]
                hidden = stacked_vecs
                for layer in range(1, self.config['fc_num_layer']):
                    hidden = tf.layers.dense(hidden, units=stacked_vecs.get_shape().as_list()[-1], activation=tf.nn.relu, name='fc{}'.format(layer))  # [param_size, (num_step + 1) x num_elem]
                output = tf.layers.dense(hidden, units=1, activation=tf.nn.relu, name='fc{}'.format(self.config['fc_num_layer']))  # [param_size, 1]
                output = tf.transpose(output)  # [1, param_size]
                self.path_emb = tf.layers.dense(output, units=self.path_embed_dim, activation=tf.nn.relu, name='last_fc')  # [1, path_embed_dim]

            else:  # self.path_learner == 'attention'
                ks = tf.stack(stacked_vec_ls, axis=0)  # [num_step + 1, num_elem, param_size]
                ks = tf.transpose(ks, [2, 0, 1])  # [param_size, num_step + 1, num_elem]
                ks = ks + positional_encoding(inp=ks, pos_encoding_type='sinusoid')  # [param_size, num_step + 1, num_elem]
                for layer in range(self.att_num_layer):
                    att_weights, ks = multihead_attention(queries=ks,
                                                          keys=ks,
                                                          num_units=self.att_hidden_dim,
                                                          num_heads=1,
                                                          dropout_rate=0,
                                                          is_training=True,
                                                          name='multihead_att{}'.format(layer + 1),
                                                          causality=True,
                                                          reuse=None,
                                                          with_qk=False,
                                                          with_res=False)  # [att_num_heads x param_size, num_step + 1, num_step + 1], [param_size, num_step + 1, att_hidden_dim]
                att_out = ks[:, -1, :]  # [param_size, att_hidden_dim]
                flat_vec = tf.reshape(att_out, [1, param_size * self.att_hidden_dim])  # [1, param_size x att_hidden_dim]
                self.path_emb = tf.layers.dense(flat_vec, units=self.path_embed_dim, activation=tf.nn.relu, name='last_fc')  # [1, path_embed_dim]

        return self.path_emb


class FeatExtractor(object):
    def __init__(self, config):

        self.config = config

        # feat_embed settings
        self.feat_embed_dim = config['embed_dim']
        self.feat_emb = None

    def model(self, inputa, labela, emb_w):
        """
        :param inputa: support x
        :param labela: support y
        :param emb_w: profile emb_w
        :return: feat_emb [1, feat_embed_size]
        """
        if self.config['data'] == 'movielens_1m':

            # item profile idx
            item_idx = tf.cast(inputa[:, 0], dtype=tf.int32)
            rate_idx = tf.cast(inputa[:, 1], dtype=tf.int32)
            genre_idx = inputa[:, 2:27]
            director_idx = inputa[:, 27:2213]
            actor_idx = inputa[:, 2213:10243]

            # user profile idx
            user_idx = tf.cast(inputa[:, 10243], dtype=tf.int32)
            gender_idx = tf.cast(inputa[:, 10244], dtype=tf.int32)
            age_idx = tf.cast(inputa[:, 10245], dtype=tf.int32)
            occupation_idx = tf.cast(inputa[:, 10246], dtype=tf.int32)
            zipcode_idx = tf.cast(inputa[:, 10247], dtype=tf.int32)

            # item profile embed
            item_emb = tf.nn.embedding_lookup(emb_w['item_emb_w'], item_idx)  # [update_batch_size, embed_dim]
            rate_emb = tf.nn.embedding_lookup(emb_w['rate_emb_w'], rate_idx)
            genre_emb = tf.matmul(genre_idx, emb_w['genre_emb_w']) / tf.reduce_sum(genre_idx, -1, keepdims=True)
            director_emb = tf.matmul(director_idx, emb_w['director_emb_w']) / tf.reduce_sum(director_idx, -1, keepdims=True)
            actor_emb = tf.matmul(actor_idx, emb_w['actor_emb_w']) / tf.reduce_sum(actor_idx, -1, keepdims=True)

            # user profile embed
            user_emb = tf.nn.embedding_lookup(emb_w['user_emb_w'], user_idx)
            gender_emb = tf.nn.embedding_lookup(emb_w['gender_emb_w'], gender_idx)
            age_emb = tf.nn.embedding_lookup(emb_w['age_emb_w'], age_idx)
            occupation_emb = tf.nn.embedding_lookup(emb_w['occupation_emb_w'], occupation_idx)
            zipcode_emb = tf.nn.embedding_lookup(emb_w['zipcode_emb_w'], zipcode_idx)

            inputs = tf.concat([item_emb, rate_emb, genre_emb, director_emb, actor_emb,
                                user_emb, gender_emb, age_emb, occupation_emb, zipcode_emb,
                                labela], axis=-1)  # [update_batch_size, embed_dim x 10 + 1]

        elif self.config['data'] == 'yelp':

            # item profile idx
            item_idx = tf.cast(inputa[:, 0], dtype=tf.int32)
            city_idx = tf.cast(inputa[:, 1], dtype=tf.int32)
            cate_idx = inputa[:, 2:1319]

            # user profile idx
            user_idx = tf.cast(inputa[:, 1319], dtype=tf.int32)

            # item profile embed
            item_emb = tf.nn.embedding_lookup(emb_w['item_emb_w'], item_idx)
            city_emb = tf.nn.embedding_lookup(emb_w['city_emb_w'], city_idx)  # [update_batch_size, embed_dim]
            cate_emb = tf.matmul(cate_idx, emb_w['cate_emb_w']) / tf.reduce_sum(cate_idx, -1, keepdims=True)

            # user profile embed
            user_emb = tf.nn.embedding_lookup(emb_w['user_emb_w'], user_idx)

            inputs = tf.concat([item_emb, city_emb, cate_emb, user_emb, labela], axis=-1)  # [update_batch_size, embed_dim x 4 + 1]

        else:

            # item profile idx
            item_idx = tf.cast(inputa[:, 0], dtype=tf.int32)
            brand_idx = tf.cast(inputa[:, 1], dtype=tf.int32)
            cate_idx = inputa[:, 2:492]

            # user profile idx
            user_idx = tf.cast(inputa[:, 492], dtype=tf.int32)

            # item profile embed
            item_emb = tf.nn.embedding_lookup(emb_w['item_emb_w'], item_idx)
            brand_emb = tf.nn.embedding_lookup(emb_w['brand_emb_w'], brand_idx)  # [update_batch_size, embed_dim]
            cate_emb = tf.matmul(cate_idx, emb_w['cate_emb_w']) / tf.reduce_sum(cate_idx, -1, keepdims=True)

            # user profile embed
            user_emb = tf.nn.embedding_lookup(emb_w['user_emb_w'], user_idx)

            inputs = tf.concat([item_emb, brand_emb, cate_emb, user_emb, labela], axis=-1)  # [update_batch_size, embed_dim x 4 + 1]

        with tf.variable_scope('feature_extractor', reuse=tf.AUTO_REUSE):
            outputs = tf.layers.dense(inputs, units=self.feat_embed_dim, activation=tf.nn.relu, name='last_fc')
            self.feat_emb = tf.reduce_mean(outputs, axis=0, keepdims=True)  # [1, feat_embed_dim]

        return self.feat_emb
