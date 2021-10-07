from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from utils import xent, conv_block, mse, js_divergence
from task_rep import Kmeans, PathLearner


class CTML:
    def __init__(self, sess, config, test_num_step, dim_input=1, dim_output=1):
        self.sess = sess
        self.config = config
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.test_num_step = test_num_step
        self.adapt_lr = config['adapt_lr']
        self.meta_lr = tf.placeholder_with_default(config['meta_lr'], ())

        self.dim_hidden = 32  # number of filters
        self.channels = 3
        self.img_size = int(np.sqrt(self.dim_input / self.channels))
        self.path_learner = PathLearner(config)
        self.kmeans = Kmeans(config)
        if config['recon_loss_func'] == 'js_div':
            self.recon_loss_func = js_divergence
        else:  # config['recon_loss_func'] == 'mse'
            self.recon_loss_func = mse

        self.loss_func = xent
        self.weights = self.construct_weights()
        self.weights_feat = self.construct_weights_feat()

        self.inputa = None
        self.labela = None
        self.inputb = None
        self.labelb = None

        self.total_loss1 = None
        self.total_losses2 = None
        self.total_accuracy1 = None
        self.total_accuracies2 = None
        self.shortcut_loss = None
        self.metatrain_op = None

    def construct_model(self, train=True):
        self.inputa = tf.placeholder(tf.float32, shape=(None, self.config['num_class'] * self.config['support_size'], 21168))
        self.labela = tf.placeholder(tf.int32, shape=(None, self.config['num_class'] * self.config['support_size']))  # [4, no. of samples per task]
        if train:
            self.inputb = tf.placeholder(tf.float32, shape=(None, self.config['num_class'] * self.config['query_size'], 21168))
            self.labelb = tf.placeholder(tf.int32, shape=(None, self.config['num_class'] * self.config['query_size']))
        else:
            self.inputb = tf.placeholder(tf.float32, shape=(None, self.config['num_class'] * self.config['support_size'], 21168))
            self.labelb = tf.placeholder(tf.int32, shape=(None, self.config['num_class'] * self.config['support_size']))
        one_hot_labela = tf.cast(tf.one_hot(self.labela, self.dim_output), tf.float32)  # [4, no. of samples per task, num_class]
        one_hot_labelb = tf.cast(tf.one_hot(self.labelb, self.dim_output), tf.float32)

        num_step = max(self.test_num_step, self.config['num_step'])

        def task_metalearn(inp, reuse=True):
            inputa, inputb, labela, labelb = inp

            # extract feature embedding
            if self.config['support_size'] == 1:  # for 1-shot case, better use a separate feature extractor
                weights_feat = self.weights_feat
            else:  # else, use the same feature extractor as the base-learner
                weights_feat = {}
                for k, v in self.weights.items():
                    weights_feat[k] = tf.stop_gradient(v)
            input_feat = self.forward_feat(inputa, weights_feat, reuse=reuse)
            with tf.variable_scope('feature_extractor', reuse=tf.AUTO_REUSE):
                w = tf.get_variable('w', [input_feat.get_shape().as_list()[-1], self.config['embed_dim']],
                                    initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
                b = tf.get_variable('b', [self.config['embed_dim']],
                                    initializer=tf.constant_initializer(0.0))
                input_feat = tf.matmul(input_feat, w) + b  # [inputa_size, embed_dim]
                feat_emb = tf.reduce_mean(input_feat, axis=0, keepdims=True)  # [1, embed_dim]

            # extract path embedding
            path_emb = self.path_learner.model(inputa, labela, self.forward, self.weights, self.loss_func)  # [1, embed_dim]

            # clustering
            feat_assignment, clustered_feat_embed = self.kmeans.feat_model(feat_emb)  # [1, feat_num_cluster], [1, embed_dim]
            path_assignment, clustered_path_embed = self.kmeans.path_model(path_emb)  # [1, path_num_cluster], [1, embed_dim]

            # shortcut tunnel
            with tf.variable_scope('shortcut_tunnel', reuse=tf.AUTO_REUSE):
                if self.config['shortcut']:
                    recon_path_assignment = feat_assignment
                    for i in range(1, self.config['shortcut_num_layer']):
                        recon_path_assignment = tf.layers.dense(recon_path_assignment,
                                                                units=self.config['path_num_cluster'],
                                                                use_bias=False,
                                                                activation=tf.nn.relu,
                                                                name='mapping_{}'.format(i))  # [1, path_num_cluster]
                    recon_path_assignment = tf.layers.dense(recon_path_assignment,
                                                            units=self.config['path_num_cluster'],
                                                            use_bias=False,
                                                            activation=tf.nn.softmax,
                                                            name='mapping_{}'.format(self.config['shortcut_num_layer']))  # [1, path_num_cluster]
                    recon_loss = self.recon_loss_func(path_assignment, recon_path_assignment)  # scalar

                    if self.config['use_shortcut_approx']:
                        path_assignment = recon_path_assignment  # [1, path_num_cluster]
                        clustered_path_embed = self.kmeans.path_model_shortcut(path_assignment)  # [1, embed_dim]

            # aggregate
            with tf.variable_scope('aggregation_weight', reuse=tf.AUTO_REUSE):
                path_feat_probe = tf.nn.sigmoid(
                    tf.get_variable(name='lambda', shape=clustered_feat_embed.get_shape().as_list(),
                                    initializer=tf.constant_initializer(0)))
            if self.config['path_or_feat'] == 'path':
                task_embed = clustered_path_embed  # [1, embed_dim]
            elif self.config['path_or_feat'] == 'feat':
                task_embed = clustered_feat_embed  # [1, embed_dim]
            else:  # self.config['path_or_feat'] == 'path_and_feat'
                task_embed = path_feat_probe * clustered_path_embed + (1 - path_feat_probe) * clustered_feat_embed  # [1, embed_dim]

            # task-aware modulation
            with tf.variable_scope('task_aware_modulation', reuse=tf.AUTO_REUSE):
                eta = []
                for key in self.weights.keys():
                    weight_size = np.prod(self.weights[key].get_shape().as_list())
                    eta.append(tf.reshape(
                        tf.layers.dense(task_embed, weight_size, activation=tf.nn.sigmoid, name='eta_{}'.format(key)), tf.shape(self.weights[key])))
                eta = dict(zip(self.weights.keys(), eta))
                task_weights = dict(zip(self.weights.keys(), [self.weights[key] * eta[key] for key in self.weights.keys()]))

            task_outputbs, task_lossesb, task_accuraciesb = [], [], []

            task_outputa = self.forward(inputa, task_weights, reuse=reuse)
            task_lossa = self.loss_func(task_outputa, labela, self.config['support_size'])
            grads = tf.gradients(task_lossa, list(task_weights.values()))
            if self.config['stop_grad']:
                grads = [tf.stop_gradient(grad) for grad in grads]
            gradients = dict(zip(task_weights.keys(), grads))
            fast_weights = dict(
                zip(task_weights.keys(),
                    [task_weights[key] - self.adapt_lr * gradients[key] for key in task_weights.keys()]))
            output = self.forward(inputb, fast_weights, reuse=True)
            task_outputbs.append(output)
            task_lossesb.append(self.loss_func(output, labelb, self.config['query_size']))

            for j in range(num_step - 1):
                loss = self.loss_func(self.forward(inputa, fast_weights, reuse=True), labela, self.config['support_size'])
                grads = tf.gradients(loss, list(fast_weights.values()))
                if self.config['stop_grad']:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(fast_weights.keys(), grads))
                fast_weights = dict(
                    zip(fast_weights.keys(),
                        [fast_weights[key] - self.adapt_lr * gradients[key] for key in fast_weights.keys()]))
                output = self.forward(inputb, fast_weights, reuse=True)
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, labelb, self.config['query_size']))

            # [inputa_size, num_class], [[inputb_size, num_class]] * num_step, [inputa_size], [[inputb_size]] * num_step
            task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]

            task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), 1), tf.argmax(labela, 1))
            for j in range(num_step):
                task_accuraciesb.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputbs[j]), 1), tf.argmax(labelb, 1)))
            # [], [[]] * num_step
            task_output.extend([task_accuracya, task_accuraciesb])
            if self.config['shortcut']:
                # []
                task_output.append(recon_loss)

            return task_output

        unused = task_metalearn((self.inputa[0], self.inputb[0], one_hot_labela[0], one_hot_labelb[0]), False)
        out_dtype = [tf.float32, [tf.float32] * num_step, tf.float32, [tf.float32] * num_step, tf.float32, [tf.float32] * num_step]
        if self.config['shortcut']:
            out_dtype.append(tf.float32)
        result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, one_hot_labela, one_hot_labelb),
                           dtype=out_dtype, parallel_iterations=self.config['meta_batch_size'])
        if self.config['shortcut']:
            # [meta_batch_size, inputa_size, num_class], [[meta_batch_size, inputa_size, num_class]] * num_step,
            # [meta_batch_size, inputa_size], [[meta_batch_size, inputb_size]] * num_step
            # [meta_batch_size], [[meta_batch_size]] * num_step, []
            outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb, shortcut_loss = result
        else:
            # [meta_batch_size, inputa_size, num_class], [[meta_batch_size, inputa_size, num_class]] * num_step,
            # [meta_batch_size, inputa_size], [[meta_batch_size, inputb_size]] * num_step
            # [meta_batch_size], [[meta_batch_size]] * num_step
            outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb = result

        # Performance & Optimization
        self.total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(self.config['meta_batch_size'])
        self.total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(self.config['meta_batch_size']) for j in range(num_step)]
        self.total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(self.config['meta_batch_size'])
        self.total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.to_float(self.config['meta_batch_size']) for j in range(num_step)]
        overall_loss = self.total_losses2[self.config['num_step'] - 1]
        if self.config['shortcut']:
            self.shortcut_loss = tf.reduce_sum(shortcut_loss) / tf.to_float(self.config['meta_batch_size'])
            overall_loss += self.config['recon_loss_weight'] * self.shortcut_loss

        optimizer = tf.train.AdamOptimizer(self.meta_lr)
        gvs = optimizer.compute_gradients(overall_loss)
        self.metatrain_op = optimizer.apply_gradients(gvs)

    def construct_weights(self):
        """
        base-learner parameters theta_0
        """
        weights = {}

        dtype = tf.float32
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
        k = 3

        with tf.variable_scope('base_learner', reuse=tf.AUTO_REUSE):
            weights['conv1'] = tf.get_variable('conv1', [k, k, self.channels, self.dim_hidden], initializer=conv_initializer)
            weights['b1'] = tf.get_variable('b1', [self.dim_hidden], initializer=conv_initializer)
            weights['conv2'] = tf.get_variable('conv2', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer)
            weights['b2'] = tf.get_variable('b2', [self.dim_hidden], initializer=conv_initializer)
            weights['conv3'] = tf.get_variable('conv3', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer)
            weights['b3'] = tf.get_variable('b3', [self.dim_hidden], initializer=conv_initializer)
            weights['conv4'] = tf.get_variable('conv4', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer)
            weights['b4'] = tf.get_variable('b4', [self.dim_hidden], initializer=conv_initializer)

            weights['w5'] = tf.get_variable('w5', [self.dim_hidden * 5 * 5, self.dim_output], initializer=fc_initializer)
            weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        return weights

    def forward(self, inp, weights, reuse=False, scope=''):
        channels = self.channels
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])

        with tf.variable_scope('base_learner', reuse=tf.AUTO_REUSE):
            hidden1 = conv_block(inp, weights['conv1'], weights['b1'], reuse, scope + '0')
            hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse, scope + '1')
            hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse, scope + '2')
            hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse, scope + '3')

            hidden4 = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])])

        return tf.matmul(hidden4, weights['w5']) + weights['b5']

    def construct_weights_feat(self):
        """
        separate feature extractor if do not reuse that of the base-learner
        """
        weights = {}

        dtype = tf.float32
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        k = 3

        with tf.variable_scope('feature_extractor', reuse=tf.AUTO_REUSE):
            weights['conv1'] = tf.get_variable('conv1', [k, k, self.channels, self.dim_hidden], initializer=conv_initializer)
            weights['b1'] = tf.get_variable('b1', [self.dim_hidden], initializer=conv_initializer)
            weights['conv2'] = tf.get_variable('conv2', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer)
            weights['b2'] = tf.get_variable('b2', [self.dim_hidden], initializer=conv_initializer)
            weights['conv3'] = tf.get_variable('conv3', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer)
            weights['b3'] = tf.get_variable('b3', [self.dim_hidden], initializer=conv_initializer)
            weights['conv4'] = tf.get_variable('conv4', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer)
            weights['b4'] = tf.get_variable('b4', [self.dim_hidden], initializer=conv_initializer)
        return weights

    def forward_feat(self, inp, weights, reuse=False, scope=''):
        channels = self.channels
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])

        with tf.variable_scope('feature_extractor', reuse=tf.AUTO_REUSE):
            hidden1 = conv_block(inp, weights['conv1'], weights['b1'], reuse, scope + '0')
            hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse, scope + '1')
            hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse, scope + '2')
            hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse, scope + '3')

            hidden4 = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])])

        return hidden4
