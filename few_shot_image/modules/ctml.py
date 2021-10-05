from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from task_embedding import Kmeans, LSTMAutoencoder, PathLearner
from metadag import MetaGraph
from feature_extractor import FeatEmbedding
from utils import xent, conv_block


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
        self.feat_embed = FeatEmbedding(config, hidden_num=32, channels=self.channels, conv_initializer=tf.truncated_normal_initializer(stddev=0.04))
        self.lstmae = LSTMAutoencoder(config, hidden_num=config['embed_dim'], input_num=config['embed_dim'] + config['num_class'], name='lstm_ae')
        self.lstmae_graph = LSTMAutoencoder(config, hidden_num=config['embed_dim'], input_num=config['embed_dim'], name='lstm_ae_graph')
        self.metagraph = MetaGraph(config, input_dim=config['embed_dim'], hidden_dim=config['embed_dim'])
        self.path_learner = PathLearner(config)
        self.kmeans = Kmeans(config)

        self.loss_func = xent
        self.forward = self.forward
        self.weights = self.construct_weights()

        self.inputa = None
        self.labela = None
        self.inputb = None
        self.labelb = None

        self.total_loss1 = None
        self.total_losses2 = None
        self.total_accuracy1 = None
        self.total_accuracies2 = None
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

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):

            num_step = max(self.test_num_step, self.config['num_step'])

            def task_metalearn(inp, reuse=True):
                inputa, inputb, labela, labelb = inp

                input_task_emb = self.feat_embed.model(tf.reshape(inputa, [-1, self.img_size, self.img_size, self.channels]))  # [inputa_size, 128]

                proto_emb = []
                labela2idx = tf.argmax(labela, axis=1)
                for class_idx in range(self.config['num_class']):
                    tmp_gs = tf.equal(labela2idx, class_idx)
                    gs = tf.where(tmp_gs)
                    new_vec = tf.reduce_mean(tf.gather(input_task_emb, gs), axis=0)
                    proto_emb.append(new_vec)
                proto_emb = tf.squeeze(tf.stack(proto_emb))

                label_cat = tf.eye(5)

                input_task_emb_cat = tf.concat((proto_emb, label_cat), axis=-1)

                feat_embed_vec, feat_emb_loss = self.lstmae.model(input_task_emb_cat)  # [1, 128], []
                propagate_knowledge = self.metagraph.model(proto_emb)  # [1, 128]

                path_embed_vec, stacked_vec_ls = self.path_learner.model(inputa, labela, self.forward, self.weights, self.loss_func)  # [1, 128], list of tensor with shapes [[num_elem, param_size]_1, ..., [num_elem, param_size]_num_update]

                feat_embed_vec_graph, feat_emb_loss_graph = self.lstmae_graph.model(propagate_knowledge)  # [1, 128], []
                _, path_embed_vec_kmeans = self.kmeans.model(path_embed_vec)  # [1, path_num_cluster], [1, 128]

                feat_enhanced_emb_vec = tf.concat([feat_embed_vec, feat_embed_vec_graph], axis=1)  # [1, 256]
                path_enhanced_emb_vec = tf.concat([path_embed_vec, path_embed_vec_kmeans], axis=1)  # [1, 256]

                path_feat_probe = tf.nn.sigmoid(
                    tf.get_variable(name='path_feat_probe', shape=feat_enhanced_emb_vec.get_shape().as_list(),
                                    initializer=tf.constant_initializer(0)))

                if self.config['path_or_feat'] == 'only_path':
                    enhanced_emb_vec = path_enhanced_emb_vec
                elif self.config['path_or_feat'] == 'only_feat':
                    enhanced_emb_vec = feat_enhanced_emb_vec
                else:  # self.config['path_or_feat'] == 'both'
                    enhanced_emb_vec = path_feat_probe * path_enhanced_emb_vec + (1 - path_feat_probe) * feat_enhanced_emb_vec  # [1, 256]

                with tf.variable_scope('task_specific_mapping', reuse=tf.AUTO_REUSE):
                    eta = []
                    for key in self.weights.keys():
                        weight_size = np.prod(self.weights[key].get_shape().as_list())
                        eta.append(tf.reshape(
                            tf.layers.dense(enhanced_emb_vec, weight_size, activation=tf.nn.sigmoid, name='eta_{}'.format(key)), tf.shape(self.weights[key])))
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
                task_lossesb.append(self.loss_func(output, labelb, self.config['query_size']))  # TODO: check

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
                    task_lossesb.append(self.loss_func(output, labelb, self.config['query_size']))  # TODO: check

                # [inputa_size, num_class], [[inputb_size, num_class]] * num_step, [inputa_size], [[inputb_size]] * num_step
                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]

                task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), 1), tf.argmax(labela, 1))
                for j in range(num_step):
                    task_accuraciesb.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputbs[j]), 1), tf.argmax(labelb, 1)))
                # [], [[]] * num_step
                task_output.extend([task_accuracya, task_accuraciesb])

                return task_output

            unused = task_metalearn((self.inputa[0], self.inputb[0], one_hot_labela[0], one_hot_labelb[0]), False)
            out_dtype = [tf.float32, [tf.float32] * num_step, tf.float32, [tf.float32] * num_step, tf.float32, [tf.float32] * num_step]
            result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, one_hot_labela, one_hot_labelb),
                               dtype=out_dtype, parallel_iterations=self.config['meta_batch_size'])
            # [meta_batch_size, inputa_size, num_class], [[meta_batch_size, inputa_size, num_class]] * num_step,
            # [meta_batch_size, inputa_size], [[meta_batch_size, inputb_size]] * num_step
            # [meta_batch_size], [[meta_batch_size]] * num_step
            outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb = result

        # Performance & Optimization
        self.total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(self.config['meta_batch_size'])
        self.total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(self.config['meta_batch_size']) for j in range(num_step)]
        self.total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(self.config['meta_batch_size'])
        self.total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.to_float(self.config['meta_batch_size']) for j in range(num_step)]

        optimizer = tf.train.AdamOptimizer(self.meta_lr)
        gvs = optimizer.compute_gradients(self.total_losses2[self.config['num_step'] - 1])
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

        weights['conv1'] = tf.get_variable('conv1', [k, k, self.channels, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv2'] = tf.get_variable('conv2', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv3'] = tf.get_variable('conv3', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv4'] = tf.get_variable('conv4', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]))

        weights['w5'] = tf.get_variable('w5', [self.dim_hidden * 5 * 5, self.dim_output], initializer=fc_initializer)
        weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        return weights

    def forward(self, inp, weights, reuse=False, scope=''):
        channels = self.channels
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])

        hidden1 = conv_block(inp, weights['conv1'], weights['b1'], reuse, scope + '0')
        hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse, scope + '1')
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse, scope + '2')
        hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse, scope + '3')

        hidden4 = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])])

        return tf.matmul(hidden4, weights['w5']) + weights['b5']
