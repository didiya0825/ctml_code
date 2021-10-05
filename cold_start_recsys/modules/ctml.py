from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from utils import mse, mae
from task_embedding import Kmeans, PathLearner
from feature_extractor import FeatEmbedding


class CTML:
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config
        self.adapt_lr = config['adapt_lr']
        self.meta_lr = tf.placeholder_with_default(config['meta_lr'], ())

        self.feat_embed = FeatEmbedding(config)
        self.path_learner = PathLearner(config)
        self.kmeans = Kmeans(config)

        self.loss_func = mse
        self.forward = self.forward
        self.weights = self.construct_weights()

        # a: train data for inner gradient, b: test data for meta gradient
        if self.config['data'] == 'movielens_1m':
            input_dim = 10248
        elif self.config['data'] == 'yelp':
            input_dim = 1320
        else:
            input_dim = 493
        self.a_size = tf.placeholder(tf.int32, shape=None)
        self.inputa = tf.placeholder(tf.float32, shape=(None, None, input_dim))
        self.labela = tf.placeholder(tf.float32, shape=(None, None, 1))
        self.b_size = tf.placeholder(tf.int32, shape=None)
        self.inputb = tf.placeholder(tf.float32, shape=(None, None, input_dim))
        self.labelb = tf.placeholder(tf.float32, shape=(None, None, 1))

        self.mean_meta_loss = None
        self.metatrain_op = None
        self.eval_meta_loss = None
        self.eval_mae = None

    def task_metalearn(self, inp):
        """
        Perform gradient descent for one task in the meta-batch.
        :param inp: (inputa[i], inputb[i], labela[i], labelb[i]) each of shape [a_size, 10246 or 1] or [update_batch_size_eval, 10246 or 1]
        :return: outputs and losses
        """
        inputa, labela, inputb, labelb = inp

        with tf.variable_scope('task_embedding', reuse=tf.AUTO_REUSE):
            emb_w = {}
            for k, v in self.weights.items():
                if 'emb' in k:
                    emb_w[k] = tf.stop_gradient(v)
            feat_embed_vec = self.feat_embed.model(inputa, labela, emb_w)  # [1, feat_embed_dim]
            path_embed_vec, stacked_vec_ls = self.path_learner.model(inputa, labela, self.forward, self.weights, self.loss_func)  # [1, path_embed_dim]

        with tf.variable_scope('task_clustering', reuse=tf.AUTO_REUSE):
            feat_soft_assignment, feat_embed_vec_kmeans = self.kmeans.feat_model(feat_embed_vec)  # [1, feat_num_cluster], [1, feat_embed_dim]
            path_soft_assignment, path_embed_vec_kmeans = self.kmeans.path_model(path_embed_vec)  # [1, path_num_cluster], [1, path_embed_dim]

            feat_enhanced_emb_vec = tf.concat([feat_embed_vec, feat_embed_vec_kmeans], axis=1)  # [1, feat_embed_dim x 2]
            path_enhanced_emb_vec = tf.concat([path_embed_vec, path_embed_vec_kmeans], axis=1)  # [1, path_embed_dim x 2]

            path_feat_probe = tf.nn.sigmoid(
                tf.get_variable(name='path_feat_probe', shape=feat_enhanced_emb_vec.get_shape().as_list(),  # [1, embed_dim x 2]
                                initializer=tf.constant_initializer(0)))

            if self.config['path_or_feat'] == 'only_path':
                enhanced_emb_vec = path_enhanced_emb_vec
            elif self.config['path_or_feat'] == 'only_feat':
                enhanced_emb_vec = feat_enhanced_emb_vec
            else:  # self.config['path_or_feat'] == 'both'
                enhanced_emb_vec = path_feat_probe * path_enhanced_emb_vec + (1 - path_feat_probe) * feat_enhanced_emb_vec  # [1, embed_dim x 2]

        with tf.variable_scope('task_specific_mapping', reuse=tf.AUTO_REUSE):
            eta = []
            gated_weight_keys = [k for k in self.weights.keys() if 'fcn' in k]
            for key in gated_weight_keys:
                weight_size = np.prod(self.weights[key].get_shape().as_list())
                eta.append(tf.reshape(
                    tf.layers.dense(enhanced_emb_vec, weight_size, activation=tf.nn.sigmoid, name='eta_{}'.format(key)), tf.shape(self.weights[key])))
            eta = dict(zip(gated_weight_keys, eta))

            gated_weights = {}
            for key in self.weights.keys():
                if key in gated_weight_keys:
                    gated_weights[key] = self.weights[key] * eta[key]  # theta_0 is adapted to task-specific theta_0i by multiplying parameter gate
                else:
                    gated_weights[key] = self.weights[key]
            fast_weights = gated_weights

        for j in range(self.config['num_step']):
            output = self.forward(inputa, fast_weights)
            loss = self.loss_func(output, labela)
            grads = tf.gradients(loss, list(fast_weights.values()))
            gradients = dict(zip(fast_weights.keys(), grads))
            if self.config['stop_grad']:  # FOMAML
                for k, v in gradients.items():
                    gradients[k] = tf.stop_gradient(v)
            else:
                for k, v in gradients.items():
                    if 'emb_w' in k:
                        gradients[k] = tf.stop_gradient(v)
            fast_weights = dict(
                zip(fast_weights.keys(),
                    [fast_weights[key] - self.adapt_lr * gradients[key] for key in fast_weights.keys()]))

        task_outputb = self.forward(inputb, fast_weights)
        task_lossb = self.loss_func(task_outputb, labelb)
        task_mae = mae(task_outputb, labelb)
        # task_ndcg20 = nDCG(task_outputb, labelb, topk=20)

        return task_lossb, task_mae

    def construct_model(self):

        lossb_ls = []

        for i in range(self.config['meta_batch_size']):
            a_size = self.a_size[i]
            task_inputa = self.inputa[i][:a_size]
            task_labela = self.labela[i][:a_size]
            b_size = self.b_size[i]
            task_inputb = self.inputb[i][:b_size]
            task_labelb = self.labelb[i][:b_size]
            result = self.task_metalearn([task_inputa, task_labela, task_inputb, task_labelb])
            lossb_ls.append(result[0])

        # metatrain_op
        self.mean_meta_loss = tf.reduce_sum(lossb_ls) / self.config['meta_batch_size']  # scalar
        optimizer = tf.train.AdamOptimizer(self.meta_lr)
        # compute gradients of global theta_0, path learner, clustering network, and task-aware modulation
        gvs = optimizer.compute_gradients(self.mean_meta_loss)
        self.metatrain_op = optimizer.apply_gradients(gvs)

        # eval
        task_inputa = self.inputa[0]
        task_labela = self.labela[0]
        task_inputb = self.inputb[0]
        task_labelb = self.labelb[0]
        result = self.task_metalearn([task_inputa, task_labela, task_inputb, task_labelb])
        self.eval_meta_loss = result[0]
        self.eval_mae = result[1]

    def construct_weights(self):
        """
        base-learner parameters theta_0
        profile embedding layers [num, base_embed_dim] & MLP layers [fcn1_hidden_dim, fcn2_hidden_dim]
        :return: weights dict {'rate_emb_w': tf.Variable, 'fcn1/kernel': tf.Variable, ...}
        """
        if self.config['data'] == 'movielens_1m':
            weights = {
                # create item profile emb_w
                'item_emb_w': tf.get_variable("item_emb_w", [self.config['num_item'], self.config['base_embed_dim']]),
                'rate_emb_w': tf.get_variable("rate_emb_w", [self.config['num_rate'], self.config['base_embed_dim']]),
                'genre_emb_w': tf.get_variable("genre_emb_w", [self.config['num_genre'], self.config['base_embed_dim']]),
                'director_emb_w': tf.get_variable("director_emb_w", [self.config['num_director'], self.config['base_embed_dim']]),
                'actor_emb_w': tf.get_variable("actor_emb_w", [self.config['num_actor'], self.config['base_embed_dim']]),

                # create user profile emb_w
                'user_emb_w': tf.get_variable("user_emb_w", [self.config['num_user'], self.config['base_embed_dim']]),
                'gender_emb_w': tf.get_variable("gender_emb_w", [self.config['num_gender'], self.config['base_embed_dim']]),
                'age_emb_w': tf.get_variable("age_emb_w", [self.config['num_age'], self.config['base_embed_dim']]),
                'occupation_emb_w': tf.get_variable("occupation_emb_w", [self.config['num_occupation'], self.config['base_embed_dim']]),
                'zipcode_emb_w': tf.get_variable("zipcode_emb_w", [self.config['num_zipcode'], self.config['base_embed_dim']]),

                # create mlp
                'fcn1/kernel': tf.get_variable(name='fcn1/kernel', shape=[self.config['base_embed_dim'] * 10, self.config['fcn1_hidden_dim']]),
                'fcn1/bias': tf.get_variable(name='fcn1/bias', shape=[self.config['fcn1_hidden_dim']]),
                'fcn2/kernel': tf.get_variable(name='fcn2/kernel', shape=[self.config['fcn1_hidden_dim'], self.config['fcn2_hidden_dim']]),
                'fcn2/bias': tf.get_variable(name='fcn2/bias', shape=[self.config['fcn2_hidden_dim']]),
                'fcn3/kernel': tf.get_variable(name='fcn3/kernel', shape=[self.config['fcn2_hidden_dim'], 1]),
                'fcn3/bias': tf.get_variable(name='fcn3/bias', shape=[1])
            }

        elif self.config['data'] == 'yelp':
            weights = {
                # create item profile emb_w
                'item_emb_w': tf.get_variable("item_emb_w", [self.config['num_item'], self.config['base_embed_dim']]),
                'city_emb_w': tf.get_variable("city_emb_w", [self.config['num_city'], self.config['base_embed_dim']]),
                'cate_emb_w': tf.get_variable("cate_emb_w", [self.config['num_cate'], self.config['base_embed_dim']]),

                # create user profile emb_w
                'user_emb_w': tf.get_variable("user_emb_w", [self.config['num_user'], self.config['base_embed_dim']]),

                # create mlp
                'fcn1/kernel': tf.get_variable(name='fcn1/kernel', shape=[self.config['base_embed_dim'] * 4, self.config['fcn1_hidden_dim']]),
                'fcn1/bias': tf.get_variable(name='fcn1/bias', shape=[self.config['fcn1_hidden_dim']]),
                'fcn2/kernel': tf.get_variable(name='fcn2/kernel', shape=[self.config['fcn1_hidden_dim'], self.config['fcn2_hidden_dim']]),
                'fcn2/bias': tf.get_variable(name='fcn2/bias', shape=[self.config['fcn2_hidden_dim']]),
                'fcn3/kernel': tf.get_variable(name='fcn3/kernel', shape=[self.config['fcn2_hidden_dim'], 1]),
                'fcn3/bias': tf.get_variable(name='fcn3/bias', shape=[1])
            }

        else:
            weights = {
                # create item profile emb_w
                'item_emb_w': tf.get_variable("item_emb_w", [self.config['num_item'], self.config['base_embed_dim']]),
                'brand_emb_w': tf.get_variable("brand_emb_w", [self.config['num_brand'], self.config['base_embed_dim']]),
                'cate_emb_w': tf.get_variable("cate_emb_w", [self.config['num_cate'], self.config['base_embed_dim']]),

                # create user profile emb_w
                'user_emb_w': tf.get_variable("user_emb_w", [self.config['num_user'], self.config['base_embed_dim']]),

                # create mlp
                'fcn1/kernel': tf.get_variable(name='fcn1/kernel', shape=[self.config['base_embed_dim'] * 4, self.config['fcn1_hidden_dim']]),
                'fcn1/bias': tf.get_variable(name='fcn1/bias', shape=[self.config['fcn1_hidden_dim']]),
                'fcn2/kernel': tf.get_variable(name='fcn2/kernel', shape=[self.config['fcn1_hidden_dim'], self.config['fcn2_hidden_dim']]),
                'fcn2/bias': tf.get_variable(name='fcn2/bias', shape=[self.config['fcn2_hidden_dim']]),
                'fcn3/kernel': tf.get_variable(name='fcn3/kernel', shape=[self.config['fcn2_hidden_dim'], 1]),
                'fcn3/bias': tf.get_variable(name='fcn3/bias', shape=[1])
            }

        return weights

    def forward(self, inp, weights):
        """
        :param inp: [update_batch_size, 10248]
        :param weights: task_weights dict {'rate_emb_w': tf.Variable, 'fcn1/kernel': tf.Variable, ...}
        :return: output [update_batch_size/eval, 1]
        """
        if self.config['data'] == 'movielens_1m':

            # item profile idx
            item_idx = tf.cast(inp[:, 0], dtype=tf.int32)
            rate_idx = tf.cast(inp[:, 1], dtype=tf.int32)
            genre_idx = inp[:, 2:27]
            director_idx = inp[:, 27:2213]
            actor_idx = inp[:, 2213:10243]

            # user profile idx
            user_idx = tf.cast(inp[:, 10243], dtype=tf.int32)
            gender_idx = tf.cast(inp[:, 10244], dtype=tf.int32)
            age_idx = tf.cast(inp[:, 10245], dtype=tf.int32)
            occupation_idx = tf.cast(inp[:, 10246], dtype=tf.int32)
            zipcode_idx = tf.cast(inp[:, 10247], dtype=tf.int32)

            # item profile embed
            item_emb = tf.nn.embedding_lookup(weights['item_emb_w'], item_idx)  # [update_batch_size/eval, base_embed_dim]
            rate_emb = tf.nn.embedding_lookup(weights['rate_emb_w'], rate_idx)
            genre_emb = tf.matmul(genre_idx, weights['genre_emb_w']) / tf.reduce_sum(genre_idx, -1, keepdims=True)
            director_emb = tf.matmul(director_idx, weights['director_emb_w']) / tf.reduce_sum(director_idx, -1, keepdims=True)
            actor_emb = tf.matmul(actor_idx, weights['actor_emb_w']) / tf.reduce_sum(actor_idx, -1, keepdims=True)

            # user profile embed
            user_emb = tf.nn.embedding_lookup(weights['user_emb_w'], user_idx)
            gender_emb = tf.nn.embedding_lookup(weights['gender_emb_w'], gender_idx)
            age_emb = tf.nn.embedding_lookup(weights['age_emb_w'], age_idx)
            occupation_emb = tf.nn.embedding_lookup(weights['occupation_emb_w'], occupation_idx)
            zipcode_emb = tf.nn.embedding_lookup(weights['zipcode_emb_w'], zipcode_idx)

            # forward mlp
            fcn = tf.concat([item_emb, rate_emb, genre_emb, director_emb, actor_emb, user_emb, gender_emb, age_emb, occupation_emb, zipcode_emb], axis=-1)  # [update_batch_size/eval, base_embed_dim x 10]
            fcn_layer_1 = tf.nn.relu(tf.matmul(fcn, weights['fcn1/kernel']) + weights['fcn1/bias'])  # [update_batch_size/eval, l1]
            fcn_layer_2 = tf.nn.relu(tf.matmul(fcn_layer_1, weights['fcn2/kernel']) + weights['fcn2/bias'])  # [update_batch_size/eval, l2]
            linear_out = tf.matmul(fcn_layer_2, weights['fcn3/kernel']) + weights['fcn3/bias']  # [update_batch_size/eval, 1]

        elif self.config['data'] == 'yelp':

            # item profile idx
            item_idx = tf.cast(inp[:, 0], dtype=tf.int32)
            city_idx = tf.cast(inp[:, 1], dtype=tf.int32)
            cate_idx = inp[:, 2:1319]

            # user profile idx
            user_idx = tf.cast(inp[:, 1319], dtype=tf.int32)

            # item profile embed
            item_emb = tf.nn.embedding_lookup(weights['item_emb_w'], item_idx)  # [update_batch_size/eval, base_embed_dim]
            city_emb = tf.nn.embedding_lookup(weights['city_emb_w'], city_idx)
            cate_emb = tf.matmul(cate_idx, weights['cate_emb_w']) / tf.reduce_sum(cate_idx, -1, keepdims=True)

            # user profile embed
            user_emb = tf.nn.embedding_lookup(weights['user_emb_w'], user_idx)

            # forward mlp
            fcn = tf.concat([item_emb, city_emb, cate_emb, user_emb], axis=-1)  # [update_batch_size/eval, base_embed_dim x 4]
            fcn_layer_1 = tf.nn.relu(tf.matmul(fcn, weights['fcn1/kernel']) + weights['fcn1/bias'])  # [update_batch_size/eval, l1]
            fcn_layer_2 = tf.nn.relu(tf.matmul(fcn_layer_1, weights['fcn2/kernel']) + weights['fcn2/bias'])  # [update_batch_size/eval, l2]
            linear_out = tf.matmul(fcn_layer_2, weights['fcn3/kernel']) + weights['fcn3/bias']  # [update_batch_size/eval, 1]

        else:

            # item profile idx
            item_idx = tf.cast(inp[:, 0], dtype=tf.int32)
            brand_idx = tf.cast(inp[:, 1], dtype=tf.int32)
            cate_idx = inp[:, 2:492]

            # user profile idx
            user_idx = tf.cast(inp[:, 492], dtype=tf.int32)

            # item profile embed
            item_emb = tf.nn.embedding_lookup(weights['item_emb_w'], item_idx)  # [update_batch_size/eval, base_embed_dim]
            brand_emb = tf.nn.embedding_lookup(weights['brand_emb_w'], brand_idx)
            cate_emb = tf.matmul(cate_idx, weights['cate_emb_w']) / tf.reduce_sum(cate_idx, -1, keepdims=True)

            # user profile embed
            user_emb = tf.nn.embedding_lookup(weights['user_emb_w'], user_idx)

            # forward mlp
            fcn = tf.concat([item_emb, brand_emb, cate_emb, user_emb], axis=-1)  # [update_batch_size/eval, base_embed_dim x 4]
            fcn_layer_1 = tf.nn.relu(tf.matmul(fcn, weights['fcn1/kernel']) + weights['fcn1/bias'])  # [update_batch_size/eval, l1]
            fcn_layer_2 = tf.nn.relu(tf.matmul(fcn_layer_1, weights['fcn2/kernel']) + weights['fcn2/bias'])  # [update_batch_size/eval, l2]
            linear_out = tf.matmul(fcn_layer_2, weights['fcn3/kernel']) + weights['fcn3/bias']  # [update_batch_size/eval, 1]

        return linear_out
