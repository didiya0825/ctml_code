from __future__ import print_function
from __future__ import division

import tensorflow as tf


class FeatEmbedding(object):
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
        with tf.variable_scope('feat_embed', reuse=tf.AUTO_REUSE):

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

            outputs = tf.layers.dense(inputs, units=self.feat_embed_dim, activation=tf.nn.relu, name='last_fc')
            self.feat_emb = tf.reduce_mean(outputs, axis=0, keepdims=True)  # [1, feat_embed_dim]

        return self.feat_emb
