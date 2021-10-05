from __future__ import print_function
from __future__ import division

import os
import pickle
import random
import sys
import time
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf

from modules.ctml import CTML
from modules.configs import *

np.random.seed(123)
random.seed(123)
tf.set_random_seed(123)


def train(config, model, saver, sess, exp_string, datasets, resume_epoch=1):

    print('Done initializing, starting training.')
    sys.stdout.flush()

    start_time = time.time()

    overall_maes, ws_maes, cs_maes = [], [], []  # test mae

    for epoch in range(resume_epoch, config['num_epoch'] + 1):

        print('Epoch {} start!'.format(epoch))

        train_set_size = len(datasets['train'])  # number of tasks in train
        random.shuffle(datasets['train'])
        num_batch = int(train_set_size / config['meta_batch_size'])  # // round down, e.g., int(100/30)=3
        supp_sets, query_sets = list(zip(*datasets['train']))  # [(T1_support, T2_support, ...), (T1_query, T2_query, ...)]
        assert len(supp_sets) == len(query_sets) == train_set_size

        train_meta_loss_ls = []  # for entire epoch
        print_meta_loss_ls = []  # for every print_interval print
        for i in range(num_batch):
            supp_batch = list(supp_sets[config['meta_batch_size'] * i:min(config['meta_batch_size'] * (i + 1), len(supp_sets))])  # tuple, need to convert to list
            supp_xs = [x[:, :-1] for x in supp_batch]  # [[supp_size, movie_sparse_dim + user_sparse_dim], ...]
            supp_ys = [x[:, -1] for x in supp_batch]  # [[supp_size], ...]
            query_batch = list(query_sets[config['meta_batch_size'] * i:min(config['meta_batch_size'] * (i + 1), len(query_sets))])
            query_xs = [x[:, :-1] for x in query_batch]  # [[query_size, movie_sparse_dim + user_sparse_dim], ...]
            query_ys = [x[:, -1] for x in query_batch]  # [[query_size], ...]

            a_size = np.array([len(x) for x in supp_batch], dtype=np.int32)  # [meta_batch_size]
            inputa = np.stack([np.pad(x, ((0, np.max(a_size) - len(x)), (0, 0)), 'constant') for x in supp_xs], axis=0)  # [meta_batch_size, np.max(a_size), sparse_dim]
            labela = np.expand_dims(np.stack([np.pad(y, (0, np.max(a_size) - len(y)), 'constant') for y in supp_ys], axis=0), -1)  # [meta_batch_size, np.max(a_size), 1]
            b_size = np.array([len(x) for x in query_batch], dtype=np.int32)  # [meta_batch_size]
            inputb = np.stack([np.pad(x, ((0, np.max(b_size) - len(x)), (0, 0)), 'constant') for x in query_xs], axis=0)  # [meta_batch_size, np.max(b_size), sparse_dim]
            labelb = np.expand_dims(np.stack([np.pad(y, (0, np.max(b_size) - len(y)), 'constant') for y in query_ys], axis=0), -1)  # [meta_batch_size, np.max(b_size), 1]
            feed_dict = {model.a_size: a_size, model.inputa: inputa, model.labela: labela,
                         model.b_size: b_size, model.inputb: inputb, model.labelb: labelb}

            ops = [model.mean_meta_loss, model.metatrain_op]
            outputs = sess.run(ops, feed_dict)

            print_meta_loss_ls.append(outputs[0])
            train_meta_loss_ls.append(outputs[0])

            if i % config['print_interval'] == 0:
                print_avg_meta_loss = sum(print_meta_loss_ls) / len(print_meta_loss_ls)
                print('[{}] Batch {}: avg_meta_loss {:.4f}'.format(
                    time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)),
                    i,
                    print_avg_meta_loss))
                sys.stdout.flush()
                print_meta_loss_ls = []

        train_avg_meta_loss = sum(train_meta_loss_ls) / len(train_meta_loss_ls)
        print('Epoch {} done! time elapsed {}, avg_meta_loss {:.4f}'.format(
            epoch,
            time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)),
            train_avg_meta_loss))
        sys.stdout.flush()

        maes = []
        for state in ['val_ws', 'val_cs', 'test_ws', 'test_cs']:

            eval_set_size = len(datasets[state])  # number of tasks in evaluate set
            supp_sets, query_sets = list(zip(*datasets[state]))  # [(T1_support, T2_support, ...), (T1_query, T2_query, ...)]
            assert len(supp_sets) == len(query_sets) == eval_set_size

            meta_loss_ls, mae_ls = [], []
            for i in range(eval_set_size):
                inputa = np.expand_dims(supp_sets[i][:, :-1], 0)  # [1, supp_size, movie_sparse_dim + user_sparse_dim]
                labela = np.expand_dims(np.expand_dims(supp_sets[i][:, -1], 0), -1)  # [1, supp_size, 1]
                inputb = np.expand_dims(query_sets[i][:, :-1], 0)  # [1, query_size, movie_sparse_dim + user_sparse_dim]
                labelb = np.expand_dims(np.expand_dims(query_sets[i][:, -1], 0), -1)  # [1, query_size, 1]
                feed_dict = {model.inputa: inputa, model.labela: labela,
                             model.inputb: inputb, model.labelb: labelb}

                ops = [model.eval_meta_loss, model.eval_mae]
                outputs = sess.run(ops, feed_dict)

                meta_loss_ls.append(outputs[0])
                mae_ls.append(outputs[1])

            avg_meta_loss = sum(meta_loss_ls) / len(meta_loss_ls)
            avg_mae = sum(mae_ls) / len(mae_ls)
            maes.append(avg_mae)
            print('Evaluate {} done! time elapsed {}, avg_meta_loss {:.4f}, avg_mae {:.4f}'.format(
                state,
                time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)),
                avg_meta_loss,
                avg_mae))
            sys.stdout.flush()

        overall_val_mae = (maes[0] * len(datasets['val_ws']) + maes[1] * len(datasets['val_cs'])) / (
                len(datasets['val_ws']) + len(datasets['val_cs']))
        overall_test_mae = (maes[2] * len(datasets['test_ws']) + maes[3] * len(datasets['test_cs'])) / (
                len(datasets['test_ws']) + len(datasets['test_cs']))
        print('overall_val_mae {:.4f}, overall_test_mae {:.4f}'.format(overall_val_mae, overall_test_mae))
        print('')
        sys.stdout.flush()

        overall_maes.append(overall_test_mae)
        ws_maes.append(maes[2])
        cs_maes.append(maes[3])

        if epoch % config['save_interval'] == 0:
            ckpt_alias = config['logdir'] + '/' + exp_string + 'Epoch{}_mae{:.4f}_ws{:.4f}_cs{:.4f}'.format(
                epoch,
                overall_test_mae,
                maes[2],
                maes[3]
            )
            saver.save(sess, ckpt_alias)  # save checkpoint


def main(args):

    config = {k: v for (k, v) in vars(args).items()}

    # add dataset-specific config
    if config['data'] == 'movielens_1m':
        config_xtra = config_ml
    elif config['data'] == 'yelp':
        config_xtra = config_yelp
        config['print_interval'] = config['print_interval'] * 5
    elif config['data'] == 'amazon_cds':
        config_xtra = config_amzcd
    else:
        raise ValueError('Unrecognized dataset')

    for k, v in config_xtra.items():
        config[k] = v
    print(config)

    if config['data'] == 'yelp':
        data_path = os.path.join(config['datadir'], config['data'] + '_dataset')
    else:
        data_path = os.path.join(config['datadir'], config['data'])

    # Load datasets
    load_start_time = time.time()

    print('Load datasets start!')
    with open(os.path.join(data_path, 'user_dict.pkl'), mode='rb') as fp:
        user_dict = pickle.load(fp)
    with open(os.path.join(data_path, 'item_dict.pkl'), mode='rb') as fp:
        item_dict = pickle.load(fp)
    train_df = pd.read_csv(os.path.join(data_path, 'train_df.csv'))
    val_df = pd.read_csv(os.path.join(data_path, 'val_df.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'test_df.csv'))
    with open(os.path.join(data_path, 'user_set_dict.pkl'), mode='rb') as fp:
        user_set_dict = pickle.load(fp)
    print('Load datasets done! time elapsed {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - load_start_time))))

    datasets = {}
    for user_set_name in ['train_user_set', 'val_ws_user_set', 'val_cs_user_set', 'test_ws_user_set', 'test_cs_user_set']:
        state_name = user_set_name.split('_user_set')[0]
        datasets[state_name] = []
        print('Load {} start!'.format(user_set_name))
        for u_id in user_set_dict[user_set_name]:
            if 'train' in user_set_name:
                u_ratings = train_df[train_df.user_id == u_id].reset_index(drop=True)  # all samples for the train user. return dataframe
            elif 'val' in user_set_name:
                u_ratings = val_df[val_df.user_id == u_id].reset_index(drop=True)  # all samples for the val user. return dataframe
            else:  # 'test' in user_set_name
                u_ratings = test_df[test_df.user_id == u_id].reset_index(drop=True)  # all samples for the test user. return dataframe

            # preproc support set
            support_set = None  # [support_size, movie_sparse_dim + user_sparse_dim + 1]
            count = 0
            for row in u_ratings[10 - config['support_size']:10].itertuples():
                i_feature = item_dict[row.item_id]
                u_feature = user_dict[row.user_id]
                rating = np.array([row.rating], dtype=np.int8)
                sample = np.expand_dims(np.concatenate((i_feature, u_feature, rating)), 0)  # [1, movie_sparse_dim + user_sparse_dim + 1]
                if count == 0:
                    support_set = sample
                else:
                    support_set = np.concatenate((support_set, sample), 0)
                count += 1

            # preproc query set
            query_set = None  # [remaining, movie_sparse_dim + user_sparse_dim + 1]
            count = 0
            for row in u_ratings[10:].itertuples():
                i_feature = item_dict[row.item_id]
                u_feature = user_dict[row.user_id]
                rating = np.array([row.rating], dtype=np.int8)
                sample = np.expand_dims(np.concatenate((i_feature, u_feature, rating)), 0)  # [1, movie_sparse_dim + user_sparse_dim + 1]
                if count == 0:
                    query_set = sample
                else:
                    query_set = np.concatenate((query_set, sample), 0)
                count += 1

            datasets[state_name].append([support_set, query_set])

        print('Load {} done! time elapsed {}'.format(user_set_name, time.strftime('%H:%M:%S', time.gmtime(time.time() - load_start_time))))

    for k, v in datasets.items():
        print('{}: {}'.format(k, len(v)))

    # configure unique model alias based on hyper-parameters settings
    exp_string = str(config['data'])
    exp_string += '.ss' + str(config['support_size']) + '.mbs' + str(
        config['meta_batch_size']) + '.nstep' + str(config['num_step']) + '.alr' + str(
        config['adapt_lr']) + '.mlr' + str(config['meta_lr']) + '.embed' + str(
        config['embed_dim']) + '.' + str(config['path_or_feat'])
    if config['path_or_feat'] in ['both', 'only_path']:
        if config['path_stop_grad']:
            exp_string += '.psg'
        if config['add_param']:
            exp_string += '.ap'
        if config['add_loss']:
            exp_string += '.al'
        if config['add_grad']:
            exp_string += '.ag'
        if config['add_fisher']:
            exp_string += '.af'
        exp_string += '.' + str(config['path_learner'])
        if config['path_learner'] == 'gru':
            exp_string += '.ghd' + str(config['gru_hidden_dim']) + '.god' + str(config['gru_output_dim'])
        exp_string += '.pnc' + str(config['path_num_cluster'])
        exp_string += '.fnc' + str(config['feat_num_cluster'])
    print(exp_string)
    sys.stdout.flush()

    if not os.path.exists(config['logdir']):
        os.makedirs(config['logdir'])

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        print('Construct graph start!')
        const_start_time = time.time()
        model = CTML(sess, config)
        model.construct_model()
        print('Construct graph done! time elapsed {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - const_start_time))))

        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=50)
        tf.global_variables_initializer().run()

        resume_epoch = 1
        if config['resume']:
            model_file = tf.train.latest_checkpoint(config['logdir'] + '/' + exp_string)
            if model_file:
                resume_epoch = int(model_file.split('_')[-4].split('Epoch')[-1]) + 1
                print("Restoring model weights from " + model_file)
                saver.restore(sess, model_file)

        train(config, model, saver, sess, exp_string, datasets, resume_epoch)


if __name__ == "__main__":

    def str2bool(arg):
        return arg.lower() == 'true'

    parser = argparse.ArgumentParser(description='Clustered Task-Aware Meta-Learning')

    # General Setting
    parser.add_argument('--data', type=str, default='movielens_1m', help='which dataset to use, movielens_1m, yelp or amazon_cds')
    parser.add_argument('--datadir', type=str, default='data', help='datasets directory')
    parser.add_argument('--logdir', type=str, default='ckpts', help='log and checkpoints directory')

    # Base-Learner Setting
    parser.add_argument('--base_embed_dim', type=int, default=8, help='dimension of embedding matrices')
    parser.add_argument('--fcn1_hidden_dim', type=int, default=32, help='hidden size of the first fully-connected layer')
    parser.add_argument('--fcn2_hidden_dim', type=int, default=16, help='hidden size of the second fully-connected layer')

    # Meta-Learner Setting
    parser.add_argument('--embed_dim', type=int, default=20, help='dimension of path & feature embedding')
    parser.add_argument('--path_or_feat', type=str, default='both', help='whether to use path or feature for task representation, '
                                                                         'both, only_path, or only_feat (equivalent to ARML implementation)')
    parser.add_argument('--path_stop_grad', type=str2bool, default=False, help='if True, treat gradients used in path embedding as constant')
    parser.add_argument('--add_param', type=str2bool, default=True, help='if True, include updated parameters for path modeling')
    parser.add_argument('--add_loss', type=str2bool, default=True, help='if True, include losses for path modeling')
    parser.add_argument('--add_grad', type=str2bool, default=True, help='if True, include gradients for path modeling')
    parser.add_argument('--add_fisher', type=str2bool, default=True, help='if True, include fisher information for path modeling')
    parser.add_argument('--path_learner', type=str, default='gru', help='design of path learner, gru or linear')
    parser.add_argument('--gru_hidden_dim', type=int, default=4, help='hidden size for gru path learner')
    parser.add_argument('--gru_output_dim', type=int, default=1, help='output size for gru path learner')
    parser.add_argument('--path_num_cluster', type=int, default=8, help='number of clusters for path embedding')
    parser.add_argument('--feat_num_cluster', type=int, default=8, help='number of clusters for feature embedding')
    parser.add_argument('--stop_grad', type=str2bool, default=True, help='if True, do not use second derivatives in meta-update (for speed)')

    # Meta-Training Setting
    parser.add_argument('--num_step', type=int, default=5, help='number of steps for task adaptation')
    parser.add_argument('--meta_lr', type=float, default=5e-5, help='meta-update learning rate')
    parser.add_argument('--adapt_lr', type=float, default=5e-6, help='task adaptation learning rate')
    parser.add_argument('--meta_batch_size', type=int, default=1, help='number of tasks sampled per meta-update')
    parser.add_argument('--support_size', type=int, default=10, help='support/training set size for task adaptation (i.e., K-shot), can vary from 0 to 10')
    parser.add_argument('--num_epoch', type=int, default=20, help='number of epochs over entire dataset for meta-training')
    parser.add_argument('--resume', type=str2bool, default=True, help='if True, resume training from the latest checkpoint')
    parser.add_argument('--print_interval', type=int, default=600, help='interval (number of iterations/batches) in between printing training output')
    parser.add_argument('--save_interval', type=int, default=1, help='interval (number of epochs) in between saving checkpoint')

    args = parser.parse_args()

    main(args)
