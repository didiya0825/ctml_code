from __future__ import print_function
from __future__ import division

import random
import sys
import os
import argparse
import numpy as np
import time
import tensorflow as tf

from modules.data_generator import DataGenerator
from modules.ctml import CTML

np.random.seed(123)
random.seed(5)
tf.set_random_seed(1234)


def train(config, model, saver, sess, exp_string, data_generator, resume_itr=0):

    if resume_itr == 0:
        saver.save(sess, config['logdir'] + '/' + exp_string + '/model' + str(resume_itr))

    print('Done initializing, starting training.')
    sys.stdout.flush()

    prelosses, postlosses, preaccs, postaccs, reconlosses = [], [], [], [], []  # for print, reset after print

    num_class = data_generator.num_class

    for itr in range(resume_itr, config['num_metatrain_iter']):
        if config['data'] == 'meta_dataset':
            image_batch, label_batch = data_generator.make_input_meta_dataset(train=True)  # np.array [4, no. of samples per task, 21168], np.array [4, no. of samples per task]
        else:
            image_batch, label_batch = data_generator.make_input_miniimagenet(train=True)
        inputa = image_batch[:, :num_class * config['support_size'], :]  # [4, support_size, 21168]
        inputb = image_batch[:, num_class * config['support_size']:, :]  # [4, query_size, 21168]
        labela = label_batch[:, :num_class * config['support_size']]  # [4, support_size]
        labelb = label_batch[:, num_class * config['support_size']:]  # [4, query_size]
        feed_dict = {model.inputa: inputa, model.labela: labela,
                     model.inputb: inputb, model.labelb: labelb}

        ops = [model.metatrain_op,
               model.total_loss1, model.total_losses2[config['num_step'] - 1],
               model.total_accuracy1, model.total_accuracies2[config['num_step'] - 1]]
        if config['shortcut']:
            ops.append(model.shortcut_loss)
        outputs = sess.run(ops, feed_dict)

        if not np.isnan(outputs[1]) and not np.isnan(outputs[2]):
            prelosses.append(outputs[1])
            postlosses.append(outputs[2])

        if not np.isnan(outputs[3]) and not np.isnan(outputs[4]):
            preaccs.append(outputs[3])
            postaccs.append(outputs[4])

        if config['shortcut']:
            if not np.isnan(outputs[-1]):
                reconlosses.append(outputs[-1])

        if (itr != 0) and itr % config['print_interval'] == 0:
            print_str = 'Iteration {}'.format(itr)
            std = np.std(postlosses, 0)
            ci95 = 1.96 * std / np.sqrt(config['print_interval'])
            print_str += ': preloss: ' + str(np.mean(prelosses)) + ', postloss: ' + str(np.mean(postlosses)) + \
                         ', confidence: ' + str(ci95) + ', preacc: ' + str(np.mean(preaccs)) + ', postacc: ' + str(np.mean(postaccs))
            if config['shortcut']:
                print_str += ', reconloss: ' + str(np.mean(reconlosses))
            print(print_str)
            sys.stdout.flush()
            prelosses, postlosses, preaccs, postaccs, reconlosses = [], [], [], [], []

        if (itr != 0) and itr % config['save_interval'] == 0:
            saver.save(sess, config['logdir'] + '/' + exp_string + '/model' + str(itr))  # save checkpoint

    saver.save(sess, config['logdir'] + '/' + exp_string + '/model' + str(itr))


def test(config, model, sess, data_generator):

    print('Done initializing, starting testing.')
    if config['use_shortcut_approx']:
        print('Apply shortcut approximation.')
    sys.stdout.flush()

    np.random.seed(1)
    random.seed(1)

    accuracies = []

    num_class = data_generator.num_class

    time_ls = []
    for i in range(config['num_test_task']):

        start_time = time.time()

        if config['data'] == 'meta_dataset':
            image_batch, label_batch = data_generator.make_input_meta_dataset(train=False)  # np.array [4, no. of samples per task, 21168], np.array [4, no. of samples per task]
        else:
            image_batch, label_batch = data_generator.make_input_miniimagenet(train=False)  # np.array [4, no. of samples per task, 21168], np.array [4, no. of samples per task]
        inputa = image_batch[:, :num_class * config['support_size'], :]  # [1, support_size, 21168]
        inputb = image_batch[:, num_class * config['support_size']:, :]  # [1, query_size, 21168]
        labela = label_batch[:, :num_class * config['support_size']]  # [1, support_size]
        labelb = label_batch[:, num_class * config['support_size']:]  # [1, query_size]
        feed_dict = {model.inputa: inputa, model.labela: labela,
                     model.inputb: inputb, model.labelb: labelb,
                     model.meta_lr: 0.0}

        ops = [model.total_accuracy1] + model.total_accuracies2
        outputs = sess.run(ops, feed_dict)

        time_elapsed = int((time.time() - start_time) * 1000.0)
        time_ls.append(time_elapsed)
        accuracies.append(outputs)

    mean_time = np.mean(np.array(time_ls), 0)
    accuracies = np.array(accuracies)
    means = np.mean(accuracies, 0)
    stds = np.std(accuracies, 0)
    ci95 = 1.96 * stds / np.sqrt(config['num_test_task'])

    print('Mean test/validation accuracy, stddev, and 95 confidence interval')
    print((means, stds, ci95))
    print('average time: {}'.format(mean_time))
    sys.stdout.flush()


def main(args):

    config = {k: v for (k, v) in vars(args).items()}
    if config['eval']:
        config['train'] = False
    else:
        config['train'] = True
    print(config)

    sess = tf.InteractiveSession()

    # number of adaptation steps may be different for meta-training and meta-testing
    if config['train']:
        test_num_step = config['num_step']
    else:
        test_num_step = config['test_num_step']

    # set the meta_batch_size to 1 for meta-testing
    if not config['train']:
        orig_meta_batch_size = config['meta_batch_size']
        config['meta_batch_size'] = 1

    # for meta-testing, set the query_size = support_size
    if config['train']:
        data_generator = DataGenerator(config, config['support_size'] + config['query_size'], config['meta_batch_size'])
    else:
        data_generator = DataGenerator(config, config['support_size'] * 2, config['meta_batch_size'])

    dim_input = data_generator.dim_input
    dim_output = data_generator.dim_output

    model = CTML(sess, config, test_num_step, dim_input, dim_output)
    model.construct_model(train=config['train'])
    tf.global_variables_initializer().run()
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=60)

    # resume meta_batch_size back to orig_meta_batch_size to restore the correct model
    if not config['train']:
        config['meta_batch_size'] = orig_meta_batch_size

    # configure unique model alias based on hyper-parameters settings
    exp_string = str(config['data'])
    exp_string += '.ss' + str(config['support_size']) + '.qs' + str(config['query_size']) + '.mbs' + str(
        config['meta_batch_size']) + '.nstep' + str(config['num_step']) + '.alr' + str(
        config['adapt_lr']) + '.mlr' + str(config['meta_lr']) + '.embed' + str(
        config['embed_dim']) + '.' + str(config['similarity_kernel']) + '.' + str(config['path_or_feat'])
    if 'path' in config['path_or_feat']:
        exp_string += '.rnstep' + str(config['rehearse_num_step'])
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
        elif config['path_learner'] == 'fc':
            exp_string += '.fnl' + str(config['fc_num_layer'])
        elif config['path_learner'] == 'attention':
            exp_string += '.anl' + str(config['att_num_layer']) + '.ahd' + str(config['att_hidden_dim'])
        exp_string += '.pnc' + str(config['path_num_cluster'])
    if 'feat' in config['path_or_feat']:
        exp_string += '.fnc' + str(config['feat_num_cluster'])
    if config['shortcut']:
        exp_string += '.' + str(config['recon_loss_func']) + str(config['recon_loss_weight'])
        exp_string += '.snl' + str(config['shortcut_num_layer'])
    print(exp_string)
    sys.stdout.flush()

    if not os.path.exists(config['logdir']):
        os.makedirs(config['logdir'])

    resume_itr = 0
    if config['resume'] and config['train']:
        model_file = tf.train.latest_checkpoint(config['logdir'] + '/' + exp_string)  # return the full path to the latest checkpoint or `None` if no checkpoint was found.
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1 + 5:]) + 1  # start from which iteration
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    if config['train']:
        train(config, model, saver, sess, exp_string, data_generator, resume_itr)
    else:
        for test_iter in config['test_iters']:
            model_file = '{0}/{2}/model{1}'.format(config['logdir'], test_iter, exp_string)
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)
            test(config, model, sess, data_generator)


if __name__ == "__main__":

    def str2bool(arg):
        return arg.lower() == 'true'

    parser = argparse.ArgumentParser(description='Clustered Task-Aware Meta-Learning')

    # General Setting
    parser.add_argument('--data', type=str, default='meta_dataset', help="which dataset to use, 'meta_dataset' or 'miniimagenet'")
    parser.add_argument('--num_class', type=int, default=5, help='number of classes per task (i.e., N-way)')
    parser.add_argument('--eval', action='store_true', help='whether to enter evaluation mode')
    parser.add_argument('--datadir', type=str, default='data', help='datasets directory')
    parser.add_argument('--logdir', type=str, default='ckpts', help='log and checkpoints directory')

    # Meta-Learner Setting
    parser.add_argument('--stop_grad', type=str2bool, default=False, help='if True, do not use second derivatives in meta-update (for speed)')
    parser.add_argument('--path_or_feat', type=str, default='path_and_feat', help="whether to use path or feature for task representation, "
                                                                                  "'path', 'feat' or 'path_and_feat'")
    # # step-wise components
    parser.add_argument('--path_stop_grad', type=str2bool, default=False, help='if True, treat gradients used in path embedding as constant')
    parser.add_argument('--add_param', type=str2bool, default=True, help='if True, include updated parameters for path modeling')
    parser.add_argument('--add_loss', type=str2bool, default=True, help='if True, include losses for path modeling')
    parser.add_argument('--add_grad', type=str2bool, default=True, help='if True, include gradients for path modeling')
    parser.add_argument('--add_fisher', type=str2bool, default=True, help='if True, include fisher information for path modeling')
    # # path learner
    parser.add_argument('--rehearse_num_step', type=int, default=5, help='number of steps for rehearsed task learning')
    parser.add_argument('--path_learner', type=str, default='gru', help="design of path learner, 'gru', 'linear', 'fc' or 'attention'")
    parser.add_argument('--gru_hidden_dim', type=int, default=4, help='only for gru path learner, hidden state dimension')
    parser.add_argument('--gru_output_dim', type=int, default=1, help='only for gru path learner, output dimension')
    parser.add_argument('--fc_num_layer', type=int, default=3, help='only for fc path learner, number of layers')
    parser.add_argument('--att_num_layer', type=int, default=3, help='only for attention path learner, number of layers')
    parser.add_argument('--att_hidden_dim', type=int, default=4, help='only for attention path learner, attention hidden dimension d_a')
    # # clustering
    parser.add_argument('--embed_dim', type=int, default=128, help='dimension of path & feature embedding')
    parser.add_argument('--similarity_kernel', type=str, default='student_t', help="similarity kernel used to compare with centroids, 'student_t' or 'dot_product'")
    parser.add_argument('--path_num_cluster', type=int, default=6, help='number of clusters for path embedding')
    parser.add_argument('--feat_num_cluster', type=int, default=6, help='number of clusters for feature embedding')
    # # shortcut tunnel
    parser.add_argument('--shortcut', type=str2bool, default=True, help='if True, construct and train the shortcut tunnel')
    parser.add_argument('--recon_loss_func', type=str, default='js_div', help="reconstruction loss function, 'js_div' or 'mse'")
    parser.add_argument('--recon_loss_weight', type=float, default=1.0, help='reconstruction loss weight')
    parser.add_argument('--shortcut_num_layer', type=int, default=2, help='number of layers for the shortcut tunnel')

    # Meta-Training Setting
    parser.add_argument('--num_step', type=int, default=5, help='number of steps for actual task learning')
    parser.add_argument('--meta_lr', type=float, default=0.001, help='meta-update learning rate')
    parser.add_argument('--adapt_lr', type=float, default=0.01, help='task adaptation learning rate')
    parser.add_argument('--meta_batch_size', type=int, default=4, help='number of tasks sampled per meta-update')
    parser.add_argument('--support_size', type=int, default=5, help='support/training set size for task adaptation (i.e., K-shot)')
    parser.add_argument('--query_size', type=int, default=15, help='query/testing set size for meta-update')
    parser.add_argument('--num_metatrain_iter', type=int, default=60000, help='number of meta-training iterations')
    parser.add_argument('--resume', type=str2bool, default=True, help='if True, resume training from the latest checkpoint')
    parser.add_argument('--print_interval', type=int, default=10, help='interval (number of iterations) in between printing training output')
    parser.add_argument('--save_interval', type=int, default=1000, help='interval (number of iterations) in between saving checkpoint')

    # Meta-Testing Setting
    parser.add_argument('--use_shortcut_approx', type=str2bool, default=False, help='if True, use shortcut approximation for meta-testing, '
                                                                                    'applicable only when shortcut is trained')
    parser.add_argument('--test_num_step', type=int, default=10, help='number of steps for task adaptation during meta-testing')
    parser.add_argument('--test_dataset', type=int, default=-1, help='the sub_dataset to be tested (for meta_dataset only), '
                                                                     '-1: test all, 0: Bird, 1: Aircraft, 2: Fungi, 3: Flower, 4: Texture, 5: Traffic Sign')
    parser.add_argument('--num_test_task', type=int, default=1000, help='number of tasks sampled for meta-testing')
    parser.add_argument('--test_iters', type=int, default=[59000, 58000, 57000], nargs='+', help='the meta-training iterations to be tested')
    parser.add_argument('--eval_test', type=str2bool, default=True, help='if True, evaluate on the the test set, else evaluate on the validation set')

    args = parser.parse_args()

    main(args)
