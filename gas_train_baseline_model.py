
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from datetime import datetime
import math
import argparse
import numpy as np

import tensorflow as tf
from gas_train_base_learner import get_gas_dataset
from gas_models import BaselineModel


def get_predictions( Xtst, ytst, eval_batch_size, sess, model ):
    num_eval_examples = len(ytst)
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))                 
    tst_pred = np.empty_like(ytst)
    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = Xtst[bstart:bend, :]
        y_batch_aux = ytst[bstart:bend]
        
        dict_nat = {model.x_input: x_batch, model.y_input_aux: y_batch_aux, model.is_training:False}
        cur_aux_acc, cur_xent_aux, cur_pred = sess.run([
            model.accuracy_aux,
            model.xent_aux, model.predictions_aux], feed_dict = dict_nat)

        tst_pred[bstart:bend] = cur_pred
    return tst_pred


def eval_test( best_acc, Xtst, ytst, model, sess, saver, model_dir, global_step, args, Xtrn, ytrn ):
    num_eval_examples = len(ytst)
    assert( Xtst.shape[0] == num_eval_examples )
    #tst_pred = np.empty_like(ytst)

    eval_batch_size = args.eval_batch_size
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))                 
    aux_acc = 0
    loss = 0
    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = Xtst[bstart:bend, :]
        y_batch_aux = ytst[bstart:bend]
        
        dict_nat = {model.x_input: x_batch, model.y_input_aux: y_batch_aux, model.is_training:False}
        cur_aux_acc, cur_xent_aux = sess.run([
            model.accuracy_aux,
            model.xent_aux], feed_dict = dict_nat)

        #tst_pred[bstart:bend] = cur_pred

        aux_acc += cur_aux_acc
        loss += cur_xent_aux

    aux_acc /= num_batches
    loss /= num_batches

    #if best_loss > loss : 
    if best_acc < aux_acc:
        print('\n\nSaving the new trained checkpoint..')
        best_acc = aux_acc
        saver.save(sess, os.path.join(model_dir, 'checkpoint'), global_step=global_step)

        tst_pred = get_predictions( Xtst, ytst, eval_batch_size, sess, model )   
        print('  ---     acc = ', np.mean( tst_pred == ytst ))

        fp_name = os.path.join( args.data, 'tst_pred_wk_' + str(args.weak_baseline) + '.npy' )
        print('\n\nSaving the new predictions at..', fp_name)
        np.save( fp_name, tst_pred )

        trn_pred = get_predictions( Xtrn, ytrn, eval_batch_size, sess, model )   
        print('  ---     trn acc = ', np.mean( trn_pred == ytrn ))

        fp_name = os.path.join( args.data, 'trn_pred_wk_' + str(args.weak_baseline) + '.npy' )
        print('\n\nSaving the new predictions at..', fp_name)
        np.save( fp_name, trn_pred )


    print('   test==> aux-accuracy={:.2f}%, loss={:.4}, best-loss={:.2f}% '.
          format(100 * aux_acc, loss, 100*best_acc))
    print('  Finished Evaluating adversarial test performance at ({})'.format(datetime.now()))
    return best_acc #, tst_pred, aux_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Online LWA Codebase')
    parser.add_argument('-wk', '--weak_baseline', default=False, action='store_true')
    parser.add_argument('-d', '--data', default='./data/GAS/', type=str, metavar='DIR', help='path to dataset')
    parser.add_argument('-md', '--model_dir', default='./models/GAS-baseline/', type=str, help='store trained models here')
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('-eb', '--eval_batch_size', default=128, type=int, help='eval batch size')
    parser.add_argument('-e', '--epochs', default=30, type=int, help='Epochs')
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('-wd', '--weight_decay', default=0.1, type=float, help='weight decay')
    args = parser.parse_args()
    print('args = ', args)

    trn_X, trn_y, tst_X, tst_y = get_gas_dataset( args.data )
    if args.weak_baseline:
        n_features = trn_X.shape[-1]
        trn_X = trn_X[:, :n_features//2]
        tst_X = tst_X[:, :n_features//2]
    n_features = trn_X.shape[-1]
    n_classes = len( np.unique(trn_y) )
    print('n_features = ', n_features)
    print('n_classes = ', n_classes)

    tf.reset_default_graph()
    tf.set_random_seed(1234)
    np.random.seed(1111)
    batch_size = args.batch_size

    global_step = tf.contrib.framework.get_or_create_global_step()
    model = BaselineModel( n_features, n_classes, args.weight_decay )
    #train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(model.xent_aux, global_step=global_step, var_list=model.trn_vars)
    #train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(model.xent_aux + model.l2_loss_aux, global_step=global_step, var_list=model.trn_vars)
    p_lr = tf.placeholder(tf.float32, shape=[])
    lr = args.learning_rate
    train_step = tf.train.AdamOptimizer(p_lr).minimize(model.xent_aux + model.l2_loss_aux, global_step=global_step, var_list=model.all_minimization_vars)

    best_saver = tf.train.Saver(max_to_keep=3, var_list=tf.trainable_variables())
    saver = tf.train.Saver(max_to_keep=3)

    model_dir = args.model_dir
    if args.weak_baseline:
        model_dir = os.path.join( model_dir, '-weak' )

    print('\n\nmodel directory = ', model_dir)
    if not os.path.exists(model_dir): os.makedirs(model_dir)

    ckpt = tf.train.latest_checkpoint(model_dir)
    print('\n\nrestore model directory = ', model_dir)

    N = len(trn_X)
    B = N // batch_size

    best_acc = 0.0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        best_acc = eval_test(best_acc, tst_X, tst_y, model, sess, saver, model_dir, global_step, args, trn_X, trn_y)

        for epoch in range(args.epochs):

            if epoch in [ args.epochs//4, args.epochs//2, 3*args.epochs//4 ]:
                lr /= 2.

            # Shuffle dataset
            p = np.random.permutation(len(trn_y))
            pX, pY = trn_X[p], trn_y[p]

            for b in range(B): 
                x_batch = pX[ b*batch_size : (b+1)*batch_size ]
                y_batch_aux = pY[ b*batch_size : (b+1)*batch_size ]

                nat_dict = {model.x_input: x_batch, 
                            model.y_input_aux: y_batch_aux, 
                            model.is_training:True, p_lr:lr, p_lr:lr  }
                sess.run(train_step, feed_dict=nat_dict)      

            best_acc = eval_test(best_acc, tst_X, tst_y, model, sess, saver, model_dir, global_step, args, trn_X, trn_y)


