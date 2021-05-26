
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from datetime import datetime
import math
import argparse
import numpy as np

import tensorflow as tf
from gas_train_base_learner import get_gas_dataset
from gas_models import AbstentionModel

from gas_utils import get_model_dir_name
from gas_utils import post_processing_mix_match_one_sided_models_same_lambda_th

# N = total number of data points
# S (+ or -)
# Coverage : #( f>th ) / N
# Accuracy : #( f>th, y==1 ) / N
# Error    : #( f>th, y==-1 ) / N

def eval_test_adversarial(cls, best_loss, Xtst, ytst, model, sess, saver, model_dir, global_step, args):
    print('\nEvaluate adversarial test performance at ({})'.format(datetime.now()))
    eval_batch_size = args.eval_batch_size
    
    #Xtst, ytst = mnist.test.images, mnist.test.labels
    num_eval_examples = len(ytst)
    assert( Xtst.shape[0] == num_eval_examples )

    # Iterate over the samples batch-by-batch
    #assert( num_eval_examples % eval_batch_size == 0 )
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))                 
    aux_acc = 0
    acc = 0
    cov = 0
    loss = 0
    loss_l1 = 0
    loss_l2 = 0
    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = Xtst[bstart:bend, :]
        y_batch_aux = ytst[bstart:bend]
        
        dict_nat = {model.x_input: x_batch, model.y_input_aux: y_batch_aux, 
                    model.is_training:False}
        cur_cov, cur_aux_acc, cur_acc, cur_xent, cur_l1, cur_l2 = sess.run([
            model.mean_binary_cov, 
            model.accuracy_aux,
            model.mean_binary_acc, 
            model.xent, 
            model.binary_prob_xent, 
            model.xent_aux], feed_dict = dict_nat)

        acc += cur_acc
        cov += cur_cov
        aux_acc += cur_aux_acc
        
        loss += cur_xent
        loss_l1 += cur_l1
        loss_l2 += cur_l2

    aux_acc /= num_batches
    acc /= num_batches
    cov /= num_batches
    loss /= num_batches
    loss_l1 /= num_batches
    loss_l2 /= num_batches

    if best_loss > loss : 
        print('\n\nSaving the new trained checkpoint..')
        best_loss = loss
        saver.save(sess, os.path.join(model_dir, 'checkpoint'), global_step=global_step)
    
    #saver.save(sess, os.path.join(model_dir, 'checkpoint'), global_step=global_step)
    print('   test==> aux-accuracy={:.2f}%, accuracy={:.2f}%, coverage={:.4}, loss={:.4}, best-loss={:.4}, binary_prob_xent={:.4}, xent_aux={:.4},'.
          format(100 * aux_acc, 100 * acc, cov, loss, best_loss, loss_l1, loss_l2))
    print('  Finished Evaluating adversarial test performance at ({})'.format(datetime.now()))
    return best_loss
    
def evaluate_one_data_batch(cls, b, B, train_X, train_Y, batch_size, sess, model, best_loss, ii):
    # Output to stdout
    idx = random.randint(0,B-1)
    x_batch = train_X[idx*batch_size: (idx+1)*batch_size]
    y_batch_aux = train_Y[idx*batch_size: (idx+1)*batch_size]

    nat_dict = {model.x_input: x_batch, model.y_input_aux: y_batch_aux,
                model.is_training:False}
    
    cov, aux_acc, acc, xent, l1, l2 = sess.run([
            model.mean_binary_cov, 
            model.accuracy_aux,
            model.mean_binary_acc, 
            model.xent, 
            model.binary_prob_xent, 
            model.xent_aux], feed_dict = nat_dict)
    
    print('  Batch {}({}/{}):    ({})'.format(ii, b, B, datetime.now()))
    print('    training==> aux-accuracy={:.2f}%, accuracy={:.4}%, xent={:.4}, binary_prob_xent={:.4},xent_aux={:.4}, coverage={:.4}'.
          format(aux_acc*100, acc*100,  xent, l1, l2, cov))
    print('    best test loss: {:.2f}'.format(best_loss))



def train_model(train_X, train_Y, val_X, val_Y, test_X, test_Y, cls, 
    model_dir, threshold, _lambda, alpha=0.5, max_num_training_steps=21, lr=1e-3, max_lr=1e-5,
    warm_start=True, backbone=False, args=None):

    n_features = train_X.shape[1]    
    n_classes = len( np.unique(train_Y) )

    print(train_Y.shape)
    print(val_Y.shape)
    print(test_Y.shape)
    
    print('\n\nmodel directory = ', model_dir)
    if not os.path.exists(model_dir): os.makedirs(model_dir)

    tf.reset_default_graph()
    tf.set_random_seed( 1234 )
    np.random.seed( 1111 )
    batch_size = args.batch_size 

    # Setting up the data and the model
    global_step = tf.contrib.framework.get_or_create_global_step()

    p_lr = tf.placeholder(tf.float32, shape=[])
    p_max_lr = tf.placeholder(tf.float32, shape=[])

    model = AbstentionModel( n_features, n_classes, threshold=threshold, mu=_lambda, alpha=alpha)
    max_step = tf.train.AdamOptimizer(p_max_lr).minimize(model.lambda_opt_xent, var_list=model._lambdas)
    #max_step = tf.train.AdamOptimizer(1e-3).minimize(model.lambda_opt_xent, var_list=model._lambdas)
    
    train_step = tf.train.AdamOptimizer(p_lr).minimize(model.xent, global_step=global_step, var_list=model.all_minimization_vars)
    #train_step = tf.train.AdamOptimizer(lr).minimize(model.xent, global_step=global_step, var_list=model.trn_vars)
    
    best_saver = tf.train.Saver(max_to_keep=3, var_list=tf.trainable_variables())
    saver = tf.train.Saver(max_to_keep=3)
    #with open(model_dir + '/config.json', 'w' ) as f: json.dump( config, f)   

    ckpt = tf.train.latest_checkpoint(model_dir)
    print('\n\nrestore model directory = ', model_dir)

    import time
    start_time = time.time()

    N = len(train_X)
    #assert(N % batch_size == 0)
    B = int(N/batch_size)

    backbone_update_freq = 20
    cur_epoch_counter = 0
    early_stop_criterion = 20
    best_loss = +10000.0
    prev_loss = +10000.0
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('\nInitial lambdas = ', sess.run(model._lambdas))
        #assert(1==2)

        if warm_start:
            saver.restore(sess, ckpt)
            #best_saver.restore(sess, restore_ckpt)
            
            best_loss = eval_test_adversarial(cls, best_loss, test_X, test_Y, model, sess, saver, model_dir, global_step, args)
            
            #print('\n\nSaving the new trained checkpoint..')
            #saver.save(sess, os.path.join(model_dir, 'checkpoint'), global_step=global_step)

        best_loss = eval_test_adversarial(cls, best_loss, test_X, test_Y, model, sess, saver, model_dir, global_step, args)

        for ii in range(max_num_training_steps):            
            p = np.random.permutation(len(train_Y))
            pX, pY = train_X[p], train_Y[p]

            if ii in [ max_num_training_steps//4, max_num_training_steps//2, 3*max_num_training_steps//4 ]:
                max_lr /= 2.
                lr /= 2.

            print('Epoch = ', ii, ' -- lr=', lr, ' -- max_lr=', max_lr)
            for b in range(B): 
                #x_batch = train_X[ b*batch_size : (b+1)*batch_size ]
                #y_batch_aux = train_Y[ b*batch_size : (b+1)*batch_size ]
                x_batch = pX[ b*batch_size : (b+1)*batch_size ]
                y_batch_aux = pY[ b*batch_size : (b+1)*batch_size ]

                #print( 'unique= ', np.unique(y_batch_aux) )

                nat_dict = {model.x_input: x_batch, model.y_input_aux: y_batch_aux, 
                            model.is_training:True, p_lr:lr, p_max_lr:max_lr }
                sess.run(train_step, feed_dict=nat_dict)                
                sess.run(max_step, feed_dict=nat_dict)                
                
                #if (backbone == False) and (b%backbone_update_freq == 0):
                #    sess.run(backbone_train_step, feed_dict=nat_dict)                
                
                #if (b % (B-1) == 0):
                #    evaluate_one_data_batch(cls, b, B, train_X, train_Y, batch_size, sess, model, best_loss, ii)

            #print('\n\nEvaluate adversarial accuracy on test data..', ii)
            prev_loss = best_loss
            best_loss = eval_test_adversarial(cls, best_loss, test_X, test_Y, model, sess, saver, model_dir, global_step, args)

            #print('\nlambdas = ', sess.run(model._lambdas))
            #print('\nepsilons = ', sess.run(model._epsilons))
            
            if prev_loss == best_loss:
                cur_epoch_counter += 1 
                if cur_epoch_counter >= early_stop_criterion:
                    print('\nExiting early..')
                    break
            else:
                cur_epoch_counter = 0

        #assert(1 == 2)

    print('took ', int((time.time() - start_time)), 's')


def train_learning_with_abstention(train_X, train_Y, val_X, val_Y, test_X, test_Y, lambdas = [1.0], threshold = 0.5, max_num_training_steps=21,
            lr=1e-4, max_lr=1e-5, backbone=False, warm_start=True, alpha=0.99, args=None):
    print('\n\n Training multiple one sided models...')
    print('mus = ', lambdas)
    
    cls=1
    for _lambda in lambdas:                       
        model_dir = get_model_dir_name(cls, _lambda, alpha, backbone=False, args=args)
        
        train_model(train_X, train_Y, val_X, val_Y, test_X, test_Y, cls,
            model_dir, threshold, _lambda, alpha=alpha, max_num_training_steps=max_num_training_steps, lr=lr,
            max_lr=max_lr, backbone=backbone, warm_start=warm_start, args=args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Abstention training Codebase')
    parser.add_argument('-d', '--data', default='./data/GAS/', type=str, metavar='DIR', help='path to dataset')
    parser.add_argument('-md', '--model_dir', default='./models/abstention/trn_scratch', type=str, help='store trained models here')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('-eb', '--eval_batch_size', default=128, type=int, help='eval batch size')
    parser.add_argument('-e', '--epochs', default=400, type=int, help='Epochs')
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('-wd', '--weight_decay', default=0.1, type=float, help='weight decay')
    args = parser.parse_args()
    print('args = ', args)

    trn_X, trn_y, tst_X, tst_y = get_gas_dataset( args.data )

    fp_name = os.path.join( args.data, 'trn_pred_wk_' + str(False) + '.npy' )
    sc_trn_y = np.load( fp_name )

    fp_name = os.path.join( args.data, 'tst_pred_wk_' + str(False) + '.npy' )
    sc_tst_y = np.load( fp_name )
    print('  ---     trn acc = ', np.mean( sc_trn_y == trn_y ))
    print('  ---     tst acc = ', np.mean( sc_tst_y == tst_y ))

    n_features = trn_X.shape[-1]
    trn_X = trn_X[:, :n_features//2]
    tst_X = tst_X[:, :n_features//2]
    n_features = trn_X.shape[-1]

    n_classes = len( np.unique(trn_y) )
    print('n_features = ', n_features)
    print('n_classes = ', n_classes)
    
    #mus = [1.]
    #mus = [0.49, 0.98, 1.67, 1.96, 2.5]
    mus = np.linspace(0.1,3,30)
    print(mus)

    #thresholds = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9];
    thresholds =np.linspace(0, 1, num=100)
    print(thresholds)

    alpha= 0.999999 #0.999 #0.99
    #desired_errors = [0.005, 0.01, 0.02, 0.10];
    desired_errors = [ 0.1, 0.2 ];

    lr = 1e-3
    max_lr = 1e-5
    #lr = 1e-4
    #max_lr = 1e-6

    #train_learning_with_abstention(trn_X, trn_y, tst_X, tst_y, tst_X, tst_y, lambdas = mus, threshold = 0.5, max_num_training_steps=args.epochs,
    train_learning_with_abstention(trn_X, sc_trn_y, tst_X, sc_tst_y, tst_X, sc_tst_y, lambdas = mus, threshold = 0.5, max_num_training_steps=args.epochs,
            lr=lr, max_lr=max_lr, warm_start=False, alpha=alpha, args=args)

    x = post_processing_mix_match_one_sided_models_same_lambda_th(tst_X, sc_tst_y, tst_X, sc_tst_y, lambdas = mus, thresholds = thresholds, 
                          desired_errors = desired_errors, alpha=alpha, args=args)
