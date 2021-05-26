
import os
from datetime import datetime
import math
import argparse
import numpy as np

import tensorflow as tf
from gas_train_base_learner import get_gas_dataset
from gas_models import AbstentionModel

def get_model_dir_name(cls, mu, alpha, backbone=False, args=None):
    if backbone:
        model_dir = args.model_dir + '_backbone_one_sided_formulation(a='+ str(alpha)+',cls=' + str(cls) + ',mu=' + str(mu) + ')'
    else:
        model_dir = args.model_dir + '_one_sided_formulation(a='+ str(alpha)+',cls=' + str(cls) + ',mu=' + str(mu) + ')'
    return model_dir

def run_model_on_data(Xtst, ytst, cls, _th, mu, alpha, args=None):
    eval_batch_size = len(ytst) #config['eval_batch_size']
    n_classes = 6
    
    num_eval_examples = len(ytst)
    assert( Xtst.shape[0] == num_eval_examples )

    # Iterate over the samples batch-by-batch
    assert( num_eval_examples % eval_batch_size == 0 )
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size)) 
    print('num_batches = ', num_batches)
    
    y_scores = np.zeros( (num_eval_examples, n_classes), dtype=np.float32 )
    #print('\n\ncls={}, _lambda={}, _threshold={}'.format(cls, _lambda, _th))
    #print('\nEvaluate adversarial test performance at ({})'.format(datetime.now()))
    
    # Load the graph 
    model_dir = get_model_dir_name(cls, mu, alpha, args=args)
    tf.reset_default_graph()
    tf.set_random_seed(1234)
    np.random.seed(1111)

    n_features = Xtst.shape[1]    
    n_classes = len( np.unique(ytst) )
    model = AbstentionModel( n_features, n_classes, threshold=_th, mu=mu, alpha=alpha)

    print('model dir = ', model_dir)
    ckpt = tf.train.latest_checkpoint(model_dir)
    saver = tf.train.Saver(max_to_keep=3)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt)

        acc = 0
        cov = 0
        for ibatch in range(num_batches):
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)

            x_batch = Xtst[bstart:bend, :]
            y_batch = ytst[bstart:bend]
            dict_nat = {model.x_input: x_batch, model.y_input_aux: y_batch, model.is_training:False}

            raw_sigmoid_scores, cur_cov, cur_acc = sess.run([model.softmax_out, 
                model.mean_binary_cov, model.mean_binary_acc], feed_dict = dict_nat)
            
            print('raw_sigmoid_scores = ', raw_sigmoid_scores.shape, np.unique( raw_sigmoid_scores[:,0] )  )
            y_scores[ bstart:bend ] = raw_sigmoid_scores

            acc += cur_acc
            cov += cur_cov

        acc /= num_batches
        cov /= num_batches

        #print('    test accuracy={:.2f}%, coverage={:.4}'.format(100 * acc, cov))
        #print('  Finished Evaluating adversarial test performance at ({})'.format(datetime.now()))
    return y_scores


def gather_all_predictions(val_X, val_Y, test_X, test_Y, alpha, lambdas, args=None):
    # Gather all predictions
    #n_classes = len( np.unique(test_X) )
    #classes = list(range(0,n_classes))
    classes = list(range(0,6))

    _predictions = {}
    for cls in classes:
        _predictions[cls] = {}
       
    for _lambda in lambdas:
        scores = run_model_on_data(val_X, val_Y, 1, 0.5, _lambda, alpha, args=args)
        #scores = run_model_on_data(test_X, test_Y, 1, 0.5, _lambda, alpha)
        for cls in classes:
            _predictions[cls][_lambda] = scores[:,cls]
                            
    _test_predictions = {}
    for cls in classes:
        _test_predictions[cls] = {}
        
    for _lambda in lambdas:
        scores = run_model_on_data(test_X, test_Y, 1, 0.5, _lambda, alpha, args=args)
        for cls in classes:
            _test_predictions[cls][_lambda] = scores[:,cls]
            
    #get_predictions_normalized(lambdas, _predictions)
    #get_predictions_normalized(lambdas, _test_predictions)
    return _predictions, _test_predictions

def get_coverage_error_for_given_parameters_pred_max( _predictions, lambdas, thresholds, cur_params, y_true ):
    classes = list(range(0,6))

    n_examples = len(y_true)    
    belongs_to = np.zeros( (n_examples, ), dtype=np.int32 )
    max_score = np.zeros( (n_examples, ), dtype=np.float32 )
    max_class = np.zeros( (n_examples, ), dtype=np.int32 )
    
    _lambda_idx, _threshold_idx = cur_params
    y_pred = np.zeros( (n_examples, ), dtype=np.int32 )
    
    for i in range(n_examples):
        max_score = -1000.0
        max_class = -1
        for cls in classes:
            _lambda = lambdas[ _lambda_idx[cls] ]
            _threshold = thresholds[ _threshold_idx[cls] ]
            if (_predictions[cls][_lambda][i] >= _threshold):
                if max_score < _predictions[cls][_lambda][i]:
                    max_score = _predictions[cls][_lambda][i]
                    max_class =  cls
        
        if max_class == -1:
            belongs_to[i] = 0 #every OSC rejected this example
        else:
            belongs_to[i] = 1
            y_pred[i] = max_class
    '''
    for cls in classes:
        _lambda = lambdas[ _lambda_idx[cls] ]
        _threshold = thresholds[ _threshold_idx[cls] ]
        cur_class_pos = (_predictions[cls][_lambda] >= _threshold) * 1
        
        belongs_to += cur_class_pos
        
        # set the class to be current cls wherever cur_class_pos was true
        y_pred[ cur_class_pos==1 ] = cls
    '''
        
    #TODO
    #figure out which ones are rejected (if no one says it belongs to them, or if more than one says it belongs to them)
    
    n_rejections = np.sum( belongs_to != 1 )
    coverage = 1.0 - (n_rejections / n_examples)
    
    accuracy = np.sum( ( belongs_to == 1 ) * (y_true == y_pred) ) / n_examples
    error = coverage - accuracy
    
    return error, coverage



def post_processing_mix_match_one_sided_models_same_lambda_th(val_X, val_Y, test_X, test_Y, lambdas = [1.0], thresholds = [0.5],
        desired_errors = [0.01, 0.02], alpha=0.5, args=None):
    print('\n\n Mixing multiple one sided models...')
    
    classes = list(range(0,6))
    lambdas = sorted(lambdas)
    thresholds = sorted(thresholds)
    print('lambdas = ', lambdas)
    #print('thresholds = ', thresholds)
    
    _predictions, _test_predictions = gather_all_predictions( val_X, val_Y, test_X, test_Y, alpha, lambdas, args=args)
                
    # Will mix-match now on the validation set
    print('\n\nResults = ')
    
    
    #- [DONE] Sort lambdas, thresholds
    #- [DONE] Pick initial set of parameters (lambda_1, ..., lambda_10, threshold_1, ..., threshold_10)
    #- [DONE] Find out the performance for this set of params (coverage, error)
    #- [DONE] Navigate to its one neighbours and find out their performance, pick the one with the highest coverage for given error
    #- Do this randomized start couple of times
    
    #thresholds = np.unique(_predictions[classes[0]][lambdas[-1]])[::10]
    print('thresholds = ', thresholds)
    
    n_lambdas    = len(lambdas)
    n_thresholds = len(thresholds)
    _lambda_idx  = np.random.randint( n_lambdas, size=10 )
    _threshold_idx = np.random.randint( n_thresholds, size=10 )
    
    y = val_Y
    #y = test_Y
    
    for error in desired_errors:
        best_coverage = 0.0
        best_params   = None
        
        for _lidx in range(n_lambdas):
            for _tidx in range(n_thresholds):
                _lambda_idx[:] = _lidx
                _threshold_idx[:] = _tidx
        
                cur_params = (_lambda_idx, _threshold_idx)
            
                cur_error, cur_coverage = get_coverage_error_for_given_parameters_pred_max( _predictions, lambdas, thresholds, cur_params, y )
                #cur_error, cur_coverage = get_coverage_error_for_given_parameters( _predictions, lambdas, thresholds, cur_params, y )
                #print('cur_error=', cur_error, ' --> cur_coverage=', cur_coverage)
                if (cur_error <= error) and (cur_coverage > best_coverage):
                    #print('cur_error=', cur_error, ' --> better  cur_coverage=', cur_coverage, ' parmas=', cur_params)
                    best_coverage, best_params = cur_coverage, deepcopy(cur_params)
        
        if best_params is not None:
            # Lets evaluate the performance on test data with these parameters
            _lambda_idx, _threshold_idx = best_params
            #test_error, test_coverage = get_coverage_error_for_given_parameters( _test_predictions, lambdas, thresholds, best_params, test_Y )
            test_error, test_coverage = get_coverage_error_for_given_parameters_pred_max( _test_predictions, lambdas, thresholds, best_params, test_Y )

            print('\n\nFor desired_error=', error, " -> best coverage=", best_coverage, ' with params=', best_params)
            print('For desired_error={:.4}, ==> test cov={:.4}, err=={:.4}'.format(error, test_coverage, test_error) )
        else:
            print('For desired_error={:.4}, could not find any parameters'.format(error))
        

