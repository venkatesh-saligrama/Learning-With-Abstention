import math
import numpy as np
import tensorflow as tf
from copy import deepcopy
import config

from resnet import ResnetModel

classes = list(range(0,10))
config = config.get_config()

def get_model_dir_name(cls, mu, alpha, backbone=False):
    if backbone:
        model_dir = config['model_dir'] + '_backbone_one_sided_formulation(a='+ str(alpha)+',cls=' + str(cls) + ',mu=' + str(mu) + ')'
    else:
        model_dir = config['model_dir'] + '_one_sided_formulation(a='+ str(alpha)+',cls=' + str(cls) + ',mu=' + str(mu) + ')'
    return model_dir

def get_coverage_error_accuracy_for_model_pairs_conditional(y, y0, y1):
    # Given two model predictions, find out coverage, error, accuracy
    
    n_examples   = len(y)
    n_rejections = np.sum( y0 == y1 )
    n_class_zero = np.sum( (y0==1) * (y1==0) ) 
    n_class_one  = np.sum( (y0==0) * (y1==1) ) 
    
    print('n_rejections = ', n_rejections)
    print('n_class_zero = ', n_class_zero)
    print('n_class_one = ', n_class_one)
    assert((n_rejections + n_class_zero + n_class_one) == n_examples)
    
    coverage = 1.0 - (n_rejections / n_examples)
    
    n_correct = np.sum( y[(y0==1) * (y1==0)] == 0 ) + np.sum( y[(y0==0) * (y1==1)] == 1 )
    accuracy  = n_correct / (n_class_zero + n_class_one)
    
    error = 1.0 - accuracy
    
    print('coverage = ', coverage)
    print('accuracy = ', accuracy)
    print('error = ', error)
    
    return coverage, accuracy, error

def get_coverage_error_accuracy_for_model_pairs_aditya(y, y0, y1):
    # Given two model predictions, find out coverage, error, accuracy
    
    n_examples   = len(y)
    n_rejections = np.sum( y0 == y1 )
    n_class_zero = np.sum( (y0==1) * (y1==0) ) 
    n_class_one  = np.sum( (y0==0) * (y1==1) ) 
    
    #print('n_rejections = ', n_rejections)
    #print('n_class_zero = ', n_class_zero)
    #print('n_class_one = ', n_class_one)
    
    coverage = 1.0 - (n_rejections / n_examples)
    
    n_correct = np.sum( y[(y0==1) * (y1==0)] == 0 ) + np.sum( y[(y0==0) * (y1==1)] == 1 )
    accuracy  = n_correct / (n_examples)
    
    error = coverage - accuracy
    
    #print('coverage = ', coverage)
    #print('accuracy = ', accuracy)
    #print('error = ', error)
    
    return coverage, accuracy, error, n_rejections, n_class_zero, n_class_one

def run_model_on_data(Xtst, ytst, cls, _th, mu, alpha):
    eval_batch_size = config['eval_batch_size']
    
    num_eval_examples = len(ytst)
    assert( Xtst.shape[0] == num_eval_examples )

    # Iterate over the samples batch-by-batch
    assert( num_eval_examples % eval_batch_size == 0 )
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size)) 
    
    y_scores = np.zeros( (num_eval_examples, 10), dtype=np.float32 )
    #print('\n\ncls={}, _lambda={}, _threshold={}'.format(cls, _lambda, _th))
    #print('\nEvaluate adversarial test performance at ({})'.format(datetime.now()))
    
    # Load the graph 
    model_dir = get_model_dir_name(cls, mu, alpha)
    tf.reset_default_graph()
    tf.set_random_seed(config['tf_random_seed'])
    np.random.seed(config['np_random_seed'])

    model = ResnetModel(threshold=_th, mu=mu, alpha=alpha)

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
            
            y_scores[ bstart:bend ] = raw_sigmoid_scores

            acc += cur_acc
            cov += cur_cov

        acc /= num_batches
        cov /= num_batches

        #print('    test accuracy={:.2f}%, coverage={:.4}'.format(100 * acc, cov))
        #print('  Finished Evaluating adversarial test performance at ({})'.format(datetime.now()))
    del model
    return y_scores


def get_coverage_error_for_given_parameters_obj( _predictions, params, y_true ):
    n_examples = len(y_true)    
    belongs_to = np.zeros( (n_examples, ), dtype=np.int32 )
    
    #_lambda_idx, _threshold_idx = cur_params
    y_pred = np.zeros( (n_examples, ), dtype=np.int32 )
    
    for cls in classes:
        _lambda = params['lambda'+str(cls)]  # lambdas[ _lambda_idx[cls] ]
        _threshold = params['th'+str(cls)] # thresholds[ _threshold_idx[cls] ]
        cur_class_pos = (_predictions[cls][_lambda] >= _threshold) * 1
        
        belongs_to += cur_class_pos
        
        # set the class to be current cls wherever cur_class_pos was true
        y_pred[ cur_class_pos==1 ] = cls
        
    #TODO
    #figure out which ones are rejected (if no one says it belongs to them, or if more than one says it belongs to them)
    
    n_rejections = np.sum( belongs_to != 1 )
    coverage = 1.0 - (n_rejections / n_examples)
    
    accuracy = np.sum( ( belongs_to == 1 ) * (y_true == y_pred) ) / n_examples
    error = coverage - accuracy
    
    return error, coverage

def get_coverage_error_for_given_parameters_pred_max( _predictions, lambdas, thresholds, cur_params, y_true ):
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
    
def get_coverage_error_for_given_parameters( _predictions, lambdas, thresholds, cur_params, y_true ):
    n_examples = len(y_true)    
    belongs_to = np.zeros( (n_examples, ), dtype=np.int32 )
    
    _lambda_idx, _threshold_idx = cur_params
    y_pred = np.zeros( (n_examples, ), dtype=np.int32 )
    
    for cls in classes:
        _lambda = lambdas[ _lambda_idx[cls] ]
        _threshold = thresholds[ _threshold_idx[cls] ]
        cur_class_pos = (_predictions[cls][_lambda] >= _threshold) * 1
        
        belongs_to += cur_class_pos
        
        # set the class to be current cls wherever cur_class_pos was true
        y_pred[ cur_class_pos==1 ] = cls
        
    #TODO
    #figure out which ones are rejected (if no one says it belongs to them, or if more than one says it belongs to them)
    
    n_rejections = np.sum( belongs_to != 1 )
    coverage = 1.0 - (n_rejections / n_examples)
    
    accuracy = np.sum( ( belongs_to == 1 ) * (y_true == y_pred) ) / n_examples
    error = coverage - accuracy
    
    return error, coverage

def get_coverage_error_for_given_parameters_per_class( _predictions, cls, _lambda, _threshold, y_true ):
    n_examples = len(y_true)    
    belongs_to = np.zeros( (n_examples, ), dtype=np.int32 )
    
    y_pred = -1 * np.ones( (n_examples, ), dtype=np.int32 )
    
    cur_class_pos = (_predictions[cls][_lambda] >= _threshold) * 1
    belongs_to += cur_class_pos
        
    # set the class to be current cls wherever cur_class_pos was true
    y_pred[ cur_class_pos==1 ] = cls

    n_rejections = np.sum( belongs_to != 1 )
    coverage = 1.0 - (n_rejections / n_examples)
    
    accuracy = np.sum( ( belongs_to == 1 ) * (y_true == y_pred) ) / n_examples
    error = coverage - accuracy
    
    return error, coverage

def get_predictions_normalized(lambdas, _predictions):
    for _lambda in lambdas:
        sum_scores = 0.0 * _predictions[0][_lambda]
        for cls in classes:
            sum_scores += _predictions[cls][_lambda]
        
        for cls in classes:
            _predictions[cls][_lambda] /= sum_scores

def gather_all_predictions(val_X, test_X, val_Y, test_Y, alpha, lambdas):
    # Gather all predictions
    _predictions = {}
    for cls in classes:
        _predictions[cls] = {}
       
    for _lambda in lambdas:
        scores = run_model_on_data(val_X, val_Y, 1, 0.5, _lambda, alpha)
        #scores = run_model_on_data(test_X, test_Y, 1, 0.5, _lambda, alpha)
        for cls in classes:
            _predictions[cls][_lambda] = scores[:,cls]
                            
    _test_predictions = {}
    for cls in classes:
        _test_predictions[cls] = {}
        
    for _lambda in lambdas:
        scores = run_model_on_data(test_X, test_Y, 1, 0.5, _lambda, alpha)
        for cls in classes:
            _test_predictions[cls][_lambda] = scores[:,cls]
            
    #get_predictions_normalized(lambdas, _predictions)
    #get_predictions_normalized(lambdas, _test_predictions)
    return _predictions, _test_predictions


def post_processing_mix_match_one_sided_models_same_lambda_th( val_X, test_X, val_Y, test_Y, lambdas = [1.0], thresholds = [0.5],
        desired_errors = [0.01, 0.02], alpha=0.5):
    print('\n\n Mixing multiple one sided models...')
    
    lambdas = sorted(lambdas)
    thresholds = sorted(thresholds)
    #print('lambdas = ', lambdas)
    #print('thresholds = ', thresholds)
    
    _predictions, _test_predictions = gather_all_predictions(val_X, test_X, val_Y, test_Y, alpha, lambdas)
                
    # Will mix-match now on the validation set
    print('\n\nResults = ')
    
    
    #- [DONE] Sort lambdas, thresholds
    #- [DONE] Pick initial set of parameters (lambda_1, ..., lambda_10, threshold_1, ..., threshold_10)
    #- [DONE] Find out the performance for this set of params (coverage, error)
    #- [DONE] Navigate to its one neighbours and find out their performance, pick the one with the highest coverage for given error
    #- Do this randomized start couple of times
    
    thresholds = np.unique(_predictions[classes[0]][lambdas[-1]])[::10]
    #print('thresholds = ', thresholds)
    
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
        

