
###############################################################################
# Script for reproducing the results of CQR paper
###############################################################################

import six
import sys
sys.modules['sklearn.externals.six'] = six

import numpy as np
from reproducible_experiments.run_cqr_experiment import run_experiment
#from run_cqr_experiment import run_experiment


# list methods to test
'''test_methods = ['linear',
                'neural_net',
                'random_forest',
                'quantile_net',
                'cqr_quantile_net',
                'cqr_asymmetric_quantile_net',
                'rearrangement',
                'cqr_rearrangement',
                'cqr_asymmetric_rearrangement',
                'quantile_forest',
                'cqr_quantile_forest',
                'cqr_asymmetric_quantile_forest']'''

# list of datasets
'''dataset_names = ['meps_19',
                 'meps_20',
                 'meps_21',
                 'star',
                 'facebook_1',
                 'facebook_2',
                 'bio',
                 'blog_data',
                 'concrete',
                 'bike',
                 'community']'''

test_methods = ['cqr_quantile_net', 'lwa_neural_net']
#test_methods = ['cqr_quantile_net'] #, 'lwa_neural_net']
#dataset_names = ['concrete' ]
dataset_names = ['bike'] #'concrete'

significance_list = [0.90, 0.925, 0.95, 0.975]

#theta = 0.115789  # concrete
#theta = 0.094736   # bike
#quantiles_net = [theta, 0.8+theta]
quantiles_net = [0.05, 0.95]

#theta=0.86   # concrete
theta = 0.84  # Bike
_lambda= (theta) / (1. - theta)
_lambda1, _lambda2 = _lambda, _lambda
 
# vector of random seeds
#random_state_train_test = np.arange(1) # np.arange(20)
#random_state_train_test = np.arange(2) # np.arange(20)
random_state_train_test = np.arange(20)

for significance in significance_list:
    alpha = 1.0 - significance
    quantiles_net = [ alpha/2 , 1.0 - alpha/2]
    print(significance, ' --> quantiles_net = ', quantiles_net)

    _lambda = (1.0 - alpha/2) / (alpha/2)
    _lambda1, _lambda2 = _lambda, _lambda
    print(significance, ' --> _lambda1, _lambda2 = ', _lambda1, _lambda2)

    for test_method_id in range( len( test_methods ) ):
        for dataset_name_id in range( len(dataset_names) ):
            for random_state_train_test_id in range( len( random_state_train_test ) ):
                dataset_name = dataset_names[dataset_name_id]
                test_method = test_methods[test_method_id]
                random_state = random_state_train_test[random_state_train_test_id]

                # run an experiment and save average results to CSV file
                run_experiment(dataset_name, test_method, random_state, 
                      quantiles_net=quantiles_net,
                      significance=significance,
                      _lambda1=_lambda1, _lambda2=_lambda2 )

'''
dataset_name = 'concrete' # 'community'
random_state = 12

test_method = 'cqr_quantile_net' #'cqr_quantile_forest'
run_experiment(dataset_name, test_method, random_state)

test_method = 'lwa_neural_net' #'cqr_quantile_forest'
run_experiment(dataset_name, test_method, random_state)
'''
