
###############################################################################
# Script for reproducing the results of CQR paper
###############################################################################

import six
import sys
sys.modules['sklearn.externals.six'] = six

import numpy as np
from reproducible_experiments.run_cqr_experiment import run_experiment, val_run_experiment
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
dataset_names = ['concrete' ]
 
# vector of random seeds
random_state = 1234 #np.arange(1) # np.arange(20)

#            test_method = test_methods[test_method_id]
#            random_state = random_state_train_test[random_state_train_test_id]

target_cov = 90
n_vals = 20

dataset_name = 'concrete'
test_method = 'cqr_quantile_net'
max_cov, max_len, max_theta = None, None, None
for theta in np.linspace(0.0, 0.1, num=n_vals):
   quantiles_net = [ theta, 1. - 0.1 + theta ]
   cur_cov, cur_len = val_run_experiment(dataset_name, test_method, random_state, quantiles_net=quantiles_net)
   if (max_theta is None) :
       max_cov, max_len, max_theta = cur_cov, cur_len, theta
   elif ( cur_cov >= target_cov ) and ( cur_len <= max_len ):
       max_cov, max_len, max_theta = cur_cov, cur_len, theta
print(' --> CQR Net --> max_cov, max_len, max_theta ',  max_cov, max_len, max_theta )


test_method = 'lwa_neural_net'
max_cov, max_len, max_theta = None, None, None
for theta in np.linspace(4, 99, num=n_vals):
   _lambda = theta
   cur_cov, cur_len = val_run_experiment(dataset_name, test_method, random_state, _lambda1=_lambda, _lambda2=_lambda)
   if (max_theta is None) :
       max_cov, max_len, max_theta = cur_cov, cur_len, theta
   elif ( cur_cov >= target_cov ) and ( cur_len <= max_len ):
       max_cov, max_len, max_theta = cur_cov, cur_len, theta
print(' --> LWA Net --> max_cov, max_len, max_lambda ',  max_cov, max_len, max_theta )
