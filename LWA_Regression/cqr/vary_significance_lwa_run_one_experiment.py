
###############################################################################
# Script for reproducing the results of CQR paper
###############################################################################

import warnings
warnings.filterwarnings("ignore")

import six
import sys
sys.modules['sklearn.externals.six'] = six

from joblib import Parallel, delayed

import os
import numpy as np
import pandas as pd
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

#test_methods = ['cqr_quantile_net', 'lwa_neural_net']
test_methods = ['cqr_quantile_net'] #, 'lwa_neural_net']
dataset_names = ['concrete' ]
#dataset_names = ['bike'] #'concrete'

significance_list = [0.90, 0.925, 0.95, 0.975]
#significance_list = [0.90]

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
#random_state_train_test = np.arange(20)
n_jobs = 10 #8
random_state_train_test = np.arange(20).tolist()

outdir = './results/'
if not os.path.exists(outdir):
    os.mkdir(outdir)

for significance in significance_list:
    alpha = round( 1.0 - significance, 5 )
    quantiles_net = [ alpha/2 , 1.0 - alpha/2]
    print(significance, ' --> quantiles_net = ', quantiles_net)

    _lambda = round( (1.0 - alpha/2) / (alpha/2), 5)
    _lambda1, _lambda2 = _lambda, _lambda
    print(significance, ' --> _lambda1, _lambda2 = ', _lambda1, _lambda2)

    for test_method_id in range( len( test_methods ) ):
        for dataset_name_id in range( len(dataset_names) ):
            dataset_name = dataset_names[dataset_name_id]
            test_method = test_methods[test_method_id]

            results = Parallel(n_jobs=n_jobs)(delayed( run_experiment )(  dataset_name, test_method, random_state, 
                      quantiles_net=quantiles_net, significance=alpha, _lambda1=_lambda1, _lambda2=_lambda2, 
                      flush_summary=False, save_to_csv=False ) for random_state in random_state_train_test)
            df = results[0]
            for df2 in results[1:]:
                df = pd.concat([df2, df], ignore_index=True)
            #print(results)
            print(df)
            #out_name = outdir + 'results-' + dataset_name  + '-significance-parallel.csv'
            out_name = outdir + 'results-' + dataset_name  + '-significance-tmp.csv'

            if os.path.isfile(out_name):
                df2 = pd.read_csv(out_name)
                df = pd.concat([df2, df], ignore_index=True)

            df.to_csv(out_name, index=False)

            #exit(1)

            '''
            for random_state_train_test_id in range( len( random_state_train_test ) ):
                random_state = random_state_train_test[random_state_train_test_id]

                # run an experiment and save average results to CSV file
                run_experiment(dataset_name, test_method, random_state, 
                      quantiles_net=quantiles_net,
                      significance=alpha,
                      _lambda1=_lambda1, _lambda2=_lambda2 ) '''

'''
dataset_name = 'concrete' # 'community'
random_state = 12

test_method = 'cqr_quantile_net' #'cqr_quantile_forest'
run_experiment(dataset_name, test_method, random_state)

test_method = 'lwa_neural_net' #'cqr_quantile_forest'
run_experiment(dataset_name, test_method, random_state)
'''
