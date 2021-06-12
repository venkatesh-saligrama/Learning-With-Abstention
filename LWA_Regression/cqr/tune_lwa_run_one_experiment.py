
###############################################################################
# Script for reproducing the results of CQR paper
###############################################################################

import warnings
warnings.filterwarnings("ignore")

from joblib import Parallel, delayed
import gc 
import os
import six
import sys
sys.modules['sklearn.externals.six'] = six

import pandas as pd
import numpy as np
from reproducible_experiments.run_cqr_experiment import run_experiment, val_run_experiment
#from run_cqr_experiment import run_experiment


outdir = './results/'
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
random_state_train_test = np.arange(20).tolist()
#random_state_train_test = np.arange(5).tolist()
#random_state_train_test = np.arange(2).tolist()
random_state = 1234 #np.arange(1) # np.arange(20)

#            test_method = test_methods[test_method_id]
#            random_state = random_state_train_test[random_state_train_test_id]

import random

n_jobs=10
n_vals =20

#significance_list = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
significance_list = [0.96, 0.97, 0.98]
#significance_list = [ 0.99]
#significance_list = [0.99] #, 0.925, 0.95, 0.975]

#dataset_name = 'concrete'
dataset_name = 'bike' #'concrete'
test_method = 'cqr_quantile_net'
'''
for significance in significance_list:
  target_cov = 100*significance
  max_cov, max_len, max_theta = None, None, None
  alpha = round( 1.0 - significance, 5 )
  #for theta in np.linspace(0.0, 2*alpha, num=n_vals):
  #  quantiles_net = [ theta, 1. - 2*alpha + theta ]

  #results = Parallel(n_jobs=n_jobs)(delayed( val_run_experiment )(  dataset_name, test_method, random_state, 
  results = Parallel(n_jobs=n_jobs)(delayed( val_run_experiment )(  dataset_name, test_method, random.randint(1, 99), 
                      quantiles_net=[ theta, 1. - 2*alpha + theta ], significance=alpha, theta=theta, 
                      save_to_csv=False ) for theta in np.linspace(0.0, 2*alpha, num=n_vals))
  gc.collect()

  for (cur_cov, cur_len, theta) in results:
    #cur_cov, cur_len = val_run_experiment(dataset_name, test_method, random_state, significance=alpha, quantiles_net=quantiles_net)
    if (max_theta is None) :
        max_cov, max_len, max_theta = cur_cov, cur_len, theta
    elif ( cur_cov >= target_cov ) and ( cur_len <= max_len ):
        max_cov, max_len, max_theta = cur_cov, cur_len, theta
  del results
  gc.collect()
  print(' --> CQR Net --> significance, --> max_cov, max_len, max_theta, 0.8+theta ',  significance, max_cov, max_len, max_theta, 1. - 2*alpha+max_theta )
#assert(1==2)

  results = Parallel(n_jobs=n_jobs)(delayed( run_experiment )(  dataset_name, test_method, random_state, 
                  quantiles_net=[ max_theta, 1. - 2*alpha + max_theta ], significance=alpha, 
                  flush_summary=False, save_to_csv=False ) for random_state in random_state_train_test)
  gc.collect()
  #print(results)
  #print(' len(results) ', len(results))
  #assert(1==2)
  df = results[0]
  for df2 in results[1:]:
      df = pd.concat([df2, df], ignore_index=True)

  del results
  gc.collect()
  print(df)
  #out_name = outdir + 'results-' + dataset_name  + '-significance-parallel.csv'
  out_name = outdir + 'results-' + dataset_name  + '-significance-tmp.csv'

  if os.path.isfile(out_name):
                df2 = pd.read_csv(out_name)
                df = pd.concat([df2, df], ignore_index=True)

  df.to_csv(out_name, index=False)
'''


#n_vals=40
test_method = 'lwa_neural_net'
max_cov, max_len, max_theta = None, None, None
#for theta in np.linspace(4, 99, num=n_vals):
for significance in significance_list:
  target_cov = 100*significance
  alpha = round( 1.0 - significance, 5 )
  max_cov, max_len, max_theta = None, None, None

  #results = Parallel(n_jobs=n_jobs)(delayed( val_run_experiment )(  dataset_name, test_method, random_state, 
  results = Parallel(n_jobs=n_jobs)(delayed( val_run_experiment )(  dataset_name, test_method, random.randint(1, 99), 
                      significance=alpha,  _lambda1=( ( theta ) / ( 1. - theta ) ),
                      _lambda2=( ( theta ) / ( 1. - theta ) ), theta=theta,
                      save_to_csv=False ) for theta in np.linspace(1.0 - 2*alpha, 1.0 - (alpha/2.5), num=n_vals)  )

  gc.collect()
  for (cur_cov, cur_len, theta) in results:
  #for theta in np.linspace(1.0 - 2*alpha, 1.0 - (alpha/2.5), num=n_vals):
    #_lambda = theta
    _lambda = ( theta ) / ( 1. - theta )
    #cur_cov, cur_len = val_run_experiment(dataset_name, test_method, random_state, significance=alpha, _lambda1=_lambda, _lambda2=_lambda)
    if (max_theta is None) :
        max_cov, max_len, max_theta = cur_cov, cur_len, theta
    elif ( cur_cov >= target_cov ) and ( cur_len <= max_len ):
        max_cov, max_len, max_theta = cur_cov, cur_len, theta
  del results
  gc.collect()
  print(' --> LWA Net --> significance --> max_cov, max_len, max_lambda ',  significance, max_cov, max_len, max_theta )

  results = Parallel(n_jobs=n_jobs)(delayed( run_experiment )(  dataset_name, test_method, random_state, 
                  significance=alpha,  _lambda1=( ( max_theta ) / ( 1. - max_theta ) ),
                      _lambda2=( ( max_theta ) / ( 1. - max_theta ) ),
                  flush_summary=False, save_to_csv=False ) for random_state in random_state_train_test)
  gc.collect()
  df = results[0]
  for df2 in results[1:]:
      df = pd.concat([df2, df], ignore_index=True)
      #print(results)

  del results
  gc.collect()
  #print(df)
  #out_name = outdir + 'results-' + dataset_name  + '-significance-parallel.csv'
  out_name = outdir + 'results-' + dataset_name  + '-significance-tmp.csv'

  if os.path.isfile(out_name):
                df2 = pd.read_csv(out_name)
                df = pd.concat([df2, df], ignore_index=True)

  df.to_csv(out_name, index=False)



