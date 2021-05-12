
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import math
import random
from datetime import datetime
import json
import shutil
from timeit import default_timer as timer

import numpy as np
import pickle
import config

from combine_one_sided_models import post_processing_mix_match_one_sided_models_same_lambda_th

import sys
version = sys.version_info

config = config.get_config()
classes = list(range(0,10))
mus = np.linspace(0.1,3,30)
thresholds = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9];
print('Config = ', config)
print('Mus = ', mus)

def bernoulli_flip(p):
    return True if random.random() < p else False

def get_predictions_for_all_experts( scores, mu_t_pairs, V_t, data_idx ):
    n_experts = len( V_t ) # currently active experts
    predictions = np.zeros( n_experts, dtype=int )
    map_idx_to_pred = {}
    for j, idx in enumerate(V_t):
        mu, t = mu_t_pairs[idx]

        max_score = -1000.0
        max_class = -1
        for cls in classes:
            if (scores[cls][ mu ][data_idx] >= t):
                if max_score < scores[cls][mu][data_idx]:
                    max_score = scores[cls][mu][data_idx] 
                    max_class = cls
        
        predictions[j] = max_class #every OSC rejected this example
        map_idx_to_pred[ idx ] = j
    return predictions, map_idx_to_pred 

with open('./data/predictions.bin', 'rb') as fp:
    _predictions = pickle.load( fp )

with open('./data/test_predictions.bin', 'rb') as fp:
    _test_predictions = pickle.load( fp )

data = np.load('/home/anilkag/code/rnn_results/aditya/CIFAR-10/std_ce_64_dim_ft.npz', allow_pickle=True,)
test_X, test_Y = data['test_embd'], data['test_Y']
val_X, val_Y = data['val_embd'], data['val_Y']
train_X, train_Y = data['train_embd'], data['train_Y']
del train_X, val_X, test_X

print('np.unique(train_Y) = ', np.unique(train_Y))
#assert(1==2)

'''
Online Learning with Abstention scheme
'''
eta = 0.01  # learning rate
p = 0.3     # bernoulli coin bias
T = len(test_Y) #10000     # number of rounds

mu_t_pairs = []
V_t = []            # All possible experts (mus \times thresholds)
n_experts=0
for mu in mus:
    for t in thresholds:
        mu_t_pairs.append( (mu, t) )
        V_t.append( n_experts )
        n_experts += 1
active_experts = np.arange(n_experts)
assert(len(active_experts) == n_experts)

W_t = np.array([ 1./n_experts ]*n_experts)  # Weights : one for each expert
l_t = np.array([ 0. ]*n_experts)            # #of abstaintions for each expert
m_t = np.array([ 0. ]*n_experts)            # #of mistakes for each expert
algo_abstained = 0                          # #of abstaintions for the online learner
algo_error     = 0                          # #of mistakes for the online learner
n_data_points = test_Y.shape[0]
print('N data points = ', n_data_points)
print('W_t shape = ', W_t.shape)

for i in range(T):
   # Get context x_t, also have label y_t (dont reveal till abstention)
   # Evaluate f_i(x_t) for each i in V_t 
   #data_idx = random.randint( 0, n_data_points-1 )
   data_idx = i
   y_t = test_Y[ data_idx ]
   all_Vt_predictions, map_idx_to_pred = get_predictions_for_all_experts( _test_predictions, mu_t_pairs, V_t, data_idx )

   '''
   # Decision
   # If all have the same decision, make that decision
   # Otherwise
       At each time, toss an independent coin C_t ~ Bernoulli(p)
       If C_t = 1, abstain
       If C_t = 0, sample f_t ~ Pi = w_{t,f} / \sum_f w_{t,f} and play f_t(x_t)
   '''
   #all_equal = np.all( all_Vt_predictions == all_Vt_predictions[ active_experts[0] ] )
   all_equal = np.all( all_Vt_predictions[active_experts] == all_Vt_predictions[ active_experts[0] ] )
   if all_equal:
       prediction = all_Vt_predictions[0]
   else:
       C_t = bernoulli_flip(p)
       if C_t: 
           prediction = -1 # Abstain
       else:
           Pi = W_t / np.sum(W_t)
           sample_clf = np.random.choice( n_experts, p=Pi )
           prediction = all_Vt_predictions[ map_idx_to_pred[sample_clf] ] 

   if i%100 == 0:
       print('round=', i, ' -- example=', data_idx, ' -- prediction=', prediction, ' -- #active experts=', len(active_experts))
   '''
   # Update
     If we abstained, then get y_t
     Refine version space: 
         for f \in V_t, if f(x_t) \not \in \{ ?, y_t \}
           w_{t,f} = 0
   '''
   if prediction == -1:
       algo_abstained += 1
       for j, idx in enumerate(V_t):
           if all_Vt_predictions[ idx ] not in [-1, y_t]:
               W_t[ idx ] = 0
   elif prediction != y_t:
       algo_error += 1

   for j, idx in enumerate(V_t):
       if all_Vt_predictions[idx] == -1:
           l_t[idx] += 1
           if W_t[idx] != 0:
               W_t[ idx ] = W_t[ idx ] * (1. - eta)
       elif all_Vt_predictions[idx] != y_t:
           m_t[idx] += 1

   #V_t = []
   active_experts = []
   for j in range(n_experts):
       if W_t[j] != 0:
           #V_t.append(j)
           active_experts.append(j)
   active_experts = np.array( active_experts )

   if len(active_experts) == 0:
       print('active_experts set is empty... exiting the routine..')
       break

   '''
   # Always
     \ell^f_t = 1{ f(x_t) = ? } L^f_t
     L^f_t = \sum_{s < t} \ell^f_s
     w_{t,f} = w_{t-1, f} ( 1 - \eta \ell^f_t )
   '''

   '''
   # Need to validate that the best expert is in this set V_t
   # What are the metrics we are reporting?
   '''

print('#algo abstained = ', algo_abstained, '/', T)
print('#algo mistakes = ', algo_error, '/', T)
print('#of experts with non-zero weights = ', np.count_nonzero(W_t), '/', n_experts)
print('#of active experts = ', len(active_experts), '/', n_experts)
print('[experts] #mistakes(min) = ', np.min(m_t), '/', T)
print('[experts] #mistakes(max) = ', np.max(m_t), '/', T)
print('[experts] #abstained(min) = ', np.min(l_t), '/', T)
print('[experts] #abstained(max) = ', np.max(l_t), '/', T)
