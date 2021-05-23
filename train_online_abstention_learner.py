
'''
#Run for T = 500:500:10K
#Each run for 20 times

#For each run - record 
#1. our mistakes, our abstentions
#2. Opt mistakes, abst of opt mistake maker
#3. Mistake-matched abstentions, mistakes of this.

#Statistics
#1. M_t = avg_over_runs(our mistakes - opt mistakes), A_t = avg_over_runs(our abstentions - abst of opt mistake maker)
#2. Extra abstentions =  avg_over_runs(our abstentions - mistake-matched-abstention)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import logging
import random
from datetime import datetime
import json
import shutil
from timeit import default_timer as timer


import multiprocessing
import numpy as np
import pickle
import config
import argparse
#from combine_one_sided_models import post_processing_mix_match_one_sided_models_same_lambda_th

import sys
version = sys.version_info

config = config.get_config()
classes = list(range(0,10))
mus = np.linspace(0.1,3,30)
#thresholds = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.88, 0.9];
thresholds = np.linspace(0.05,0.95,100)
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


def run_one_experiment( process_id, T, val_Y, test_Y, _predictions,  _test_predictions, return_stats ):

    '''
    Online Learning with Abstention scheme
    '''

    #T = len(test_Y) #10000     # number of rounds
    p = math.sqrt(2*math.log(T)/T) # 0.02 #0.3     # bernoulli coin bias
    eta = p #0.01  # learning rate
    theta = 0.01

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
    o_t = np.array([ 0. ]*n_experts)            # #of abstaintions for each expert
    m_t = np.array([ 0. ]*n_experts)            # #of mistakes for each expert
    algo_abstained = 0                          # #of abstaintions for the online learner
    algo_error     = 0                          # #of mistakes for the online learner
    n_data_points = test_Y.shape[0]
    print('N data points = ', n_data_points)
    print('W_t shape = ', W_t.shape)

    data_permutation = np.random.permutation( len(test_Y) )  #(T)
    print(data_permutation)

    for i in range(T):
       # Get context x_t, also have label y_t (dont reveal till abstention)
       # Evaluate f_i(x_t) for each i in V_t 
       #data_idx = random.randint( 0, n_data_points-1 )
       #data_idx = i
       data_idx = data_permutation[i]
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
       if False : #all_equal:
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
                   #W_t[ idx ] = 0
                   o_t[ idx ] += 1
                   if o_t[ idx ] > (2*p* theta * T):
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

    min_mistakes = np.min(m_t)

    optimal_idx = -1
    optimal_mistakes = T
    optimal_abstained = T

    amm_mistake = T
    amm_abs = T

    mistake_matched_abs = T
    mma_mis = T
    #for idx in active_experts:
    for idx in range(n_experts):
        if min_mistakes == m_t[idx]:
            if optimal_abstained > l_t[idx]:
                optimal_mistakes = m_t[idx]
                optimal_abstained = l_t[idx]
                optimal_idx = idx

        if m_t[idx] <= algo_error:
            if mistake_matched_abs > l_t[idx]:
                mma_mis = m_t[idx]
                mistake_matched_abs = l_t[idx]

        if l_t[idx] <= algo_abstained:
            if amm_mistake > m_t[idx]:
                amm_mistake = m_t[idx]
                amm_abs = l_t[idx]


    if optimal_idx != -1:
        print('[optimal experts] #mistakes = ', optimal_mistakes, '/', T)
        print('[optimal experts] #abstained = ', optimal_abstained, '/', T)
        print('Mistake Matched #Abenstions = ', mistake_matched_abs, '/',T)
        print('Mistake Matched #Mistakes = ', mma_mis, '/',T)
        print('[AMM] #Abenstions = ', amm_abs, '/',T)
        print('[AMM] #Mistakes = ', amm_mistake, '/',T)
    else:
        print('No optimal expert found.')

    #
    #for idx in active_experts:
    #    print('[experts=', idx, '] --> mu, t', str(mu_t_pairs[idx]))
    #    print('[expert=', idx, '] #mistakes = ', m_t[idx], '/', T)
    #    print('[expert=', idx, '] #abstained = ', l_t[idx], '/', T)
    stats = [ algo_error, algo_abstained, 
              optimal_mistakes, optimal_abstained,
              mma_mis, mistake_matched_abs,
              amm_mistake, amm_abs,
              np.count_nonzero(W_t), len(active_experts), np.min(m_t), np.max(m_t), np.min(l_t), np.max(l_t) ]
    return ( len(active_experts) == 0 ), stats


def rerun_if_failed_one_experiment( process_id, T, val_Y, test_Y, _predictions,  _test_predictions, return_stats ):
    failed, stats = run_one_experiment( process_id, T, val_Y, test_Y, _predictions,  _test_predictions, return_stats )
    cnt = 0
    while failed and (cnt < 3):
        failed, stats = run_one_experiment( process_id, T, val_Y, test_Y, _predictions,  _test_predictions, return_stats )
        cnt += 1
    if not failed: 
        return_stats[ str(T) + '-r-' + str(process_id) ] = stats
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Online LWA Codebase')
    parser.add_argument('-d', '--data', default='./data/', type=str, metavar='DIR', help='path to dataset')
    args = parser.parse_args()
    print('args = ', args)


    with open( args.data + 'predictions.bin', 'rb') as fp:
        _predictions = pickle.load( fp )

    with open( args.data + 'test_predictions.bin', 'rb') as fp:
        _test_predictions = pickle.load( fp )

    #thresholds = np.unique(_predictions[classes[0]][mus[-1]])[::10]
    #thresholds = np.unique(_predictions[classes[0]][mus[-1]])[::100]
    #print('#thresholds = ', len(thresholds))

    data = np.load( args.data + 'std_ce_64_dim_ft.npz', allow_pickle=True,)
    test_X, test_Y = data['test_embd'], data['test_Y']
    val_X, val_Y = data['val_embd'], data['val_Y']
    train_X, train_Y = data['train_embd'], data['train_Y']
    del train_X, val_X, test_X

    start = time.time()

    n_runs = 20 #1 #5 #3 #20
    #Ts = [1500] #[500, 1000, 1500]
    Ts = list( range(500, 10500, 500) )
    print('Ts = ', Ts)
    print('n_runs = ', n_runs)
    runs = list(range(n_runs))
    rTs = []
    for t in Ts:
        for r in runs: 
            rTs.append( (r, t) )

    manager = multiprocessing.Manager()
    return_stats = manager.dict()

    for T in Ts:
       jobs = []
       #for _, rT in enumerate(rTs):
       for process_id in runs:
            #process_id, T = rT
            #T = 500
            #process_id = 0
            #run_one_experiment( process_id, T, val_Y, test_Y, _predictions,  _test_predictions )
            #p = multiprocessing.Process( target=run_one_experiment, args=( process_id, T, val_Y, test_Y, _predictions,  _test_predictions, return_stats ) )
            p = multiprocessing.Process( target=rerun_if_failed_one_experiment, args=( process_id, T, val_Y, test_Y, _predictions,  _test_predictions, return_stats ) )
            jobs.append(p)
            p.start()
            #p.join()

       for proc in jobs:
           proc.join()
       #break

    #print(return_stats.values())
    print( return_stats )
    print('time taken = ', time.time() - start, ' s')

    with open('runs-varying-Ts-20.pickle', 'wb') as handle:
        pickle.dump(return_stats, handle, protocol=pickle.HIGHEST_PROTOCOL) 

    '''
    #Statistics
    #1. M_t = avg_over_runs(our mistakes - opt mistakes), A_t = avg_over_runs(our abstentions - abst of opt mistake maker)
    #2. Extra abstentions =  avg_over_runs(our abstentions - mistake-matched-abstention)
    '''

    print('\nCompiling results..\n')
    for T in Ts:
        m_t, a_t, extra_a_t, extra_m_t, valid_runs = 0, 0, 0, 0, 0
        for process_id in runs:
           key = str(T) + '-r-' + str(process_id)
           if key in return_stats:
               valid_runs += 1
               stats = return_stats[key]

               algo_error, algo_abstained = stats[0], stats[1]
               optimal_mistakes, optimal_abstained = stats[2], stats[3]
               mma_mis, mistake_matched_abs = stats[4], stats[5]
               amm_mis, amm_abs = stats[6], stats[7]

               m_t += ( algo_error - optimal_mistakes ) 
               a_t += ( algo_abstained - optimal_abstained )
               extra_a_t += ( algo_abstained - mistake_matched_abs )
               extra_m_t += ( algo_error - amm_mis )

        m_t /= valid_runs
        a_t /= valid_runs
        extra_a_t /= valid_runs
        print('\t\tT=', T, ', m_t=', m_t, ', a_t=', a_t, ', extra_a_t=', extra_a_t, ', extra_m_t=', extra_m_t)

