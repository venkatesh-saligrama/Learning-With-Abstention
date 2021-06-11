



import numpy as np
import pandas as pd

pd.set_option('precision', 3)

def get_mean_std_various_significance_levels( out_name, significance_list=[0.90, 0.925, 0.95, 0.975], dataset='concrete' ):
    df = pd.read_csv(out_name)

    our_method = 'LWA-Net'
    baseline = 'CQR Net'
    methods = [ our_method, baseline ]
    
    for significance in significance_list: 
        coverage_str = 'Coverage (expected ' + str(100 - significance*100) + '%)'
        tmp_df = df.loc[ df['Significance'] == significance ]
        tmp_df = tmp_df[ [ 'name', 'method', 'seed', 'Significance', coverage_str, 'Avg. Length' ]  ]

        for method in methods:
            stats = tmp_df[ tmp_df['method'] == method ]
            #print(stats)
            mean_stats = stats.mean()
            std_stats = stats.std()
            #print( mean_stats[ [ coverage_str, 'Avg. Length'] ] )
            #print( std_stats[ [ coverage_str, 'Avg. Length'] ]  )
        
            print( dataset, '\t', significance, '\t', coverage_str, '\t', method, '\t', mean_stats[coverage_str], '+-', std_stats[coverage_str], '\t', mean_stats['Avg. Length'], '+-', std_stats['Avg. Length'] )
            #exit(1)

get_mean_std_various_significance_levels( './results/results-concrete-significance.csv', dataset='concrete' )
get_mean_std_various_significance_levels( './results/results-bike-significance.csv', dataset='bike' )

