

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pickle
import config
import numpy as np
from combine_one_sided_models import gather_all_predictions

data = np.load('/home/anilkag/code/rnn_results/aditya/CIFAR-10/std_ce_64_dim_ft.npz', allow_pickle=True,)
test_X, test_Y = data['test_embd'], data['test_Y']
val_X, val_Y = data['val_embd'], data['val_Y']
train_X, train_Y = data['train_embd'], data['train_Y']

del train_X

#print('shapes x_train, y_train', train_X.shape, train_Y.shape)
print('shapes x_val, y_val', val_X.shape, val_Y.shape)
print('shapes x_test, y_test', test_X.shape, test_Y.shape)    
print('\n\nnp.unique(train_Y) = ', np.unique(train_Y))

config = config.get_config()
alpha= 0.9999 #0.99
mus = np.linspace(0.1,3,30)
print('Config = ', config)
print('Mus = ', mus)

_predictions, _test_predictions = gather_all_predictions(val_X, test_X, val_Y, test_Y, alpha, mus)

print( _predictions[0][mus[0]].shape )

with open('./data/predictions.bin', 'wb') as fp:
    pickle.dump( _predictions, fp, protocol=pickle.HIGHEST_PROTOCOL )
    
with open('./data/test_predictions.bin', 'wb') as fp:
    pickle.dump( _test_predictions, fp, protocol=pickle.HIGHEST_PROTOCOL )

with open('./data/predictions.bin', 'rb') as fp:
    _predictions = pickle.load( fp )


print( _predictions[0][mus[0]].shape )
