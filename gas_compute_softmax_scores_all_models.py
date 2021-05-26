

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import argparse
import pickle
import config
import numpy as np
from gas_utils import gather_all_predictions
from gas_train_base_learner import get_gas_dataset

parser = argparse.ArgumentParser(description='Abstention training Codebase')
parser.add_argument('-d', '--data', default='./data/GAS/', type=str, metavar='DIR', help='path to dataset')
parser.add_argument('-md', '--model_dir', default='./models/abstention/trn_scratch', type=str, help='store trained models here')
args = parser.parse_args()
print('args = ', args)


trn_X, trn_y, tst_X, tst_y = get_gas_dataset( args.data )
n_features = trn_X.shape[-1]
trn_X = trn_X[:, :n_features//2]
tst_X = tst_X[:, :n_features//2]
n_features = trn_X.shape[-1]

fp_name = os.path.join( args.data, 'trn_pred_wk_' + str(False) + '.npy' )
sc_trn_y = np.load( fp_name )

fp_name = os.path.join( args.data, 'tst_pred_wk_' + str(False) + '.npy' )
sc_tst_y = np.load( fp_name )
print('  ---     trn acc = ', np.mean( sc_trn_y == trn_y ))
print('  ---     tst acc = ', np.mean( sc_tst_y == tst_y ))

val_X, val_y = tst_X, tst_y

print('shapes x_val, y_val', val_X.shape, val_y.shape)
print('shapes x_test, y_test', tst_X.shape, tst_y.shape)    
print('\n\nnp.unique(train_Y) = ', np.unique(trn_y))

alpha= 0.999999 #0.999 #0.99
mus = np.linspace(0.1,3,30)
print('Mus = ', mus)

_predictions, _test_predictions = gather_all_predictions(val_X, val_y, tst_X, tst_y, alpha, mus, args=args)

print( _predictions[0][mus[0]].shape )

with open( args.data + 'predictions.bin', 'wb') as fp:
    pickle.dump( _predictions, fp, protocol=pickle.HIGHEST_PROTOCOL )
    
with open( args.data + 'test_predictions.bin', 'wb') as fp:
    pickle.dump( _test_predictions, fp, protocol=pickle.HIGHEST_PROTOCOL )

with open( args.data + 'predictions.bin', 'rb') as fp:
    _predictions = pickle.load( fp )


print( _predictions[0][mus[0]].shape )
