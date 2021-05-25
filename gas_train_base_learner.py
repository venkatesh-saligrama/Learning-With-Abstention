
import argparse
import numpy as np
from sklearn import svm
from sklearn.datasets import load_svmlight_file
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import make_classification

def get_gas_dataset( data_path ):
    Xs, Ys = [], []
    for i in range(1, 11):
        X, y = load_svmlight_file( data_path + 'batch' + str(i) + '.dat' )
        #print(np.unique(y))
        X = X.toarray()
        Xs.append( X )
        Ys.append( y )

    split_idx = 7
    trn_X = np.concatenate( Xs[:split_idx], axis=0 )
    trn_y = np.concatenate( Ys[:split_idx], axis=0 )
    trn_y -= 1
    print('trn_X', trn_X.shape)
    print('trn_y', trn_y.shape)
    print('n_unique(y) ', np.unique( trn_y ))

    tst_X = np.concatenate( Xs[split_idx:], axis=0 )
    tst_y = np.concatenate( Ys[split_idx:], axis=0 )
    tst_y -= 1
    print('tst_X', tst_X.shape)
    print('tstn_y', tst_y.shape)
    print('n_unique(y) ', np.unique( tst_y ))

    #clf = svm.SVC()
    #clf = svm.LinearSVC(C=20.)
    #clf = svm.LinearSVC( max_iter=10000 )
    #clf = make_pipeline( StandardScaler(), svm.LinearSVC(C=100., random_state=0) )
    clf = make_pipeline( MinMaxScaler(), svm.LinearSVC(C=100., random_state=0) )

    clf.fit( trn_X, trn_y )
    print('Train Accuracy = ',  clf.score( trn_X, trn_y ) )
    print('Test Accuracy = ',  clf.score( tst_X, tst_y ) )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Online LWA Codebase')
    parser.add_argument('-d', '--data', default='./data/GAS/', type=str, metavar='DIR', help='path to dataset')
    args = parser.parse_args()
    print('args = ', args)

    get_gas_dataset( args.data )
