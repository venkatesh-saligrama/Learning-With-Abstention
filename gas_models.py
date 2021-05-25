
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import argparse
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras import Model

from gas_train_base_learner import get_gas_dataset

class FullFeatureModel( Model ):
    def __init__(self, n_features, n_classes ):
        super(FullFeatureModel, self).__init__()

        self.f1 = Dense( 64, activation=tf.nn.relu, input_shape=(n_features, ) )
        self.bn1 = BatchNormalization()
        self.f2 = Dense( 128, activation=tf.nn.relu )
        self.bn2 = BatchNormalization()
        self.clf = Dense( n_classes )

    def call(self, x):
        x = self.f1(x)
        x = self.bn1(x)
        x = self.f2(x)
        x = self.bn2(x)
        x = self.clf(x)
        return x

'''
def FullFeatureModel( n_features, n_classes ):
    return tf.keras.models.Sequential([
             #tf.keras.layers.Flatten(input_shape=(n_features)),
             tf.keras.layers.InputLayer(input_shape=(n_features)),
             tf.keras.layers.Dense(128, activation='relu'),
             tf.keras.layers.Dropout(0.2),
             tf.keras.layers.Dense(n_classes)
           ])'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Online LWA Codebase')
    parser.add_argument('-d', '--data', default='./data/GAS/', type=str, metavar='DIR', help='path to dataset')
    args = parser.parse_args()
    print('args = ', args)

    trn_X, trn_y, tst_X, tst_y = get_gas_dataset( args.data )
    n_features = trn_X.shape[-1]
    n_classes = len( np.unique(trn_y) )

    #model = FullFeatureModel( n_features, n_classes )
    model = FullFeatureModel( n_features, n_classes )
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    model.fit( trn_X, trn_y, epochs=10 )

    test_loss, test_acc = model.evaluate(tst_X,  tst_y, verbose=2)
    print('\nTest accuracy:', test_acc)
