
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import argparse
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras import Model

from gas_train_base_learner import get_gas_dataset


class BaselineModel( object ):

  def __init__(self, n_features, n_classes, weight_decay=0.01 ):
    self.data_type = tf.float32
    self.n_features = n_features
    self.n_classes = n_classes
    self.weight_decay = weight_decay
    self._build_model()

  def _feed_forward(self, x_input):
    x = x_input

    self.trn_vars = []
    embedding = x
    with tf.variable_scope('logits_f1', reuse=tf.AUTO_REUSE):
        x, w1, b1 = self._fully_connected(x, 64)
        x = tf.nn.relu(x)
        self.trn_vars.extend([w1, b1])

    with tf.variable_scope('logits_f2', reuse=tf.AUTO_REUSE):
        x, w1, b1 = self._fully_connected(x, 16)
        x = tf.nn.relu(x)
        self.trn_vars.extend([w1, b1])

    with tf.variable_scope('logit_aux', reuse=tf.AUTO_REUSE):
      pre_softmax_aux, w, b = self._fully_connected(embedding, self.n_classes)
      self.trn_vars.extend([w, b])
        
    return pre_softmax_aux, embedding

  def _build_model(self):
    with tf.variable_scope('input'):
      self.is_training = tf.placeholder(tf.bool, name='training')
      self.x_input = tf.placeholder(self.data_type,shape=[None, self.n_features])
      self.y_input_aux = tf.placeholder(tf.int64, shape=None)
      self.pre_softmax_aux, self.embedding = self._feed_forward(self.x_input)
      self.all_minimization_vars = tf.trainable_variables()
    
    #############################
    # AUXILLIARY CROSS ENTROPY LOSS
    self.l2_loss_aux = tf.add_n([ tf.nn.l2_loss(v) for v in self.trn_vars ]) * self.weight_decay
    self.y_xent_aux = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pre_softmax_aux, labels=self.y_input_aux)
    self.xent_aux = tf.reduce_sum(self.y_xent_aux)
    
    self.predictions_aux = tf.argmax(self.pre_softmax_aux, 1)
    self.correct_prediction_aux = tf.equal(self.predictions_aux, self.y_input_aux)
    self.num_correct_aux = tf.reduce_sum(tf.cast(self.correct_prediction_aux, tf.int64))
    self.accuracy_aux = tf.reduce_mean(tf.cast(self.correct_prediction_aux, tf.float32))
    #############################

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    num_non_batch_dimensions = len(x.shape)
    prod_non_batch_dimensions = 1
    for ii in range(num_non_batch_dimensions - 1):
      prod_non_batch_dimensions *= int(x.shape[ii + 1])
    x = tf.reshape(x, [tf.shape(x)[0], -1])
    w = tf.get_variable('ffDW', [prod_non_batch_dimensions, out_dim],
        self.data_type, initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim], self.data_type, initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b), w, b


class FullFeatureModel( Model ):
    def __init__(self, n_features, n_classes ):
        super(FullFeatureModel, self).__init__()

        l2_reg = tf.keras.regularizers.l2( l=0.01 )

        self.f1 = Dense( 64, activation=tf.nn.relu, input_shape=(n_features,), kernel_regularizer=l2_reg )
        self.bn1 = BatchNormalization()
        self.f2 = Dense( 16, activation=tf.nn.relu, kernel_regularizer=l2_reg )
        self.bn2 = BatchNormalization()
        self.clf = Dense( n_classes, kernel_regularizer=l2_reg )

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

    model = FullFeatureModel( n_features, n_classes )
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    #optim = tf.keras.optimizers.SGD( learning_rate=0.01, momentum=0.9, nesterov=True )
    optim = tf.keras.optimizers.Adam()
    model.compile(optimizer=optim, loss=loss_fn, metrics=['accuracy'])
    model.fit( trn_X, trn_y, epochs=10 )

    test_loss, test_acc = model.evaluate(tst_X,  tst_y, verbose=2)
    print('\nTest accuracy:', test_acc)
