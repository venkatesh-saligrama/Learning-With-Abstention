

# based on https://github.com/tensorflow/models/tree/master/resnet
class ResnetModel(object):
  """ResNet model."""

  def __init__(self, threshold=0.5, alpha=0.5, mu=0.1 ):
    self.data_type = tf.float32
    self._build_model(threshold=threshold, mu=mu, alpha=alpha)

  def _feed_forward(self, x_input):
    x = x_input

    self.trn_vars = []
    embedding = x
    with tf.variable_scope('logits_f1', reuse=tf.AUTO_REUSE):
        x, w1, b1 = self._fully_connected(x, 128)
        self.trn_vars.extend([w1, b1])
    with tf.variable_scope('logits_f2', reuse=tf.AUTO_REUSE):
        self.pre_softmaxs, w2, b2 = self._fully_connected(x, 10)
        self.trn_vars.extend([w2, b2])
        
    with tf.variable_scope('logit_aux', reuse=tf.AUTO_REUSE):
      pre_softmax_aux, w, b = self._fully_connected(embedding, 10)
      self.trn_vars.extend([w, b])
        
    return pre_softmax_aux, embedding

  def _build_model(self, threshold=0.5, mu=0.1, alpha=0.5):
    with tf.variable_scope('input'):
      self.is_training = tf.placeholder(tf.bool, name='training')
      self.x_input = tf.placeholder(self.data_type,shape=[None, 64])
      self.y_input_aux = tf.placeholder(tf.int64, shape=None)
      self.pre_softmax_aux, self.embedding = self._feed_forward(self.x_input)
    
      self._epsilons = tf.get_variable("epsilons", shape=(10,), 
                                 #initializer=tf.random_normal_initializer(stddev=0.01),
                                #initializer=tf.truncated_normal(stddev=0.01, shape=(10,)),
                                       initializer=tf.constant_initializer(0.5),
                                 constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
      self.all_minimization_vars = tf.trainable_variables()
    
      self._lambdas = tf.get_variable("lambdas", shape=(10,), 
                                #initializer=tf.random_normal_initializer(stddev=0.01),
                                #initializer=tf.truncated_normal(stddev=0.01, shape=(10,)),
                                #initializer=tf.constant_initializer(mu - 0.5),
                                initializer=tf.random_uniform_initializer(minval=0.5*mu, maxval=1.5*mu),
                                constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
      
    
    #############################
    # AUXILLIARY CROSS ENTROPY LOSS
    self.y_xent_aux = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pre_softmax_aux, labels=self.y_input_aux)
    self.xent_aux = tf.reduce_sum(self.y_xent_aux)
    
    self.predictions_aux = tf.argmax(self.pre_softmax_aux, 1)
    self.correct_prediction_aux = tf.equal(self.predictions_aux, self.y_input_aux)
    self.num_correct_aux = tf.reduce_sum(tf.cast(self.correct_prediction_aux, tf.int64))
    self.accuracy_aux = tf.reduce_mean(tf.cast(self.correct_prediction_aux, tf.float32))
    #############################
    
    self.pre_softmax = self.pre_softmaxs #tf.concat(self.pre_softmaxs, 1)
    print('self.pre_softmax = ', self.pre_softmax)
    
    self.softmax_out = tf.nn.softmax( self.pre_softmax )
    print('softmax_out = ', self.softmax_out)
    
    tol=1e-8
    
    self.binary_prob_xent = 0.0
    self.mean_binary_acc = 0.0
    self.mean_binary_cov = 0.0
    for cls in classes:
        lamda = self._lambdas[cls]
        epsilon = self._epsilons[cls]
                                  
        y_out  = self.softmax_out[:, cls] #tf.nn.sigmoid( self.pre_softmaxs[i] )
        y_pred = tf.greater_equal(y_out, threshold)  #tf.argmax(self.pre_softmax, 1)

        y_out  = tf.reshape( y_out, [-1] )
        y_pred = tf.reshape( y_pred, [-1] )
        #print('y_out ', y_out)

        y_input = tf.equal( self.y_input_aux, cls )
        y_input = tf.cast(y_input, tf.float32)
        y_input = tf.reshape( y_input, [-1] )
        #print('y_input ', y_input)
    
        n_plus  = tf.reduce_sum( y_input )
        n_minus = tf.reduce_sum( 1-y_input )
        n_total = n_plus + n_minus
    
        #print('y_input * tf.math.log(self.y_out + tol) = ', y_input * tf.math.log(self.y_out + tol))
        #print('(1 - y_input) * tf.math.log(1-self.y_out+tol)', (1 - y_input) * tf.math.log(1-self.y_out+tol) )
    
        eps = 1e-7
        y_out = tf.clip_by_value(y_out, eps, 1-eps)
    
        # Our class is 1 label    
        loss_1 = (1./n_plus) * tf.reduce_sum( -y_input * tf.math.log(y_out + tol) )
        #loss_1 = (1./n_total) * tf.reduce_sum( -tf.math.log(y_out + tol) )
        loss_2 = (1./n_minus) * lamda * tf.reduce_sum(-(1 - y_input) * tf.math.log(1-y_out+tol) )

        #x = tf.reshape( self.pre_softmaxs[cls], [-1] ) 
        #z = y_input
        #y_xent = tf.nn.relu(x) - x * z + tf.math.log(1 + tf.math.exp(-tf.math.abs(x)))
    
        #y_xent = lamda * (1.- z) * x + (z + lamda * (1.-z)) * tf.nn.softplus(-x)
        y_xent = (loss_1 + loss_2) - lamda * epsilon
        self.binary_prob_xent = self.binary_prob_xent + tf.reduce_sum( y_xent )

        #self.xent = alpha * tf.reduce_sum(y_xent) + (1.0-alpha)* self.xent_aux

        y_pred = tf.cast(y_pred, tf.int64)
        correct_prediction = tf.equal(y_pred, tf.cast(y_input, tf.int64))

        coverage = tf.reduce_mean( tf.cast(y_pred, tf.float32) )
        num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        self.mean_binary_acc += accuracy
        self.mean_binary_cov += coverage
        
    self.mean_binary_acc = self.mean_binary_acc / 10.0
    self.mean_binary_cov = self.mean_binary_cov / 10.0
    
    print('\n\n\n Building model with alpha = ', alpha)
    self.binary_prob_xent = self.binary_prob_xent + mu * (tf.reduce_sum(self._epsilons) - 0.1)
    self.xent = alpha * self.binary_prob_xent + (1.0-alpha) * 0.125 * self.xent_aux
    self.lambda_opt_xent = - self.binary_prob_xent
    print('self.xent = ', self.xent)

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

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

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])



