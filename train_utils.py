
def train_model(train_X, train_Y, val_X, val_Y, test_X, test_Y, cls, 
    model_dir, threshold, _lambda, alpha=0.5, max_num_training_steps=21, lr=1e-3, max_lr=1e-5,
    warm_start=True, backbone=False):
    
    print(train_Y.shape)
    print(val_Y.shape)
    print(test_Y.shape)
    
    print('\n\nmodel directory = ', model_dir)
    if not os.path.exists(model_dir): os.makedirs(model_dir)

    tf.reset_default_graph()
    tf.set_random_seed(config['tf_random_seed'])
    np.random.seed(config['np_random_seed'])
    batch_size = config['training_batch_size']

    # Setting up the data and the model
    global_step = tf.contrib.framework.get_or_create_global_step()

    model = ResnetModel(threshold=threshold, mu=_lambda, alpha=alpha)
    max_step = tf.train.AdamOptimizer(max_lr).minimize(model.lambda_opt_xent, var_list=model._lambdas)
    #max_step = tf.train.AdamOptimizer(1e-3).minimize(model.lambda_opt_xent, var_list=model._lambdas)
    
    train_step = tf.train.AdamOptimizer(lr).minimize(model.xent, global_step=global_step, var_list=model.all_minimization_vars)
    #train_step = tf.train.AdamOptimizer(lr).minimize(model.xent, global_step=global_step, var_list=model.trn_vars)
    
    best_saver = tf.train.Saver(max_to_keep=3, var_list=tf.trainable_variables())
    saver = tf.train.Saver(max_to_keep=3)
    with open(model_dir + '/config.json', 'w' ) as f: json.dump( config, f)   

    ckpt = tf.train.latest_checkpoint(model_dir)
    print('\n\nrestore model directory = ', model_dir)

    import time
    start_time = time.time()

    N = len(train_X)
    assert(N % batch_size == 0)
    B = int(N/batch_size)

    backbone_update_freq = 20
    cur_epoch_counter = 0
    early_stop_criterion = 20
    best_loss = +10000.0
    prev_loss = +10000.0
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('\nInitial lambdas = ', sess.run(model._lambdas))
        #assert(1==2)

        if warm_start:
            saver.restore(sess, ckpt)
            #best_saver.restore(sess, restore_ckpt)
            
            best_loss = eval_test_adversarial(cls, best_loss, test_X, test_Y, model, sess, saver, model_dir, global_step)
            
            #print('\n\nSaving the new trained checkpoint..')
            #saver.save(sess, os.path.join(model_dir, 'checkpoint'), global_step=global_step)

        for ii in range(max_num_training_steps):            
            for b in range(B): 
                x_batch = train_X[ b*batch_size : (b+1)*batch_size ]
                y_batch_aux = train_Y[ b*batch_size : (b+1)*batch_size ]

                nat_dict = {model.x_input: x_batch, model.y_input_aux: y_batch_aux, 
                            model.is_training:True}
                sess.run(train_step, feed_dict=nat_dict)                
                sess.run(max_step, feed_dict=nat_dict)                
                
                #if (backbone == False) and (b%backbone_update_freq == 0):
                #    sess.run(backbone_train_step, feed_dict=nat_dict)                
                
                #if (b % (B-1) == 0):
                #    evaluate_one_data_batch(cls, b, B, train_X, train_Y, batch_size, sess, model, best_loss, ii)

            #print('\n\nEvaluate adversarial accuracy on test data..', ii)
            prev_loss = best_loss
            best_loss = eval_test_adversarial(cls, best_loss, test_X, test_Y, model, sess, saver, model_dir, global_step)

            #print('\nlambdas = ', sess.run(model._lambdas))
            #print('\nepsilons = ', sess.run(model._epsilons))
            
            if prev_loss == best_loss:
                cur_epoch_counter += 1 
                if cur_epoch_counter >= early_stop_criterion:
                    print('\nExiting early..')
                    break
            else:
                cur_epoch_counter = 0

        #assert(1 == 2)

    print('took ', int((time.time() - start_time)), 's')
    
    
def train_learning_with_abstention(lambdas = [1.0], threshold = 0.5, max_num_training_steps=21,
            lr=1e-4, max_lr=1e-5, backbone=False, warm_start=True, alpha=0.99):
    print('\n\n Training multiple one sided models...')
    print('mus = ', lambdas)
    
    cls=1
    for _lambda in lambdas:                       
        model_dir = get_model_dir_name(cls, _lambda, alpha, backbone=False)
        
        train_model(train_X, train_Y, val_X, val_Y, test_X, test_Y, cls,
            model_dir, threshold, _lambda, alpha=alpha, max_num_training_steps=max_num_training_steps, lr=lr,
            max_lr=max_lr, backbone=backbone, warm_start=warm_start)



