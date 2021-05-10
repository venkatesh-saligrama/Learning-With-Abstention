


import random 


# N = total number of data points
# S (+ or -)
# Coverage : #( f>th ) / N
# Accuracy : #( f>th, y==1 ) / N
# Error    : #( f>th, y==-1 ) / N

def eval_test_adversarial(cls, best_loss, Xtst, ytst, model, sess, saver, model_dir, global_step):
    print('\nEvaluate adversarial test performance at ({})'.format(datetime.now()))
    eval_checkpoint_steps = config['eval_checkpoint_steps']
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    
    #Xtst, ytst = mnist.test.images, mnist.test.labels
    num_eval_examples = len(ytst)
    assert( Xtst.shape[0] == num_eval_examples )

    # Iterate over the samples batch-by-batch
    #assert( num_eval_examples % eval_batch_size == 0 )
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))                 
    aux_acc = 0
    acc = 0
    cov = 0
    loss = 0
    loss_l1 = 0
    loss_l2 = 0
    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = Xtst[bstart:bend, :]
        y_batch_aux = ytst[bstart:bend]
        
        dict_nat = {model.x_input: x_batch, model.y_input_aux: y_batch_aux, 
                    model.is_training:False}
        cur_cov, cur_aux_acc, cur_acc, cur_xent, cur_l1, cur_l2 = sess.run([
            model.mean_binary_cov, 
            model.accuracy_aux,
            model.mean_binary_acc, 
            model.xent, 
            model.binary_prob_xent, 
            model.xent_aux], feed_dict = dict_nat)

        acc += cur_acc
        cov += cur_cov
        aux_acc += cur_aux_acc
        
        loss += cur_xent
        loss_l1 += cur_l1
        loss_l2 += cur_l2

    aux_acc /= num_batches
    acc /= num_batches
    cov /= num_batches
    loss /= num_batches
    loss_l1 /= num_batches
    loss_l2 /= num_batches

    if best_loss > loss : 
        print('\n\nSaving the new trained checkpoint..')
        best_loss = loss
        saver.save(sess, os.path.join(model_dir, 'checkpoint'), global_step=global_step)
    
    #saver.save(sess, os.path.join(model_dir, 'checkpoint'), global_step=global_step)

    print('   test==> aux-accuracy={:.2f}%, accuracy={:.2f}%, coverage={:.4}, loss={:.4}, best-loss={:.4}, binary_prob_xent={:.4}, xent_aux={:.4},'.
          format(100 * aux_acc, 100 * acc, cov, loss, best_loss, loss_l1, loss_l2))
    print('  Finished Evaluating adversarial test performance at ({})'.format(datetime.now()))
    return best_loss
    
def evaluate_one_data_batch(cls, b, B, train_X, train_Y, batch_size, sess, model, best_loss, ii):
    # Output to stdout
    idx = random.randint(0,B-1)
    x_batch = train_X[idx*batch_size: (idx+1)*batch_size]
    y_batch_aux = train_Y[idx*batch_size: (idx+1)*batch_size]

    nat_dict = {model.x_input: x_batch, model.y_input_aux: y_batch_aux,
                model.is_training:False}
    
    cov, aux_acc, acc, xent, l1, l2 = sess.run([
            model.mean_binary_cov, 
            model.accuracy_aux,
            model.mean_binary_acc, 
            model.xent, 
            model.binary_prob_xent, 
            model.xent_aux], feed_dict = nat_dict)
    
    print('  Batch {}({}/{}):    ({})'.format(ii, b, B, datetime.now()))
    print('    training==> aux-accuracy={:.2f}%, accuracy={:.4}%, xent={:.4}, binary_prob_xent={:.4},xent_aux={:.4}, coverage={:.4}'.
          format(aux_acc*100, acc*100,  xent, l1, l2, cov))
    print('    best test loss: {:.2f}'.format(best_loss))
    
def change_labels_to_odd_even(y):
    print('y ', y.shape)
    odd_idx = y % 2 != 0
    even_idx =  y % 2 == 0
    
    new_y  = np.empty_like (y)
    new_y[:] = y
    
    new_y[ even_idx ] = 0
    new_y[ odd_idx ] = 1
    return new_y



