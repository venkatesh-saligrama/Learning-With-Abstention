
def get_config():

  config = {
  "_comment": "===== MODEL CONFIGURATION =====",
  "model_dir": "/home/anilkag/code/LWA/Post-AISTATS-Experiments/models/cifar_multi_class_lwa_64dim_lambda_opt_minus_one_mu",

  "_comment": "===== TRAINING CONFIGURATION =====",
  "tf_random_seed": 451760341,
  "np_random_seed": 216105420,
  "random_seed": 4557077,
  "max_num_training_steps": 100000, #100000, #30000,
  "num_output_steps": 1000,
  "num_summary_steps": 1000,
  "num_checkpoint_steps": 1000,
  "training_batch_size": 200, #50,

  "_comment": "===== EVAL CONFIGURATION =====",
  "num_eval_examples": 10000,
  "eval_batch_size": 200,
  "eval_checkpoint_steps": 3000,
  "eval_on_cpu": True,

  "_comment": "=====ADVERSARIAL EXAMPLES CONFIGURATION=====",
  "epsilon": 0.3,
  "k": 40,
  "a": 0.01,
  "random_start": True,
  "loss_func": "xent",
  "store_adv_path": "attack.npy"
  }

  return config



