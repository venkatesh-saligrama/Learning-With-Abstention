
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import math
from datetime import datetime
import json
import shutil
from timeit import default_timer as timer

from termcolor import colored
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import pickle
import config

import sys
version = sys.version_info

classes = list(range(0,10))


config = config.get_config()
mus = np.linspace(0.1,3,30)
print('Config = ', config)
print('Mus = ', mus)

assert(1==2)

data = np.load('/home/anilkag/code/rnn_results/aditya/CIFAR-10/std_ce_64_dim_ft.npz', allow_pickle=True,)
test_X, test_Y = data['test_embd'], data['test_Y']
val_X, val_Y = data['val_embd'], data['val_Y']
train_X, train_Y = data['train_embd'], data['train_Y']

print('shapes x_train, y_train', train_X.shape, train_Y.shape)
print('shapes x_val, y_val', val_X.shape, val_Y.shape)
print('shapes x_test, y_test', test_X.shape, test_Y.shape)
    
print('\n\nnp.unique(train_Y) = ', np.unique(train_Y))


