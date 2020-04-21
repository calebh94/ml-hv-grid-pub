"""
config.py

List some configuration parameters for training model

Modified by: Caleb Harris (caleb.harris94@gatech.edu)
on:          4/10/2020
"""

import os
from os import path as op
from hyperopt import hp

# Set filepaths pointing to data that will be used in training
data_dir = 'imgs//DC_imgs//classify_fixes'
test_data_dir = op.join(data_dir, 'test')
ckpt_dir = op.join(data_dir, 'models')
tboard_dir = op.join(data_dir, 'tensorboard')
tboard_dir = tboard_dir.replace('//','\\')  # Required becuase of a bug in tensorboard code
# tboard_dir = 'imgs\\DC_imgs\\classify_fixes\\tensorboard'

# Removed cloud computation options
cloud_comp = False
#TODO: Setup process for training on PACE

preds_dir = op.join(data_dir,  'preds')
plot_dir = op.join(data_dir, 'plots')

if not op.isdir(ckpt_dir):
    os.mkdir(ckpt_dir)
if not op.isdir(tboard_dir):
    os.mkdir(tboard_dir)

# Model parameters modified for TF 2.0, and to remove unneccessary selections
model_params = dict(loss=['categorical_crossentropy'],
                    # loss=['binary_crossentropy'],
                    optimizer=[dict(opt_func='adam'),
                               dict(opt_func='rmsprop')],
                    lr_phase1=[1e-4, 1e-3],  # learning rate for phase 1 (output layer only)
                    lr_phase2=[1e-5, 5e-4],  # learning rate for phase 2 (all layers beyond freeze_cutoff)
                    weight_init=['glorot_uniform'],
                    metrics=['accuracy'],
                    # metrics=['binary_accuracy'], # Using binary accuracy while there is a binary classification (better)
                    # https://www.tensorflow.org/api_docs/python/tf/keras/metrics/BinaryAccuracy
                    # Blocks organized in 10s, 66, 76, 86, etc.
                    freeze_cutoff=[0],  # Layer below which no training/updating occurs on weights
                    dense_size=[128, 256, 512],  # Number of nodes in 2nd to final layer
                    dense_activation=['relu', 'elu'],
                    dropout_rate=[1e-3, 0.2, 0.5],  # Dropout in final layer
                    max_queue_size=4,
                    workers=1,  # Single GPU, no multiprocessing
                    use_multiprocessing=False)


train_params = dict(n_rand_hp_iters=3,
                    n_total_hp_iters=3,
                    n_epo_phase1=[2, 4],  # number of epochs training only top layer
                    # n_epo_phase1=[1, 4],  # number of epochs training only top layer
                    n_epo_phase2=[8, 14],  # number of epochs fine tuning whole model
                    # n_epo_phase2=[4, 8],  # number of epochs fine tuning whole model
                    batch_size=8,  # Want as large as GPU can handle, using batch-norm layers
                    prop_total_img_set=1.0,  # Proportion of total images per train epoch
                    img_size=(256, 256, 3),
                    early_stopping_patience=4,  # Number of iters w/out val_acc increase
                    early_stopping_min_delta=0.01,
                    reduce_lr_patience=2,  # Number of iters w/out val_acc increase
                    reduce_lr_epsilon=0.01,
                    # n_classes=2,
                    n_classes=5,
                    # class_weight={0: 0.63, 1: 2.48},  # flow from directory dataset
                    # class_weight={0: 0.50, 1: 10.00},  # tfrecord dataset
                    class_weight={0: 0.50, 1: 1.00, 2: 5.00, 3: 3.00, 4: 20.00},  # tfrecord dataset
                    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#class_weights
                    shuffle_seed=42)  # Seed for random number generator

pred_params = dict(model_time='0420_192732',
                   single_batch_size=4,  # Number of images seen by a single GPU
                   n_gpus=1,
                   deci_prec=4)  # Number of decimal places in prediction precision
pred_params.update(dict(model_arch_fname='{}_arch.yaml'.format(pred_params['model_time']),
                        model_params_fname='{}_params.yaml'.format(pred_params['model_time']),
                        model_weights_fname='{}_L3.15_E07_weights.h5'.format(pred_params['model_time'])))

# Removed download parameters since not using AWS dataset (maybe in future?)

######################
# Params for hyperopt
######################
def get_params(MP, TP):
    """Return hyperopt parameters"""
    return dict(
        optimizer=hp.choice('optimizer', MP['optimizer']),
        lr_phase1=hp.uniform('lr_phase1', MP['lr_phase1'][0], MP['lr_phase1'][1]),
        lr_phase2=hp.uniform('lr_phase2', MP['lr_phase2'][0], MP['lr_phase2'][1]),
        weight_init=hp.choice('weight_init', MP['weight_init']),
        freeze_cutoff=hp.choice('freeze_cutoff', MP['freeze_cutoff']),
        dropout_rate=hp.choice('dropout_rate', MP['dropout_rate']),
        dense_size=hp.choice('dense_size', MP['dense_size']),
        dense_activation=hp.choice('dense_activation', MP['dense_activation']),
        n_epo_phase1=hp.quniform('n_epo_phase1', TP['n_epo_phase1'][0], TP['n_epo_phase1'][1], 1),
        n_epo_phase2=hp.quniform('n_epo_phase2', TP['n_epo_phase2'][0], TP['n_epo_phase2'][1], 1),
        loss=hp.choice('loss', MP['loss']))
