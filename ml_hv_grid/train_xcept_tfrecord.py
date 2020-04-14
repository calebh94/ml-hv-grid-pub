"""
train_xcept.py

Train the Xception network to classify HV towers and substations

Modified on: 3/24/2020 by Caleb Harris (caleb.harris94@gatech.edu)
"""
import os
from os import path as op
from functools import partial
from datetime import datetime as dt
import pickle
import pprint
import numpy as np

import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Nadam
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.applications.xception import Xception, preprocess_input as xcept_preproc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, TensorBoard,
                             ReduceLROnPlateau)

from hyperopt import fmin, Trials, STATUS_OK, tpe
import yaml

from utils import print_start_details, print_end_details
from utils_training import ClasswisePerformance
from config import (get_params, tboard_dir, ckpt_dir,
                    data_dir,
                    model_params as MP, train_params as TP,
                    # train_params as DF)
                    )


def get_optimizer(opt_params, lr):
    """Helper to get optimizer from text params

    Parameters
    ----------
    opt_params: dict
        Dictionary containing optimization function name and learning rate decay
    lr:  float
        Initial learning rate

    Return
    ------
    opt_function: Keras optimizer
    """

    if opt_params['opt_func'] == 'sgd':
        return SGD(lr=lr, momentum=opt_params['momentum'])
    elif opt_params['opt_func'] == 'adam':
        return Adam(lr=lr)
    elif opt_params['opt_func'] == 'rmsprop':
        return RMSprop(lr=lr)
    elif opt_params['opt_func'] == 'nadam':
        return Nadam(lr=lr)
    # elif opt_params['opt_func'] == 'powersign':
    #     # from tensorflow.contrib.opt.python.training import sign_decay as sd
    #     from tensorflow.python.training import sign_decay as sd
    #     d_steps = opt_params['pwr_sign_decay_steps']
    #     # Define the decay function (if specified)
    #     if opt_params['pwr_sign_decay_func'] == 'lin':
    #         decay_func = sd.get_linear_decay_fn(d_steps)
    #     elif opt_params['pwr_sign_decay_func'] == 'cos':
    #         decay_func = sd.get_consine_decay_fn(d_steps)
    #     elif opt_params['pwr_sign_decay_func'] == 'res':
    #         decay_func = sd.get_restart_decay_fn(d_steps,
    #                                              num_periods=opt_params['pwr_sign_decay_periods'])
    #     elif opt_params['decay_func'] is None:
    #         decay_func = None
    #     else:
    #         raise ValueError('decay function not specified correctly')
    #
    #     # Use decay function in TF optimizer
    #     return Optimizer(PowerSignOptimizer(learning_rate=lr,
    #                                           sign_decay_fn=decay_func))
    else:
        raise ValueError


def xcept_net(params):
    """Train the Xception network

    Parmeters:
    ----------
    params: dict
        Parameters returned from config.get_params() for hyperopt

    Returns:
    --------
    result_dict: dict
        Results of model training for hyperopt.
    """
    
    K.clear_session()  # Remove any existing graphs
    mst_str = dt.now().strftime("%m%d_%H%M%S")

    print('\n' + '=' * 40 + '\nStarting model at {}'.format(mst_str))
    print('Model # %s' % len(trials))
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(params)

    ######################
    # Paths and Callbacks
    ######################
    ckpt_fpath = op.join(ckpt_dir, mst_str + '_L{val_loss:.2f}_E{epoch:02d}_weights.h5')
    # ckpt_fpath = op.join(ckpt_dir, mst_str + '_E{epoch:02d}_weights.h5')
    tboard_model_dir = op.join(tboard_dir, mst_str)
    tboard_model_dir = tboard_dir + '\\' + mst_str  # TF Bug where this much use '\\' as slashes!
    # 'imgs\\DC_imgs\\classify\\tensorboard'
    if op.exists(tboard_model_dir) == False:
        os.mkdir(tboard_model_dir)
        os.mkdir(os.path.join(tboard_model_dir, 'train'))

    print('Creating validation generator.')

    # val_iter = val_gen.flow_from_directory(
    #     directory=op.join(data_dir, 'val'), target_size=TP['img_size'][0:2], batch_size=TP['batch_size'], shuffle=False,  # Helps maintain consistency in testing phase
    # )
    # val_iter.reset()

    callbacks_phase1 = [TensorBoard(log_dir=tboard_model_dir, histogram_freq=0,
                                    write_grads=False, embeddings_freq=0,
                                       embeddings_layer_names=['dense_preoutput', 'dense_output'])]
    # Go watch, via command line $ tensorboard --logdir {path-to-directory} (i.e. tensorboard/..
    # Set callbacks to save performance to TB, modify learning rate, and stop poor trials early
    callbacks_phase2 = [
        TensorBoard(log_dir=tboard_model_dir, histogram_freq=0,
                    write_grads=False, embeddings_freq=0,
                    embeddings_layer_names=['dense_preoutput', 'dense_output']),
        # ClasswisePerformance(val_iter),
        #TODO: Fix ClasswisePerformance to have working
        ModelCheckpoint(ckpt_fpath, monitor='val_loss', mode='min',
                        save_weights_only=True, save_best_only=False),
        EarlyStopping(min_delta=TP['early_stopping_min_delta'],
                      patience=TP['early_stopping_patience'], verbose=1),
        ReduceLROnPlateau(min_delta=TP['reduce_lr_epsilon'],
                          patience=TP['reduce_lr_patience'], verbose=1)]

    #########################
    # Construct model
    #########################
    # Get the original xception model pre-initialized weights
    base_model = Xception(weights='imagenet',
                          include_top=False,  # Peel off top layer
                          input_shape=TP['img_size'],
                          pooling='avg')  # Global average pooling

    x = base_model.output  # Get final layer of base XCeption model

    # Add a fully-connected layer
    x = Dense(params['dense_size'], activation=params['dense_activation'],
              kernel_initializer=params['weight_init'],
              name='dense_preoutput')(x)
    if params['dropout_rate'] > 0:
        x = Dropout(rate=params['dropout_rate'])(x)

    # Finally, add output layer
    #TODO: fogure out where this should be given
    pred = Dense(TP['n_classes'],
                 activation=params['dense_activation'],
                 name='dense_output')(x)

    model = Model(inputs=base_model.input, outputs=pred)

    #####################
    # Save model details
    #####################
    model_yaml = model.to_yaml()
    save_template = op.join(ckpt_dir, mst_str + '_{}.{}')
    arch_fpath = save_template.format('arch', 'yaml')
    if not op.exists(arch_fpath):
        with open(arch_fpath.format('arch', 'yaml'), 'w') as yaml_file:
            yaml_file.write(model_yaml)

    # Save params to yaml file
    params_fpath = save_template.format('params', 'yaml')
    if not op.exists(params_fpath):
        with open(params_fpath, 'w') as yaml_file:
            yaml_file.write(yaml.dump(params))
            yaml_file.write(yaml.dump(TP))
            yaml_file.write(yaml.dump(MP))
            # yaml_file.write(yaml.dump(DF))

    ##########################
    # Train the new top layers
    ##########################
    # Train the top layers which we just added by setting all orig layers untrainable
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model (after setting non-trainable layers)
    model.compile(optimizer=get_optimizer(params['optimizer'],
                                          lr=params['lr_phase1']),
                  loss=params['loss'],
                  metrics=MP['metrics'])

    print('Phase 1, training near-output layer(s)')

    # params['steps_per_test_epo'] = int(np.ceil(total_test_images / TP['batch_size']) + 1)
    # params['steps_per_train_epo'] = int(np.ceil(total_train_images / TP['batch_size']) + 1)
    # params['max_queue_size'] = 4
    # params['workers'] = 1  #Important, right?
    # params['use_multiprocessing'] = False  #important, right?s
    # params['class_weight'] = TP['class_weight']  # From config...
    # params['batch_size'] = TP['batch_size']

    hist = model.fit(
        # train_gen.flow_from_directory(directory=op.join(data_dir, 'train'), target_size=TP['img_size'][0:2], batch_size=TP['batch_size'], shuffle=True),
        x=training_dataset,
        steps_per_epoch=int(TRAIN_SIZE / BATCH_SIZE),  #TODO: Need to check to make sure batching correcty...
        #TODO: otherwise, the entire dataset is not being used
        epochs=int(params['n_epo_phase1']),
        callbacks=callbacks_phase1,
        max_queue_size=MP['max_queue_size'],
        workers=MP['workers'],
        use_multiprocessing=MP['use_multiprocessing'],
        class_weight=TP['class_weight'],
        verbose=1
    )

    ###############################################
    # Train entire network to fine-tune performance
    ###############################################
    # Visualize layer names/indices to see how many layers to freeze:
    #print('Layer freeze cutoff = {}'.format(params['freeze_cutoff']))
    #for li, layer in enumerate(base_model.layers):
    #    print(li, layer.name)

    # Set all layers trainable
    for layer in model.layers:
        layer.trainable = True

    # Recompile model for second round of training
    model.compile(optimizer=get_optimizer(params['optimizer'],
                                          params['lr_phase2']),
                  loss=params['loss'],
                  metrics=MP['metrics'])

    print('\nPhase 2, training from layer {} on.'.format(params['freeze_cutoff']))
    # val_iter.reset()  # Reset for each model so it's consistent; ideally should reset every epoch

    hist = model.fit(
        x=training_dataset,
        steps_per_epoch=int(TRAIN_SIZE / BATCH_SIZE),
        epochs=int(params['n_epo_phase2']),
        max_queue_size=MP['max_queue_size'],
        workers=MP['workers'],
        use_multiprocessing=MP['use_multiprocessing'],
        validation_data=eval_dataset,
        validation_steps=EVAL_SIZE,
        callbacks=callbacks_phase2,
        class_weight=TP['class_weight'],
        verbose=1)

    # Return best of last validation accuracies
    check_ind = -1 * (TP['early_stopping_patience'] + 1)
    # result_dict = dict(loss=np.min(hist.history['val_loss'][check_ind:]),
    #                    status=STATUS_OK)
    result_dict = dict(loss=np.min(hist.history['val_loss'][check_ind:]),
                       status=STATUS_OK)
    #TODO: Make changes to ensure that validation loss is the prime value...

    return result_dict


if __name__ == '__main__':
    start_time = dt.now()
    print_start_details(start_time)

    # ###################################
    # # Calculate number of train/test images
    # ###################################
    # total_test_images = 0
    # total_train_images = 0
    # # Print out how many images are available for train/test
    # # for fold in ['train', 'test']:
    # for fold in ['train','val','test']:
    #     # for sub_fold in ['negatives', 'towers', 'substations']:
    #     for sub_fold in ['Pole', 'Nopole']:
    #         temp_img_dir = op.join(data_dir, fold, sub_fold)
    #         n_fnames = len([fname for fname in os.listdir(temp_img_dir)
    #                         if op.splitext(fname)[1] in ['.png', '.jpg']])
    #         print('For {}ing, found {} {} images'.format(fold, n_fnames, sub_fold))
    #
    #         if fold == 'test':
    #             total_test_images += n_fnames
    #         elif fold == 'train':
    #             total_train_images += n_fnames
    # if TP['steps_per_test_epo'] is None:
    #     TP['steps_per_test_epo'] = int(np.ceil(total_test_images /
    #                                            DF['flow_from_dir']['batch_size']) + 1)
    # TP['steps_per_test_epo'] = int(np.ceil(total_test_images / TP['batch_size']) + 1)
    # TP['steps_per_train_epo'] = int(np.ceil(total_train_images / TP['batch_size']) / 4 + 1)  #TODO: fix, added /4 to reduce...



    ###################################
    # Set up generators
    ###################################
    train_gen = ImageDataGenerator(preprocessing_function=xcept_preproc,
                                   # **DF['image_data_generator'])
                                   )
    #TODO: Add imagedatagenerator params...such as flip, rotateoin, normalization, etc.
    test_gen = ImageDataGenerator(preprocessing_function=xcept_preproc)

    val_gen = ImageDataGenerator(preprocessing_function=xcept_preproc)

    ###################################
    # Get the Dataset!
    ###################################
    # tfrecord filenames
    tfrecord_train = ['C:\\Users\\harri\\Downloads\\Gabriel_DC_poles_training_patches_g0.tfrecord.gz']
    tfrecord_eval = ['C:\\Users\\harri\\Downloads\\Gabriel_DC_poles_eval_patches_g0.tfrecord.gz']

    # Define variables needed for functions
    opticalBands = ['b1', 'b2', 'b3']
    BANDS = opticalBands
    RESPONSE = 'pole'
    FEATURES = BANDS + [RESPONSE]
    KERNEL_SIZE = 256
    KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]
    COLUMNS = [
        tf.io.FixedLenFeature(shape=KERNEL_SHAPE, dtype=tf.float32) for k in FEATURES
    ]
    FEATURES_DICT = dict(zip(FEATURES, COLUMNS))

    # Parameters to define
    BATCH_SIZE = TP['batch_size']
    BUFFER_SIZE = 0
    TRAIN_SIZE = 2500  ##############
    EVAL_SIZE = 1600  ##############


    # Define functions from EE examples
    def parse_tfrecord(example_proto):
        """The parsing function.
        Read a serialized example into the structure defined by FEATURES_DICT.
        Args:
          example_proto: a serialized Example.
        Returns:
          A dictionary of tensors, keyed by feature name.
        """
        return tf.io.parse_single_example(example_proto, FEATURES_DICT)


    def to_tuple(inputs):
        """Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
        Turn the tensors returned by parse_tfrecord into a stack in HWC shape.
        Args:
          inputs: A dictionary of tensors, keyed by feature name.
        Returns:
          A dtuple of (inputs, outputs).
        """
        inputsList = [inputs.get(key) for key in FEATURES]
        stacked = tf.stack(inputsList, axis=0)
        # Convert from CHW to HWC
        stacked = tf.transpose(stacked, [1, 2, 0])
        return stacked[:, :, :len(BANDS)], stacked[:, :, len(BANDS):]


    def to_tuple_tile_classification(inputs):
        """Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
        Turn the tensors returned by parse_tfrecord into a stack in HWC shape.
        Args:
          inputs: A dictionary of tensors, keyed by feature name.
        Returns:
          A dtuple of (inputs, outputs).
        """
        inputsList = [inputs.get(key) for key in FEATURES]
        stacked = tf.stack(inputsList, axis=0)
        # Convert from CHW to HWC
        stacked = tf.transpose(stacked, [1, 2, 0])

        # Hot encode the labels
        label = tf.reduce_max(stacked[:, :, len(BANDS):])
        indices = [0, 1]
        depth = 2
        labels = tf.one_hot(indices, depth)[int(label)]
        # Scale image data to between -1 and 1
        imgs = stacked[:, :, :len(BANDS)]
        imgs /= 127.5
        imgs -= 1.
        # return stacked[:,:,:len(BANDS)], stacked[:,:,len(BANDS):]
        # return stacked[:, :, :len(BANDS)], [tf.reduce_max(stacked[:, :, len(BANDS):])]
        return imgs, labels

    def get_dataset_orig(files):
        """Function to read, parse and format to tuple a set of input tfrecord files.
        Get all the files matching the pattern, parse and convert to tuple.
        Args:
          pattern: A file pattern to match in a Cloud Storage bucket.
        Returns:
          A tf.data.Dataset
        """
        # glob = tf.gfile.Glob(pattern)
        dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
        dataset = dataset.map(parse_tfrecord, num_parallel_calls=5)
        # test = to_tuple_tile_classification(dataset.take(1))
        dataset = dataset.map(to_tuple, num_parallel_calls=5)
        return dataset


    def get_dataset(files):
        """Function to read, parse and format to tuple a set of input tfrecord files.
        Get all the files matching the pattern, parse and convert to tuple.
        Args:
          pattern: A file pattern to match in a Cloud Storage bucket.
        Returns:
          A tf.data.Dataset
        """
        # glob = tf.gfile.Glob(pattern)
        dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
        dataset = dataset.map(parse_tfrecord, num_parallel_calls=5)
        # test = to_tuple_tile_classification(dataset.take(1))
        dataset = dataset.map(to_tuple_tile_classification, num_parallel_calls=5)
        return dataset

    def get_training_dataset_orig(files):
        """Get the preprocessed training dataset
      Returns:
        A tf.data.Dataset of training data.
      """
        # glob = 'gs://' + BUCKET + '/' + FOLDER + '/' + TRAINING_BASE + '*'
        dataset = get_dataset_orig(files)
        dataset = dataset.batch(BATCH_SIZE).repeat()
        return dataset

    def get_training_dataset(files):
        """Get the preprocessed training dataset
      Returns:
        A tf.data.Dataset of training data.
      """
        # glob = 'gs://' + BUCKET + '/' + FOLDER + '/' + TRAINING_BASE + '*'
        dataset = get_dataset(files)
        dataset = dataset.batch(BATCH_SIZE).repeat()
        return dataset


    # training_dataset_orig = get_training_dataset_orig(tfrecord_train)

    training_dataset = get_training_dataset(tfrecord_train)
    eval_dataset = get_training_dataset(tfrecord_eval)

    ############################################################
    # Run training with hyperparam optimization (using hyperopt)
    ############################################################
    #TODO: Distributed training (https://blog.goodaudience.com/on-using-hyperopt-advanced-machine-learning-a2dde2ccece7)
    trials = Trials()
    algo = partial(tpe.suggest, n_startup_jobs=TP['n_rand_hp_iters'])
    argmin = fmin(xcept_net, space=get_params(MP, TP), algo=algo,
                  max_evals=TP['n_total_hp_iters'], trials=trials)

    end_time = dt.now()
    print_end_details(start_time, end_time)
    print("Evalutation of best performing model:")
    print(trials.best_trial['result']['loss'])
    #TODO: Change to printing Accuracy?  Or need to print precision and accuracy and loss?+

    start_string = start_time.strftime("%m-%d-%y_%H-%M-%S")
    with open(op.join(ckpt_dir, 'trials_{}.pkl'.format(start_string)), "wb") as pkl_file:
        pickle.dump(trials, pkl_file)
