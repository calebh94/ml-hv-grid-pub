"""
test_xcept.py

Run testing on a trained Xception network that can classify HV towers and
substations
"""

import os
from os import path as op
from datetime import datetime as dt

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications.xception import preprocess_input as xcept_preproc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import multi_gpu_model
from sklearn.metrics import classification_report
from scipy.special import softmax

import yaml

from utils import print_start_details, print_end_details, load_model
from config import (ckpt_dir, data_dir, preds_dir, pred_params as pred_p, test_data_dir,
                    train_params as TP
                    # data_flow as DF)
                    )

import matplotlib.pyplot as plt

### Parameters!
debug = False

########################################
# Calculate number of test images
########################################
if debug:
    test_data_dir = op.join(data_dir, 'test_focus')
    print('Using test images in {}\n'.format(test_data_dir))

    total_test_images = 0
    # for sub_fold in ['negatives', 'towers', 'substations']:
    for sub_fold in ['Pole', 'Nopole']:
        temp_img_dir = op.join(test_data_dir, sub_fold)
        n_fnames = len([fname for fname in os.listdir(temp_img_dir)
                        if op.splitext(fname)[1] in ['.png', '.jpg']])
        print('For testing, found {} {} images'.format(n_fnames, sub_fold))

        total_test_images += n_fnames

# steps_per_test_epo = int(np.ceil(total_test_images /
#                                  TP['batch_size']) + 1)

# # Set up generator
# test_gen = ImageDataGenerator(preprocessing_function=xcept_preproc)

# print('\nCreating test generator.')
# test_iter = test_gen.flow_from_directory(
#     directory=test_data_dir, target_size=(256, 256), batch_size=1, shuffle=False)

# test_iter.reset()  # Reset for each model to ensure consistency


###################################
# Get the Dataset!
###################################
# tfrecord filenames
tfrecord_test = ['C:\\Users\\harri\\Downloads\\Gabriel_DC_poles_pred_patches_g0.tfrecord.gz']

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
BATCH_SIZE = 16
BUFFER_SIZE = 0
TEST_SIZE = 2000


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


def get_training_dataset(files):
    """Get the preprocessed training dataset
  Returns:
    A tf.data.Dataset of training data.
  """
    # glob = 'gs://' + BUCKET + '/' + FOLDER + '/' + TRAINING_BASE + '*'
    dataset = get_dataset(files)
    dataset = dataset.batch(BATCH_SIZE).repeat()
    return dataset


testing_dataset = get_training_dataset(tfrecord_test)

####################################
# Load model and params
####################################
print('Loading model.')
if pred_p['n_gpus'] > 1:
    # Load weights on CPU to avoid taking up GPU space
    with tf.device('/cpu:0'):
        template_model = load_model(op.join(ckpt_dir, pred_p['model_arch_fname']),
                                    op.join(ckpt_dir, pred_p['model_weights_fname']))
    parallel_model = multi_gpu_model(template_model, gpus=pred_p['n_gpus'])
else:
    template_model = load_model(op.join(ckpt_dir, pred_p['model_arch_fname']),
                                op.join(ckpt_dir, pred_p['model_weights_fname']))
    # template_model = load_model(op.join(ckpt_dir, pred_p['model_weights_fname']))
    parallel_model = template_model

# Turn off training. This is supposed to be faster (haven't seen this empirically though)
K.set_learning_phase = 0
for layer in template_model.layers:
    layer.trainable = False

# Load model parameters for printing
with open(op.join(ckpt_dir, pred_p['model_params_fname']), 'r') as f_model_params:
    params_yaml = f_model_params.read()
    model_params = yaml.load(params_yaml)
print('Loaded model: {}\n\twith params: {}, gpus: {}'.format(
    pred_p['model_arch_fname'], pred_p['model_weights_fname'], pred_p['n_gpus']))

#######################
# Run prediction
#######################
print('\nPredicting.')
start_time = dt.now()
print('Start time: ' + start_time.strftime('%d/%m %H:%M:%S'))

# y_true = test_iter.classes
# class_labels = list(test_iter.class_indices.keys())

# # Leave steps=None to predict entire sequence
# y_pred_probs = parallel_model.predict_generator(test_iter,
#                                                 steps=len(test_iter),
#                                                 workers=16,
#                                                 verbose=1)


if debug:
    des_test = 25 # make number with square root as int
else:
    des_test = TEST_SIZE

cnt = 0
y_true = []
for img, lbl in testing_dataset:
    if cnt >= des_test:
        break
    else:
        y_true.append(lbl)
        cnt=cnt+1

y_pred_probs = parallel_model.predict(testing_dataset,
                                                steps=des_test,
                                                workers=1,
                                                verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

if debug:
    def plot_image(i, predictions_array, true_label, img):
      predictions_array, true_label, img = predictions_array, true_label[i], img[i]
      plt.grid(False)
      plt.xticks([])
      plt.yticks([])
      # fix image!

      img = img.squeeze()
      img = (img + 1) * 127.5
      img = img.astype(int)
      plt.imshow(img, cmap=plt.cm.binary)

      predicted_label = np.argmax(predictions_array)
      if predicted_label == true_label:
        color = 'blue'
      else:
        color = 'red'

      plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                    100*np.max(softmax(predictions_array)),
                                    true_label),
                                    color=color)

    def plot_value_array(i, predictions_array, true_label):
      predictions_array, true_label = predictions_array, true_label[i]
      plt.grid(False)
      plt.xticks(range(2))
      plt.yticks([])
      thisplot = plt.bar(range(2), predictions_array, color="#777777")
      plt.ylim([0, 1])
      predicted_label = np.argmax(predictions_array)

      thisplot[predicted_label].set_color('red')
      thisplot[true_label].set_color('blue')

    # Plot the first X test images, their predicted labels, and the true labels.
    # Color correct predictions in blue and incorrect predictions in red.
    num_rows = int(np.sqrt(des_test))
    num_cols = int(np.sqrt(des_test))
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))

    img_array = []
    cnt=0
    for img, lbl in test_iter:
        img_array.append(img)
        cnt = cnt+1
        if cnt > num_images:
            break

    for i in range(num_images):
      plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
      plot_image(i, y_pred_probs[i], y_true, img_array)
      plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
      plot_value_array(i, y_pred_probs[i], y_true)
    plt.tight_layout()
    plt.show()

print(classification_report(y_true[0:des_test], y_pred))
# print(np.sum(y_pred))


#############
# End details
#############
end_time = dt.now()
print_end_details(start_time, end_time)
