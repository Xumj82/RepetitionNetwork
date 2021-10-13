from tensorflow._api.v2 import data
from tensorflow.python.ops.gen_array_ops import shape
from RepNet import ResnetPeriodEstimator
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from DatasetGenerator import prepare_train_data

# 测试用脚本

vidoe_feature_description = {
    'video': tf.io.FixedLenFeature([], tf.string),
    'periodicity': tf.io.FixedLenFeature([], tf.string),
    'with_in_period': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  example_message = tf.io.parse_single_example(example_proto, vidoe_feature_description)
  x = tf.io.parse_tensor(example_message['video'], out_type=tf.float64)
  y1 = tf.io.parse_tensor(example_message['periodicity'], out_type=tf.float64)
  y2 = tf.io.parse_tensor(example_message['with_in_period'], out_type=tf.float64)
  return x, y1, y2

train_dataset = tf.data.TFRecordDataset(filenames = './data/train.tfrecords').map(_parse_image_function)

for x,y1,y2 in train_dataset:
    # video_raw = video_features['video'].numpy()
    t = t

# prepare_train_data(10)







