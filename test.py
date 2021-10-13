from tensorflow._api.v2 import data
from tensorflow.python.ops.gen_array_ops import shape
from DatasetGenerator import video_example
from RepNet import ResnetPeriodEstimator
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from Dataset import CombinedDataset,SyntheticDataset,TrainDataset

import matplotlib.pyplot as plt

# 测试用脚本

vidoe_feature_description = {
    'video': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, vidoe_feature_description)

train_dataset = TrainDataset(path='./data/train.tfrecords').take(1)

for video_features in train_dataset:
    # video_raw = video_features['video'].numpy()
    t = video_features







