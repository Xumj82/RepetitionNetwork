from contextlib import nullcontext
import tensorflow as tf
from tensorflow.python.ops.gen_parsing_ops import serialize_tensor
from Dataset import CombinedDataset,SyntheticDataset
from tqdm import tqdm

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  value = serialize_array(value)
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.  
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def video_example(video, label):
  feature = {
      'video':_bytes_feature(video),
      'label':_bytes_feature(label),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))

def prepare_data(sample_number=4332):
    train_dataset = CombinedDataset('./data/trainvids/','./countix/countix_train.csv').take(sample_number).prefetch(tf.data.AUTOTUNE)
    record_file = './data/train.tfrecords'

    with tf.io.TFRecordWriter(record_file) as writer:
        with tqdm(total=sample_number) as pbar:
            for x, y in train_dataset:
                tf_example = video_example(x, y)
                writer.write(tf_example.SerializeToString())
                pbar.update(1)

def prepare_test_data(sample_number=2554):
    test_dataset = CombinedDataset('./data/testvids/','./countix/countix_test.csv').take(sample_number).prefetch(tf.data.AUTOTUNE)
    record_file = './data/test.tfrecords'

    with tf.io.TFRecordWriter(record_file) as writer:
        with tqdm(total=sample_number) as pbar:
            for x, y in test_dataset:
                tf_example = video_example(x, y)
                writer.write(tf_example.SerializeToString())
                pbar.update(1)