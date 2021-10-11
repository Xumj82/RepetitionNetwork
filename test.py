from tensorflow._api.v2 import data
from tensorflow.python.ops.gen_array_ops import shape
from RepNet import ResnetPeriodEstimator
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from Dataset import CombinedDataset,SyntheticDataset

import matplotlib.pyplot as plt

# 测试用脚本

# train_dataset = CombinedDataset('./data/trainvids/','./countix/countix_train.csv').take(1)
# test_dataset = CombinedDataset('./data/testvids/','./countix/countix_test.csv').take(1)
# shared_conv = ResnetPeriodEstimator()
# shared_conv.build_graph((64,112,112,3)).summary()



# dataset = SyntheticDataset('./data/synthvids/',10).take(1)
# for i in dataset:
#     p = i
logits = [[1.0, 0.0, 0.0], [0.0, 0.8, 0.2]]
labels = [[1.0, 0.0, 0.0], [0.0, 0.8, 0.2]]
t = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
print(t)
# net(tf.random.uniform((1, 64, 112, 112, 3)))
# net.summary()

# for x,y in dataset:
#     input = x
#     outputs = net(input)
#     for out in outputs:
#         print(out.shape)




