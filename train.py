import tensorflow as tf
from tensorflow.keras.utils import plot_model
from Dataset import CombinedDataset,SyntheticDataset

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
import numpy as np

from RepNet import ResnetPeriodEstimator

import time
import datetime
import os

from tqdm import tqdm

# 分类预测损失函数（64*32） 
cce = tf.keras.losses.CategoricalCrossentropy()
# 回归预测损失函数（64*1）
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# classification metric on train data
train_cls_acc_metric = keras.metrics.CategoricalAccuracy()
train_cls_loss = tf.keras.metrics.Mean('train_cls_loss', dtype=tf.float32)

# regression metric on train data
train_reg_acc_metric = keras.metrics.BinaryAccuracy()
train_reg_loss = tf.keras.metrics.Mean('train_reg_loss', dtype=tf.float32)

# classification metric on test data
test_cls_acc_metric = keras.metrics.CategoricalAccuracy()
test_cls_loss = tf.keras.metrics.Mean('test_cls_loss', dtype=tf.float32)

# regression metric on test data
test_reg_acc_metric = keras.metrics.BinaryAccuracy()
test_reg_loss = tf.keras.metrics.Mean('test_reg_loss', dtype=tf.float32)

# log file settings
# tensorborad命令 ： tensorboard --logdir logs/gradient_tape
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

#check point
ckpt_path = 'chk_point/'+current_time

# 根据y计算分类矩阵，考虑到y是每一帧图像所预测到的重复动作的周期（1 ~ 32） 使用one-hot编码得到 64*32 矩阵
def get_periodicity(y):   
    y = tf.round(y)-1
    y = tf.squeeze(y, axis=-1, name=None).numpy()
    periodicity_mitrixs  = tf.one_hot(y, 32,
           on_value=1.0, off_value=0.0,
           axis=-1)

    return periodicity_mitrixs

# 根据y计算回归矩阵，y=0 则为0， y>0 则为1
def get_with_in_period(y):
    y = tf.where(tf.math.greater(y, 1), 1, y)
    return y

def train_step(model, optimizer, x_train, y_train):
    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation.
    with tf.GradientTape() as tape:
        # Get y1(one-hot classification) and y2(regression) by y_train
        y1 = get_periodicity(y_train)
        y2 = get_with_in_period(y_train)
        # Run the forward pass of the layer.
        # The operations that the layer applies
        # to its inputs are going to be recorded
        # on the GradientTape.
        y1pred, y2pred, sim = model(x_train)

        loss1 = cce(y1, y1pred)
        loss2 = bce(y2, y2pred)
        loss_value = loss1 + 5*loss2 # 不确定是否用5， 但总体来说loss1是loss2的5倍
        
    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_value, model.trainable_weights)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # Calculate accuracy and loss metric
    train_cls_acc_metric.update_state(y1, y1pred)
    train_reg_acc_metric.update_state(y2, y2pred)
    train_cls_loss.update_state(loss1)
    train_reg_loss.update_state(loss2)

    return sim

def test_step(model, x_test, y_test):

    y1_test = get_periodicity(y_test)
    y2_test = get_with_in_period(y_test)

    y1_pred_test, y2_pred_test, sim_val = model(x_test, training=False)

    loss1_test = cce(y1_test, y1_pred_test)
    loss2_test = bce(y2_test, y2_pred_test)

    # Update val metrics
    test_cls_acc_metric.update_state(y1_test, y1_pred_test)
    test_reg_acc_metric.update_state(y1_test, y2_pred_test)
    test_cls_loss.update_state(loss1_test)
    test_reg_loss.update_state(loss2_test)

    return sim_val

def start_train(
    epochs = 100,
    batch_size=2, 
    train_sample_number = 100,
    val_sample_number = 10,
    ckpt_path = None
    ):
    model = ResnetPeriodEstimator()
    if ckpt_path:
        model.load_weights(ckpt_path)
    # Instantiate an optimizer.
    optimizer = keras.optimizers.Adam()
    # Prepare the training dataset.
    train_dataset = CombinedDataset('./data/trainvids/','./countix/countix_train.csv').take(train_sample_number).batch(batch_size)
    # Prepare the validation dataset.
    val_dataset = CombinedDataset('./data/testvids/','./countix/countix_test.csv').take(val_sample_number).batch(batch_size)

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch+1,))
        # tf.summary.trace_on(graph=True)
        # Iterate over the batches of the dataset.
        with tqdm(total=train_sample_number+val_sample_number) as pbar:
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                sim_img = train_step(model, optimizer, x_batch_train, y_batch_train)
                pbar.update(batch_size)
        
            with train_summary_writer.as_default():
                sim_img = sim_img * 255
                tf.summary.image("train_similarity", sim_img, step=epoch)
                tf.summary.scalar('cls_accuracy', train_cls_acc_metric.result(), step=epoch)
                tf.summary.scalar('reg_accuracy', train_reg_acc_metric.result(), step=epoch)
                tf.summary.scalar('cls_loss', train_cls_loss.result(), step=epoch)
                tf.summary.scalar('reg_loss', train_reg_loss.result(), step=epoch)

            # Run a validation loop at the end of each epoch.        
            for x_batch_val, y_batch_val in val_dataset:
                sim_img_test = test_step(model, x_batch_val, y_batch_val)
                pbar.update(batch_size)

            with test_summary_writer.as_default():
                sim_img_test = sim_img_test*255
                tf.summary.image("test_similarity", sim_img_test, step=epoch)
                tf.summary.scalar('cls_accuracy', test_cls_acc_metric.result(), step=epoch)
                tf.summary.scalar('reg_accuracy', test_reg_acc_metric.result(), step=epoch)
                tf.summary.scalar('cls_loss', test_cls_loss.result(), step=epoch)
                tf.summary.scalar('reg_loss', test_reg_loss.result(), step=epoch)
        
        #save check point
        model.save_weights(ckpt_path,overwrite=True)
    # save model
    # module_no_signatures_path = os.path.join('./saved_model/', 'module_no_signatures')
    # print('Saving model...')
    # tf.saved_model.save(model, module_no_signatures_path)

start_train(epochs=100, batch_size=2, train_sample_number=500, val_sample_number=20)

