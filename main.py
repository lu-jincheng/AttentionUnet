# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function, division, absolute_import, unicode_literals

import math
import os
import shutil
import copy
import numpy as np
from collections import OrderedDict
import logging
import tensorflow as tf
import util
from models import *
from scipy.ndimage import measurements
import argparse
import sys

# import segmentation_models as sm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', filename='train.log', filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


class Network(object):
    """
    A unet implementation
    :param channels: (optional) number of channels in the input image
    :param loss: (optional) name of the loss function. Default is 'cross_entropy'
    :param loss_kwargs: (optional) kwargs passed to the loss function. See Unet._get_loss for more options
    """

    def __init__(self, channels=3, n_class=4, net_type='Unet', loss="cross_entropy", loss_kwargs={}, **kwargs):  # graph

        tf.reset_default_graph()
        self.n_class = n_class
        self.x = tf.placeholder(tf.float32, shape=[None, None, None, channels], name="x")
        self.y = tf.placeholder(tf.float32, shape=[None, None, None, n_class], name="y")
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_probability")  # Dropout
        self.is_training = True
        self.img_pred = tf.placeholder(tf.float32, shape=[None, self.n_class], name="img_pred")
        self.img_y = tf.placeholder(tf.float32, shape=[None, self.n_class], name="img_y")
        if net_type == "Unet":
            logits, self.variables = create_unet(self.x, self.keep_prob, channels, is_training=self.is_training,
                                                 **kwargs)
        elif net_type == 'AGUnet':
            logits, self.variables = create_attention_gate_unet(self.x, self.keep_prob, channels,
                                                                is_training=self.is_training, **kwargs)
        elif net_type == "Resnet":
            logits, self.variables = create_resnet(self.x, self.keep_prob, channels, is_training=self.is_training,
                                                   **kwargs)
        elif net_type == "ResUnet":
            logits, self.variables = create_res_unet(self.x, self.keep_prob, channels, is_training=self.is_training,
                                                     **kwargs)
        elif net_type == "AGResUnet":
            logits, self.variables = create_attention_gate_res_unet(self.x, self.keep_prob, channels,
                                                                    is_training=self.is_training, **kwargs)
        elif net_type == "DAUnet":
            logits, self.variables = create_res_da_unet(self.x, self.keep_prob, channels,
                                                        is_training=self.is_training, attention_model='dual', **kwargs)
        elif net_type == "PAUnet":
            logits, self.variables = create_res_da_unet(self.x, self.keep_prob, channels, is_training=self.is_training,
                                                        attention_model='spatial', **kwargs)
        elif net_type == "CAUnet":
            logits, self.variables = create_res_da_unet(self.x, self.keep_prob, channels, is_training=self.is_training,
                                                        attention_model='channel-wise', **kwargs)
        elif net_type == "Densenet":
            logits, self.variables = create_densenet(self.x, self.keep_prob, channels, is_training=self.is_training,
                                                     **kwargs)
        elif net_type == "DenseUnet":
            logits, self.variables = create_dense_unet(self.x, self.keep_prob, channels, is_training=self.is_training,
                                                       **kwargs)
        elif net_type == 'SEResUnet':
            logits, self.variables = create_se_res_unet(self.x, self.keep_prob, channels, is_training=self.is_training,
                                                        **kwargs)
        else:
            print('Error network type')

        self.loss = self._get_loss(logits, loss, loss_kwargs)

        self.dice, self.iou = self._get_metrics()
        self.gradients_node = tf.gradients(self.loss, self.variables)

        with tf.name_scope("cross_entropy"):
            self.cross_entropy = cross_entropy(tf.reshape(self.y, [-1, n_class]),
                                               tf.reshape(logits, [-1, n_class]))

        with tf.name_scope("results"):
            self.predicter = logits
            self.predicter = tf.identity(self.predicter, name="predicter")
            self.correct_pred = tf.equal(self.predicter, self.y)  # logit: [batch, nx, ny, 1], self.y : [batch, nx, nx]
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.val_loss = tf.placeholder(tf.float32, shape=None, name="val_loss")
        self.val_dice = tf.placeholder(tf.float32, shape=None, name="val_dice")
        self.val_iou = tf.placeholder(tf.float32, shape=None, name="val_iou")
        tf.summary.scalar('train_loss', self.val_loss)
        tf.summary.scalar('val_dice', self.val_dice)
        tf.summary.scalar('val_IoU', self.val_iou)

    def _get_loss(self, logits, loss_name, loss_kwargs):
        with tf.name_scope("loss"):
            flat_logits = tf.reshape(logits, [-1, self.n_class])
            flat_labels = tf.reshape(self.y, [-1, self.n_class])

            if loss_name == "cross_entropy":
                class_weights = loss_kwargs.pop("class_weights", None)

                if class_weights is not None:
                    class_weights = tf.constant(np.array(class_weights, dtype=np.float32))

                    weight_map = tf.multiply(flat_labels, class_weights)
                    weight_map = tf.reduce_sum(weight_map, axis=1)

                    loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                          labels=flat_labels)
                    weighted_loss = tf.multiply(loss_map, weight_map)

                    loss = tf.reduce_mean(weighted_loss)

                else:
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                                     labels=flat_labels))
            elif loss_name == "dice_loss":
                smooth = 0.01
                score = 0
                for i in range(1, self.n_class):
                    intersection = tf.reduce_sum(flat_logits[:, i] * flat_labels[:, i])
                    score += (2 * intersection + smooth) / (
                            tf.reduce_sum(flat_labels[:, i]) + tf.reduce_sum(flat_logits[:, i]) + smooth)
                dice = score / 3
                loss = 1 - dice

            elif loss_name == "cross_entropy_log_dice":
                alpha = loss_kwargs.pop('alpha', None)
                smooth = 0.01
                score = 0
                for i in range(1, self.n_class):
                    intersection = tf.reduce_sum(flat_logits[:, i] * flat_labels[:, i])
                    score += (2 * intersection + smooth) / (
                            tf.reduce_sum(flat_labels[:, i]) + tf.reduce_sum(flat_logits[:, i]) + smooth)
                dice = score / 3
                cross_entropy_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits, labels=flat_labels))

                loss = (1 - alpha) * cross_entropy_loss - alpha * tf.log(dice)

            else:
                raise ValueError("Unknown loss function: " % loss_name)

            regularizer = loss_kwargs.pop("regularizer", None)
            if regularizer is not None:
                regularizers = sum([tf.nn.l2_loss(variable) for variable in self.variables])
                loss += (regularizer * regularizers)
            return loss

    def _get_metrics(self):
        with tf.name_scope('metrics'):
            dice_sum = 0
            iou_sum = 0
            smooth = 0.01
            dice = []
            iou = []
            for i in range(1, self.n_class):
                intersection = tf.reduce_sum(self.img_pred[:, i] * self.img_y[:, i])
                summation = tf.reduce_sum(self.img_y[:, i]) + tf.reduce_sum(self.img_pred[:, i])
                dice_i = (2 * intersection + smooth) / (summation + smooth)
                dice_sum += dice_i
                dice.append(dice_i)
                union = summation - intersection
                iou_i = (intersection + smooth) / (union + smooth)
                iou_sum += iou_i
                iou.append(iou_i)
            # mean_dice = dice_sum / (self.n_class - 1)
            # mean_iou = iou_sum / (self.n_class - 1)
            return dice, iou


class Trainer(object):
    """
    :param net: the unet instance to train
    :param data_provider: the data_provider instance to get data
    :param batch_size: size of training batch
    :param validation_batch_size: size of validation batch
    :param optimizer: (optional) name of the optimizer to use (momentum or adam)
    :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer
    """

    def __init__(self, net, data_provider, batch_size=1, validation_batch_size=4, optimizer="momentum", lr=0.05,
                 first_decay_step=4726, nx=256, ny=256, opt_kwargs={}):
        self.net = net
        self.data_provider = data_provider
        self.batch_size = batch_size
        self.validation_batch_size = validation_batch_size
        self.lr = lr
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.train_ops = None
        self.first_decay_step = first_decay_step
        self.nx = nx
        self.ny = ny

    def _get_train_ops(self):
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            if self.optimizer == "momentum":
                learning_rate = self.opt_kwargs.pop("learning_rate", self.lr)
                momentum = self.opt_kwargs.pop("momentum", 0.9)

                self.learning_rate_node = tf.train.cosine_decay_restarts(
                    learning_rate=learning_rate,
                    global_step=self.global_step,
                    first_decay_steps=self.first_decay_step,
                    t_mul=2.0,
                    m_mul=1.0,
                    alpha=1e-10,
                    name=None
                )
                tf.summary.scalar('lr', self.learning_rate_node)
                train_ops = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                       **self.opt_kwargs).minimize(self.net.loss,
                                                                                   global_step=self.global_step)
            elif self.optimizer == "adam":
                learning_rate = self.opt_kwargs.pop("learning_rate", self.lr)
                self.learning_rate_node = tf.Variable(learning_rate, name="learning_rate")
                tf.summary.scalar('lr', self.learning_rate_node)
                train_ops = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node,
                                                   **self.opt_kwargs).minimize(self.net.loss,
                                                                               global_step=self.global_step)
        return train_ops

    def _initialize(self, training_iters, output_path, restore, prediction_path):

        self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_node)]),
                                               name="norm_gradients")
        self.train_ops = self._get_train_ops(training_iters)
        self.merged = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        self.prediction_path = prediction_path
        abs_prediction_path = os.path.abspath(self.prediction_path)
        output_path = os.path.abspath(output_path)

        if not restore:
            logging.info("Removing '{:}'".format(abs_prediction_path))
            shutil.rmtree(abs_prediction_path, ignore_errors=True)
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)

        if not os.path.exists(abs_prediction_path):
            logging.info("Allocating '{:}'".format(abs_prediction_path))
            os.makedirs(abs_prediction_path)

        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)

        return init

    def train(self, output_path, log_path, training_iters=10, epochs=100, dropout=0.75, display_step=10,
              restore=True, prediction_path='prediction', test_size=23):

        save_path = os.path.join(output_path, "model.ckpt")
        tb_writer = tf.summary.FileWriter(log_path)
        if epochs == 0:
            return save_path
        init = self._initialize(training_iters, output_path, restore, prediction_path)
        train_init, test_init = self.data_provider.build_get_data()
        saver = tf.train.Saver(max_to_keep=1)
        with tf.Session() as sess:
            sess.run(init)
            sess.run(test_init)
            logging.info("Start optimization")
            best_dice = 0
            best_iou = 0
            best_epoch = 0
            for epoch in range(epochs):
                sess.run(train_init)
                total_loss = 0
                try:
                    for step in range((epoch * training_iters), ((epoch + 1) * training_iters)):
                        batch_x, batch_y = sess.run([self.data_provider.images, self.data_provider.gts])
                        # Run optimization op (backprop)
                        self.net.is_training = True
                        _, loss, lr, gradients = sess.run(
                            [self.train_ops, self.net.loss, self.learning_rate_node, self.net.gradients_node],
                            feed_dict={self.net.x: batch_x,
                                       self.net.y: batch_y,
                                       self.net.keep_prob: dropout,
                                       # self.net.is_training: True
                                       })
                        if step % display_step == 0:
                            logging.info(
                                "Iter {:}, Minibatch Training Loss= {:.4f}".format(self.global_step.eval(), loss))
                        total_loss += loss

                except tf.errors.OutOfRangeError:
                    print('OutOfRangeError')
                    pass
                self.output_epoch_stats(epoch, total_loss, training_iters, lr)
                self.net.is_training = False
                val_dice, val_iou = self.validate(sess, test_size, self.data_provider.images_test,
                                                  self.data_provider.gts_test, epoch)
                if best_dice <= val_dice:
                    best_dice = val_dice
                    best_iou = val_iou
                    best_epoch = epoch
                    saver.save(sess, save_path, write_meta_graph=True)
                summary, _l, _d, _i = sess.run([self.merged, self.net.val_loss, self.net.val_dice, self.net.val_iou],
                                               feed_dict={self.net.x: batch_x,
                                                          self.net.y: batch_y,
                                                          self.net.keep_prob: dropout,
                                                          # self.net.is_training: False,
                                                          self.net.val_loss: (total_loss / training_iters),
                                                          self.net.val_dice: val_dice,
                                                          self.net.val_iou: val_iou})
                tb_writer.add_summary(summary, epoch)
            logging.info("Optimization Finished!")
            logging.info(
                "Best epoch: {:.1f}, Best Dice: {:.4f}, Best IoU: {:.4f}".format(best_epoch, best_dice, best_iou))

    def validate(self, sess, test_size, x_provider, y_provider, n_epoch):
        self.net.is_training = False
        mean_dice = 0
        mean_iou = 0
        for val_index in range(test_size):
            val_x, val_y = sess.run([x_provider, y_provider])
            prediction, loss = sess.run([self.net.predicter, self.net.loss], feed_dict={self.net.x: val_x,
                                                                                        self.net.y: val_y,
                                                                                        })
            img_predict = prediction.reshape((self.nx, self.ny, 4))
            img_predict = self.remove_minor_cc(img_predict, 0.3, 4)
            img_y = val_y.reshape((self.nx * self.ny, 4))
            img_predict = img_predict.reshape((self.nx * self.ny, 4))
            dice, iou = sess.run([self.net.dice, self.net.iou], feed_dict={self.net.img_pred: img_predict,
                                                                           self.net.img_y: img_y})
            dice1 = dice[0]
            dice2 = dice[1]
            dice3 = dice[2]
            dice = (dice1 + dice2 + dice3) / 3
            iou1 = iou[0]
            iou2 = iou[1]
            iou3 = iou[2]
            iou = (iou1 + iou2 + iou3) / 3
            mean_dice += dice
            mean_iou += iou
            if val_index == 0:
                img = util.combine_img_prediction(val_x, img_y[:, 1], img_predict[:, 1], img_y[:, 2], img_predict[:, 2],
                                                  img_y[:, 3], img_predict[:, 3])
                util.save_image(img, "%s/%s.jpg" % (
                    self.prediction_path, "epoch_%s" % n_epoch + '_%s' % val_index + '_%s' % dice))

        mean_dice = mean_dice / test_size
        mean_iou = mean_iou / test_size
        logging.info("Validation Loss: {:.4f}, Dice: {:.4f}, IoU: {:.4f}".format(loss, mean_dice, mean_iou))
        return mean_dice, mean_iou

    def test(self, model_path, data_provider, test_size):
        mean_dice = 0
        mean_dice1 = 0
        mean_dice2 = 0
        mean_dice3 = 0
        mean_iou = 0
        mean_iou1 = 0
        mean_iou2 = 0
        mean_iou3 = 0
        self.net.is_training = False
        test_init = data_provider.build_get_data()

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(test_init)
            x_provider = data_provider.images_test
            y_provider = data_provider.gts_test
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
            for test_index in range(test_size):
                test_x, test_y = sess.run([x_provider, y_provider])
                prediction, loss = sess.run([self.net.predicter, self.net.loss], feed_dict={self.net.x: test_x,
                                                                                            self.net.y: test_y,
                                                                                            self.net.keep_prob: 1.,
                                                                                            })
                img_predict = prediction.reshape((256, 256, 4))
                img_predict = self.remove_minor_cc(img_predict, 0.3, 4)
                img_y = test_y.reshape((65536, 4))
                img_predict = img_predict.reshape((65536, 4))
                dice, iou = sess.run([self.net.dice, self.net.iou], feed_dict={self.net.img_pred: img_predict,
                                                                               self.net.img_y: img_y})
                dice1 = dice[0]
                dice2 = dice[1]
                dice3 = dice[2]
                dice = (dice1 + dice2 + dice3) / 3
                iou1 = iou[0]
                iou2 = iou[1]
                iou3 = iou[2]
                iou = (iou1 + iou2 + iou3) / 3
                mean_dice += dice
                mean_dice1 += dice1
                mean_dice2 += dice2
                mean_dice3 += dice3
                mean_iou += iou
                mean_iou1 += iou1
                mean_iou2 += iou2
                mean_iou3 += iou3
                img = util.combine_img_prediction(test_x, img_y[:, 1], img_predict[:, 1], img_y[:, 2],
                                                  img_predict[:, 2],
                                                  img_y[:, 3], img_predict[:, 3])
                util.save_image(img, "%s/%s.jpg" % (
                    self.prediction_path, "test_%s" % test_index + '_%s' % dice))
            mean_dice = mean_dice / test_size
            mean_iou = mean_iou / test_size
            mean_dice1 = mean_dice1 / test_size
            mean_dice2 = mean_dice2 / test_size
            mean_dice3 = mean_dice3 / test_size
            mean_iou1 = mean_iou1 / test_size
            mean_iou2 = mean_iou2 / test_size
            mean_iou3 = mean_iou3 / test_size
            logging.info("Test Loss: {:.4f}, Dice: {:.4f}, IoU: {:.4f}".format(loss, mean_dice, mean_iou))
            logging.info(
                "Dice1: {:.4f}, Dice2: {:.4f}, Dice3: {:.4f},IoU2: {:.4f}, IoU1: {:.4f}, IoU3: {:.4f}".format(
                    mean_dice1, mean_dice2, mean_dice3, mean_iou1, mean_iou2, mean_iou3))
            return mean_dice, mean_iou

    # Remove small connected components
    def remove_minor_cc(self, data, rej_ratio, n_class):
        """Remove small connected components refer to rejection ratio"""
        rem_vol = copy.deepcopy(data)
        for c in range(1, n_class):

            class_idx = data[:, :, c]
            class_idx[class_idx >= 0.5] = 1
            class_idx[class_idx < 0.5] = 0
            class_area = np.sum(class_idx)
            labeled_cc, num_cc = measurements.label(class_idx)
            for cc in range(1, num_cc + 1):
                single_cc = ((labeled_cc == cc) * 1)
                single_area = np.sum(single_cc)
                if single_area / (class_area * 1.0) < rej_ratio:
                    class_idx[labeled_cc == cc] = 0
            rem_vol[:, :, c] = class_idx

        return rem_vol

    def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
        logging.info(
            "Epoch {:}, Average loss: {:.4f}, learning rate: {:.6f}".format(epoch, (total_loss / training_iters), lr))

    def output_minibatch_stats(self, sess, batch_x, batch_y):
        self.net.is_training = False
        loss = sess.run(self.net.loss, feed_dict={self.net.x: batch_x,
                                                  self.net.y: batch_y,
                                                  self.net.keep_prob: 1.,
                                                  })
        logging.info(
            "Iter {:}, Minibatch Validation Loss= {:.4f}".format(self.global_step.eval(), loss))
