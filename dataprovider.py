from __future__ import print_function, division, absolute_import, unicode_literals
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import tensorflow as tf
import aug


# Affine transformation
def augmentation(image,
                 random_noise_prob=0.00,
                 random_brightness_prob=0.5,
                 random_brightness_max_delta=0.9,  # [0, 1)
                 random_hue_prob=0.5,
                 random_hue_max_delta=0.5,  # [0, 0.5)]
                 random_saturation_prob=0.5,
                 random_saturation_lower=0.2,
                 random_saturation_upper=5.0,
                 ):
    if random_noise_prob != 0:
        image = aug.random_noise(image, prob_noise=random_noise_prob)

    if random_brightness_prob != 0:
        image = aug.random_brightness(
            image,
            prob_brightness=random_brightness_prob,
            max_delta=random_brightness_max_delta
        )

    if random_hue_prob != 0:
        image = aug.random_hue(
            image,
            prob_hue=random_hue_prob,
            max_delta=random_hue_max_delta
        )

    if random_saturation_prob != 0:
        image = aug.random_saturation(
            image,
            prob_saturation=random_saturation_prob,
            lower=random_saturation_lower,
            upper=random_saturation_upper
        )

    return image


class Tfrecord_ImageDataProvider():
    def __init__(self, train_tfrecord_path=None, test_tfrecord_path=None, n_class=4, channels=1, train_batch_size=4,
                 test_batch_size=16, nx=256, ny=256):
        self.train_tfrecord_path = train_tfrecord_path
        self.test_tfrecord_path = test_tfrecord_path
        self.n_class = n_class
        self.channels = channels
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.nx = nx
        self.ny = ny

    def build_get_data(self):
        self.train_dataset = tf.data.TFRecordDataset(self.train_tfrecord_path)
        self.train_dataset = self.train_dataset.map(lambda x: self._read_from_tfrecord(x)).repeat()
        self.train_dataset = self.train_dataset.batch(self.train_batch_size)
        # print(self.train_dataset.output_shapes)
        self.train_iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types,
                                                              self.train_dataset.output_shapes)
        self.images, self.gts = self.train_iterator.get_next()
        train_init = self.train_iterator.make_initializer(self.train_dataset)  # initializer for train_data

        self.test_dataset = tf.data.TFRecordDataset(self.test_tfrecord_path)
        self.test_dataset = self.test_dataset.map(lambda x: self._read_from_tfrecord(x)).repeat()
        self.test_dataset = self.test_dataset.batch(self.test_batch_size)
        self.test_iterator = tf.data.Iterator.from_structure(self.test_dataset.output_types,
                                                             self.test_dataset.output_shapes)

        self.images_test, self.gts_test = self.test_iterator.get_next()
        test_init = self.test_iterator.make_initializer(self.test_dataset)

        return train_init, test_init

    def _read_from_tfrecord(self, example_proto):
        tfrecord_features = tf.parse_single_example(example_proto,
                                                    features={
                                                        'image_raw': tf.FixedLenFeature([], tf.string),
                                                        'gt_raw': tf.FixedLenFeature([], tf.string),
                                                    }, name='features')

        image = tf.decode_raw(tfrecord_features['image_raw'], tf.float32)
        image = tf.reshape(image, [self.nx, self.ny, self.channels])
        #         image = augmentation(image)

        gt = tf.decode_raw(tfrecord_features['gt_raw'], tf.float32)
        # gt = tf.reshape(gt, [self.nx, self.ny])
        gt = tf.reshape(gt, [self.nx, self.ny, self.n_class])
        # print(np.shape(gt))
        return image, gt
