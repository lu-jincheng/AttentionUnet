import tensorflow as tf
import numpy as np
import os
import os.path
import glob
from PIL import Image


class TFrecord_Create_For_Unet():

    def __init__(self, train_test, dataset, img_folder, label_names, img_type, tf_record_pre_fix, nx, ny):

        self.train_test = train_test
        self.dataset = dataset
        self.img_folder = img_folder
        self.img_type = img_type
        self.label_names = label_names
        self.a_min = -np.inf
        self.a_max = np.inf
        self.nx = nx
        self.ny = ny
        print(self.label_names)
        files_examples = [name for name in self.dataset if
                          all(label not in name for label in self.label_names)]
        files_examples.sort()

        files_gts1 = [name for name in self.dataset if self.label_names[0] in name]
        files_gts1.sort()

        files_gts2 = [name for name in self.dataset if self.label_names[1] in name]
        files_gts2.sort()

        files_gts3 = [name for name in self.dataset if self.label_names[2] in name]
        files_gts3.sort()

        print('original images: ', len(files_examples))
        print('ground truth images: ', len(files_gts1) + len(files_gts2) + len(files_gts3))

        self.image_count = len(files_examples)
        _examples = np.asarray(files_examples)
        _gts1 = np.asarray(files_gts1)
        _gts2 = np.asarray(files_gts2)
        _gts3 = np.asarray(files_gts3)

        output_directory = os.path.join(os.getcwd(), 'unet_tfrecord')

        if not os.path.exists(output_directory) or os.path.isfile(output_directory):
            os.makedirs(output_directory)

        filename = os.path.join(output_directory, tf_record_pre_fix + '_{}.tfrecords'.format(train_test))

        writer = tf.python_io.TFRecordWriter(filename)

        for image, gt1, gt2, gt3 in zip(_examples, _gts1, _gts2, _gts3):
            image = Image.open(image).resize((self.ny, self.nx))
            image = np.array(image, np.float32)
            image = self._process_data(image)
            image_raw = image.tostring()

            gt1 = np.array(Image.open(gt1).convert("L").resize((self.ny, self.nx)), np.bool)
            gt1 = self._process_labels(gt1)
            gt2 = np.array(Image.open(gt2).convert("L").resize((self.ny, self.nx)), np.bool)
            gt2 = self._process_labels(gt2)
            gt3 = np.array(Image.open(gt3).convert("L").resize((self.ny, self.nx)), np.bool)
            gt3 = self._process_labels(gt3)
            gt = self._integrate_labels(gt1, gt2, gt3)

            gt_raw = gt.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': self._bytes_feature(image_raw),  # string
                'gt_raw': self._bytes_feature(gt_raw),  # string
            }))
            writer.write(example.SerializeToString())

        writer.close()
        print("Tfrecord generation finished")

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _process_data(self, data):
        data = data[:, :, 0]
        data = data / 255.
        return data

    def _process_labels(self, label):
        nx = label.shape[0]
        ny = label.shape[1]
        labels = np.zeros((nx, ny, 2), dtype=np.float32)
        labels[..., 1] = label
        labels[..., 0] = ~label
        return labels[..., 1]

    def _integrate_labels(self, label1, label2, label3):
        nx = label1.shape[0]
        ny = label1.shape[1]
        labels = np.zeros((nx, ny, 4), dtype=np.float32)
        for i in range(nx):
            for j in range(ny):
                if label1[i, j] == 1:
                    labels[i, j, 1] = 1
                elif label2[i, j] == 1:
                    labels[i, j, 2] = 1
                elif label3[i, j] == 1:
                    labels[i, j, 3] = 1
                else:
                    labels[i, j, 0] = 1
        return labels
