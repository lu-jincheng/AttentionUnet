from __future__ import print_function, division, absolute_import, unicode_literals
import tensorflow as tf
import numpy as np
from tflearn.layers.conv import global_avg_pool


def weight_variable(shape, stddev=0.1, name="weight"):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)


def weight_variable_devonc(shape, stddev=0.1, name="weight_devonc"):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)


def bias_variable(shape, name="bias"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def calc_down_weight_bias(layer, filter_size, features_root):
    features = 2 ** layer * features_root
    stddev = np.sqrt(2 / (filter_size ** 2 * features))
    with tf.name_scope("down_conv_{}".format(str(layer))):
        if layer == 0:
            w1 = weight_variable([filter_size, filter_size, 1, features], stddev, name="w1")
        else:
            w1 = weight_variable([filter_size, filter_size, features // 2, features], stddev, name="w1")

        w2 = weight_variable([filter_size, filter_size, features, features], stddev, name="w2")
        b1 = bias_variable([features], name="b1")
        b2 = bias_variable([features], name="b2")
        return w1, w2, b1, b2


def calc_up_weight_bias(layer, filter_size, pool_size, features_root):
    features = 2 ** (layer + 1) * features_root
    stddev = np.sqrt(2 / (filter_size ** 2 * features))
    with tf.name_scope("up_conv_{}".format(str(layer))):
        wd = weight_variable_devonc([pool_size, pool_size, features // 2, features], stddev, name="wd")
        bd = bias_variable([features // 2], name="bd")
        w1 = weight_variable([filter_size, filter_size, features, features // 2], stddev, name="w1")
        w2 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w2")
        b1 = bias_variable([features // 2], name="b1")
        b2 = bias_variable([features // 2], name="b2")
        return wd, bd, w1, w2, b1, b2


def calc_upblock_weight_bias(layer, filter_size, pool_size, features_root):
    features = 2 ** (layer + 1) * features_root
    stddev = np.sqrt(2 / (filter_size ** 2 * features))
    with tf.name_scope("up_conv_{}".format(str(layer))):
        wd = weight_variable_devonc([pool_size, pool_size, features // 2, features], stddev, name="wd")
        bd = bias_variable([features // 2], name="bd")
        w1 = weight_variable([filter_size, filter_size, features, features // 2], stddev, name="w1")
        w2 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w2")
        b1 = bias_variable([features // 2], name="b1")
        b2 = bias_variable([features // 2], name="b2")
        w = [w1, w2]
        b = [b1, b2]
        return wd, bd, w, b


def calc_nsc_upblock_weight_bias(layer, filter_size, pool_size, features_root):
    features = 2 ** (layer + 1) * features_root
    stddev = np.sqrt(2 / (filter_size ** 2 * features))
    with tf.name_scope("up_conv_{}".format(str(layer))):
        wd = weight_variable_devonc([pool_size, pool_size, features // 2, features], stddev, name="wd")
        bd = bias_variable([features // 2], name="bd")
        w1 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w1")
        w2 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w2")
        b1 = bias_variable([features // 2], name="b1")
        b2 = bias_variable([features // 2], name="b2")
        w = [w1, w2]
        b = [b1, b2]
        return wd, bd, w, b


def calc_att_weight_bias(layer, filter_size, features_root):
    features = 2 ** (layer + 1) * features_root
    stddev = np.sqrt(2 / (filter_size ** 2 * features))
    with tf.name_scope("up_conv_{}".format(str(layer))):
        wa1 = weight_variable([1, 1, features // 2, features // 2], stddev, name="wa1")
        wa2 = weight_variable([1, 1, features // 2, features // 2], stddev, name="wa2")
        wa3 = weight_variable([1, 1, features // 2, 1], stddev, name="wa3")
        ba1 = bias_variable([features // 2], name="ba1")
        ba2 = bias_variable([features // 2], name="ba2")
        ba3 = bias_variable([1], name="ba3")
        wa = [wa1, wa2, wa3]
        ba = [ba1, ba2, ba3]
        return wa, ba


def calc_au_up_weight_bias(layer, filter_size, pool_size, features_root):
    features = 2 ** (layer + 1) * features_root
    stddev = np.sqrt(2 / (filter_size ** 2 * features))
    with tf.name_scope("up_conv_{}".format(str(layer))):
        wd = weight_variable_devonc([pool_size, pool_size, features // 2, features], stddev, name="wd")
        bd = bias_variable([features // 2], name="bd")
        w1 = weight_variable([filter_size, filter_size, features, features // 2], stddev, name="w1")
        w2 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w2")
        b1 = bias_variable([features // 2], name="b1")
        b2 = bias_variable([features // 2], name="b2")
        wa1 = weight_variable([1, 1, features // 2, features // 2], stddev, name="wa1")
        wa2 = weight_variable([1, 1, features // 2, features // 2], stddev, name="wa2")
        wa3 = weight_variable([1, 1, features // 2, 1], stddev, name="wa3")
        ba1 = bias_variable([features // 2], name="ba1")
        ba2 = bias_variable([features // 2], name="ba2")
        ba3 = bias_variable([1], name="ba3")
        wa = [wa1, wa2, wa3]
        ba = [ba1, ba2, ba3]
        return wd, bd, w1, w2, b1, b2, wa, ba


def calc_res_down_weight_bias(layer, filter_size, features_root):
    features = 2 ** layer * features_root
    stddev = np.sqrt(2 / (filter_size ** 2 * features))
    with tf.name_scope("down_conv_{}".format(str(layer))):
        if layer == 0:
            w1 = weight_variable([filter_size, filter_size, 1, features], stddev, name="w1")
            wi1 = weight_variable([1, 1, 1, features], stddev, name="wi1")
        else:
            w1 = weight_variable([filter_size, filter_size, features // 2, features], stddev, name="w1")
            wi1 = weight_variable([1, 1, features // 2, features], stddev, name="wi1")
        w2 = weight_variable([filter_size, filter_size, features, features], stddev, name="w2")
        w3 = weight_variable([filter_size, filter_size, features, features], stddev, name="w3")
        w4 = weight_variable([filter_size, filter_size, features, features], stddev, name="w4")
        wi2 = weight_variable([1, 1, features, features], stddev, name="wi2")
        b1 = bias_variable([features], name="b1")
        b2 = bias_variable([features], name="b2")
        b3 = bias_variable([features], name="b3")
        b4 = bias_variable([features], name="b4")
        bi1 = bias_variable([features], name="bi1")
        bi2 = bias_variable([features], name="bi2")
        w = [w1, w2, w3, w4, wi1, wi2]
        b = [b1, b2, b3, b4, bi1, bi2]
        # return w1, w2, w3, w4, b1, b2, b3, b4, wi1, bi1, wi2, bi2
        return w, b


def calc_res_up_weight_bias(layer, filter_size, pool_size, features_root):
    features = 2 ** (layer + 1) * features_root
    stddev = np.sqrt(2 / (filter_size ** 2 * features))
    with tf.name_scope("up_conv_{}".format(str(layer))):
        wd = weight_variable_devonc([pool_size, pool_size, features // 2, features], stddev, name="wd")
        bd = bias_variable([features // 2], name="bd")
        w1 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w1")
        w2 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w2")
        w3 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w3")
        w4 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w4")
        wi1 = weight_variable([1, 1, features // 2, features // 2], stddev, name="wi1")
        wi2 = weight_variable([1, 1, features // 2, features // 2], stddev, name="wi2")
        b1 = bias_variable([features // 2], name="b1")
        b2 = bias_variable([features // 2], name="b2")
        b3 = bias_variable([features // 2], name="b3")
        b4 = bias_variable([features // 2], name="b4")
        bi1 = bias_variable([features // 2], name="bi1")
        bi2 = bias_variable([features // 2], name="bi2")
        w = [w1, w2, w3, w4, wi1, wi2]
        b = [b1, b2, b3, b4, bi1, bi2]
        # return wd, bd, w1, w2, w3, w4, b1, b2, b3, b4, wi1, bi1, wi2, bi2
        return wd, bd, w, b


def calc_resu_up_weight_bias(layer, filter_size, pool_size, features_root):
    features = 2 ** (layer + 1) * features_root
    stddev = np.sqrt(2 / (filter_size ** 2 * features))
    with tf.name_scope("up_conv_{}".format(str(layer))):
        wd = weight_variable_devonc([pool_size, pool_size, features // 2, features], stddev, name="wd")
        bd = bias_variable([features // 2], name="bd")
        w1 = weight_variable([filter_size, filter_size, features, features // 2], stddev, name="w1")
        w2 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w2")
        w3 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w3")
        w4 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w4")
        wi1 = weight_variable([1, 1, features, features // 2], stddev, name="wi1")
        wi2 = weight_variable([1, 1, features // 2, features // 2], stddev, name="wi2")
        b1 = bias_variable([features // 2], name="b1")
        b2 = bias_variable([features // 2], name="b2")
        b3 = bias_variable([features // 2], name="b3")
        b4 = bias_variable([features // 2], name="b4")
        bi1 = bias_variable([features // 2], name="bi1")
        bi2 = bias_variable([features // 2], name="bi2")
        w = [w1, w2, w3, w4, wi1, wi2]
        b = [b1, b2, b3, b4, bi1, bi2]
        return wd, bd, w, b


def calc_resau_up_weight_bias(layer, filter_size, pool_size, features_root):
    features = 2 ** (layer + 1) * features_root
    stddev = np.sqrt(2 / (filter_size ** 2 * features))
    with tf.name_scope("up_conv_{}".format(str(layer))):
        wd = weight_variable_devonc([pool_size, pool_size, features // 2, features], stddev, name="wd")
        bd = bias_variable([features // 2], name="bd")
        w1 = weight_variable([filter_size, filter_size, features, features // 2], stddev, name="w1")
        w2 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w2")
        w3 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w3")
        w4 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w4")
        wi1 = weight_variable([1, 1, features, features // 2], stddev, name="wi1")
        wi2 = weight_variable([1, 1, features // 2, features // 2], stddev, name="wi2")
        wa1 = weight_variable([1, 1, features // 2, features // 2], stddev, name="wa1")
        wa2 = weight_variable([1, 1, features // 2, features // 2], stddev, name="wa2")
        wa3 = weight_variable([1, 1, features // 2, 1], stddev, name="wa3")
        b1 = bias_variable([features // 2], name="b1")
        b2 = bias_variable([features // 2], name="b2")
        b3 = bias_variable([features // 2], name="b3")
        b4 = bias_variable([features // 2], name="b4")
        bi1 = bias_variable([features // 2], name="bi1")
        bi2 = bias_variable([features // 2], name="bi2")
        ba1 = bias_variable([features // 2], name="ba1")
        ba2 = bias_variable([features // 2], name="ba2")
        ba3 = bias_variable([1], name="ba3")
        w = [w1, w2, w3, w4, wi1, wi2]
        b = [b1, b2, b3, b4, bi1, bi2]
        wa = [wa1, wa2, wa3]
        ba = [ba1, ba2, ba3]
        return wd, bd, w, b, wa, ba


def calc_dense_down_weight_bias(layer, filter_size, features_root):
    features = 2 ** layer * features_root
    stddev = np.sqrt(2 / (filter_size ** 2 * features))
    with tf.name_scope("down_conv_{}".format(str(layer))):
        w0 = weight_variable([filter_size, filter_size, 1, features], stddev,
                             name="w0") if layer == 0 else weight_variable(
            [filter_size, filter_size, features // 2, features], stddev, name="w0")
        b0 = bias_variable([features], name="b0")
        w1 = weight_variable([filter_size, filter_size, features, features], stddev, name="w1")
        w2 = weight_variable([1, 1, 2 * features, features], stddev, name="w2")
        w3 = weight_variable([filter_size, filter_size, features, features], stddev, name="w3")
        w4 = weight_variable([1, 1, 2 * features, features], stddev, name="w4")
        w5 = weight_variable([filter_size, filter_size, features, features], stddev, name="w5")
        w6 = weight_variable([1, 1, 2 * features, features], stddev, name="w6")
        w7 = weight_variable([filter_size, filter_size, features, features], stddev, name="w7")
        w8 = weight_variable([1, 1, 2 * features, features], stddev, name="w8")
        b1 = bias_variable([features], name="b1")
        b2 = bias_variable([features], name="b2")
        b3 = bias_variable([features], name="b3")
        b4 = bias_variable([features], name="b4")
        b5 = bias_variable([features], name="b5")
        b6 = bias_variable([features], name="b6")
        b7 = bias_variable([features], name="b7")
        b8 = bias_variable([features], name="b8")
        w = [w1, w2, w3, w4, w5, w6, w7, w8]
        b = [b1, b2, b3, b4, b5, b6, b7, b8]
        return w0, b0, w, b


def calc_dense_up_weight_bias(layer, filter_size, pool_size, features_root):
    features = 2 ** (layer + 1) * features_root
    stddev = np.sqrt(2 / (filter_size ** 2 * features))
    with tf.name_scope("up_conv_{}".format(str(layer))):
        wd = weight_variable_devonc([pool_size, pool_size, features // 2, features], stddev, name="wd")
        bd = bias_variable([features // 2], name="bd")
        w1 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w1")
        w2 = weight_variable([1, 1, features, features // 2], stddev, name="w2")
        w3 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w3")
        w4 = weight_variable([1, 1, features, features // 2], stddev, name="w4")
        w5 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w5")
        w6 = weight_variable([1, 1, features, features // 2], stddev, name="w6")
        w7 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w7")
        w8 = weight_variable([1, 1, features, features // 2], stddev, name="w8")
        b1 = bias_variable([features // 2], name="b1")
        b2 = bias_variable([features // 2], name="b2")
        b3 = bias_variable([features // 2], name="b3")
        b4 = bias_variable([features // 2], name="b4")
        b5 = bias_variable([features // 2], name="b5")
        b6 = bias_variable([features // 2], name="b6")
        b7 = bias_variable([features // 2], name="b7")
        b8 = bias_variable([features // 2], name="b8")
        w = [w1, w2, w3, w4, w5, w6, w7, w8]
        b = [b1, b2, b3, b4, b5, b6, b7, b8]
        return wd, bd, w, b


def calc_denseu_up_weight_bias(layer, filter_size, pool_size, features_root):
    with tf.name_scope("up_conv_{}".format(str(layer))):
        features = 2 ** (layer + 1) * features_root
        stddev = np.sqrt(2 / (filter_size ** 2 * features))
        wd = weight_variable_devonc([pool_size, pool_size, features // 2, features], stddev, name="wd")
        bd = bias_variable([features // 2], name="bd")
        w1 = weight_variable([filter_size, filter_size, features, features // 2], stddev, name="w1")
        w2 = weight_variable([1, 1, features + features // 2, features // 2], stddev, name="w2")
        w3 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w3")
        w4 = weight_variable([1, 1, features, features // 2], stddev, name="w4")
        w5 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w5")
        w6 = weight_variable([1, 1, features, features // 2], stddev, name="w6")
        w7 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w7")
        w8 = weight_variable([1, 1, features, features // 2], stddev, name="w8")
        b1 = bias_variable([features // 2], name="b1")
        b2 = bias_variable([features // 2], name="b2")
        b3 = bias_variable([features // 2], name="b3")
        b4 = bias_variable([features // 2], name="b4")
        b5 = bias_variable([features // 2], name="b5")
        b6 = bias_variable([features // 2], name="b6")
        b7 = bias_variable([features // 2], name="b7")
        b8 = bias_variable([features // 2], name="b8")
        w = [w1, w2, w3, w4, w5, w6, w7, w8]
        b = [b1, b2, b3, b4, b5, b6, b7, b8]
        return wd, bd, w, b


def calc_SE_up_weight_bias(layer, filter_size, pool_size, features_root):
    features = 2 ** (layer + 1) * features_root
    stddev = np.sqrt(2 / (filter_size ** 2 * features))
    with tf.name_scope("up_conv_{}".format(str(layer))):
        wd = weight_variable_devonc([pool_size, pool_size, features // 2, features], stddev, name="wd")
        bd = bias_variable([features // 2], name="bd")
        w1 = weight_variable([filter_size, filter_size, features, features // 2], stddev, name="w1")
        w2 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w2")
        wi1 = weight_variable([1, 1, features, features // 2], stddev, name="wi1")
        b1 = bias_variable([features // 2], name="b1")
        b2 = bias_variable([features // 2], name="b2")
        bi1 = bias_variable([features // 2], name="bi1")
        w = [w1, w2, wi1]
        b = [b1, b2, bi1]
        return wd, bd, w, b
        return wd, bd, w, b


def calc_pam_weight(layer, filter_size, features_root):
    with tf.name_scope('pos_attention_module'):
        features = 2 ** layer * features_root
        stddev = np.sqrt(2 / (filter_size ** 2 * features))
        w0 = weight_variable([3, 3, features, features], stddev, name="w0")
        w1 = weight_variable([1, 1, features, features // 8], stddev, name="w1")
        w2 = weight_variable([1, 1, features, features // 8], stddev, name="w2")
        w3 = weight_variable([1, 1, features, features], stddev, name="w3")
        w4 = weight_variable([1, 1, features, features], stddev, name="w4")
        b1 = bias_variable([features // 8], name="b1")
        b2 = bias_variable([features // 8], name="b2")
        b3 = bias_variable([features], name="b3")
        b4 = bias_variable([features], name="b4")
        gamma = tf.Variable(initial_value=tf.constant(0.1, shape=[1]), name='gamma')
        w = [w0, w1, w2, w3, w4]
        b = [b1, b2, b3, b4]
        return w, b, gamma


def conv2d(x, W, b, keep_prob_):
    with tf.name_scope("conv2d"):
        conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        conv_2d_b = tf.nn.bias_add(conv_2d, b)
        return tf.nn.dropout(conv_2d_b, keep_prob_)


def conv2d_nb(x, W, keep_prob_):
    with tf.name_scope("conv2d"):
        conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.dropout(conv_2d, keep_prob_)


def conv2d_relu(x, W, b, keep_prob_, is_training):
    with tf.name_scope("conv2d"):
        conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        conv_2d_b = tf.nn.bias_add(conv_2d, b)
        return tf.nn.relu(tf.layers.batch_normalization(tf.nn.dropout(conv_2d_b, keep_prob_), training=is_training))


def conv2d_bn_relu_nb(x, filters, keep_prob_, is_training):
    with tf.name_scope("conv2d"):
        conv_2d = tf.layers.conv2d(x, filters, 3, padding='SAME', use_bias=False)
        return tf.nn.relu(tf.layers.batch_normalization(tf.nn.dropout(conv_2d, keep_prob_), training=is_training))


def conv2d_relu_nb(x, W, keep_prob_, is_training):
    with tf.name_scope("conv2d"):
        conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.relu(tf.layers.batch_normalization(tf.nn.dropout(conv_2d, keep_prob_), training=is_training))


def SE_layer(x, channel, reduction=4):
    with tf.name_scope('SElayer'):
        avg_pool = global_avg_pool(x, name='Global_avg_pool')
        fc1 = tf.layers.dense(inputs=avg_pool, use_bias=True, units=channel // reduction)
        ReLU = tf.nn.relu(fc1)
        fc2 = tf.layers.dense(inputs=ReLU, use_bias=True, units=channel)
        Sigmoid = tf.nn.sigmoid(fc2)
        excitation = tf.reshape(Sigmoid, [-1, 1, 1, channel])
        return x * excitation


def SE_down_block(input, w, b, channel, keep_prob_, is_training):
    conv1 = conv2d_relu(input, w[0], b[0], keep_prob_, is_training)
    conv2 = conv2d_relu(conv1, w[1], b[1], keep_prob_, is_training)
    bottle2 = conv2d_relu(input, w[4], b[4], keep_prob_, is_training)
    se2 = SE_layer(conv2, channel)
    shortcut2 = se2 + bottle2
    relu2 = tf.nn.relu(shortcut2)
    conv3 = conv2d_relu(relu2, w[2], b[2], keep_prob_, is_training)
    conv4 = conv2d_relu(conv3, w[3], b[3], keep_prob_, is_training)
    bottle4 = conv2d_relu(relu2, w[5], b[5], keep_prob_, is_training)
    shortcut4 = conv4 + bottle4
    relu4 = tf.nn.relu(shortcut4)
    return relu4


def SE_up_block(input, w, b, channel, keep_prob_, is_training):
    conv1 = conv2d_relu(input, w[0], b[0], keep_prob_, is_training)
    conv2 = conv2d_relu(conv1, w[1], b[1], keep_prob_, is_training)
    bottle2 = conv2d_relu(input, w[2], b[2], keep_prob_, is_training)
    se2 = SE_layer(conv2, channel)
    shortcut2 = se2 + bottle2
    relu2 = tf.nn.relu(shortcut2)
    return relu2


def deconv2d(x, W, stride):
    with tf.name_scope("deconv2d"):
        x_shape = tf.shape(x)
        output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] // 2])
        return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='SAME',
                                      name="conv2d_transpose")


def max_pool(x, n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')


def crop_and_concat(x1, x2):
    with tf.name_scope("crop_and_concat"):
        return tf.concat([x1, x2], 3)


def pixel_wise_softmax(output_map):
    with tf.name_scope("pixel_wise_softmax"):
        max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
        return exponential_map / normalize


def cross_entropy(y_, output_map):
    return -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(output_map, 1e-10, 1.0)), name="cross_entropy")


def basic_block(input, w, b, keep_prob_, is_training):
    conv1 = conv2d_relu(input, w[0], b[0], keep_prob_, is_training)
    conv2 = conv2d_relu(conv1, w[1], b[1], keep_prob_, is_training)
    return conv2


def residual_block(input, w, b, keep_prob_, is_training):
    conv1 = conv2d_relu(input, w[0], b[0], keep_prob_, is_training)
    conv2 = conv2d_relu(conv1, w[1], b[1], keep_prob_, is_training)
    bottle2 = conv2d_relu(input, w[4], b[4], keep_prob_, is_training)
    shortcut2 = conv2 + bottle2
    relu2 = tf.nn.relu(shortcut2)
    conv3 = conv2d_relu(relu2, w[2], b[2], keep_prob_, is_training)
    conv4 = conv2d_relu(conv3, w[3], b[3], keep_prob_, is_training)
    bottle4 = conv2d_relu(relu2, w[5], b[5], keep_prob_, is_training)
    shortcut4 = conv4 + bottle4
    relu4 = tf.nn.relu(shortcut4)
    return relu4


def dense_block(input, w, b, keep_prob_, is_training):
    conv1 = conv2d_relu(input, w[0], b[0], keep_prob_, is_training)
    merge1 = tf.concat([input, conv1], axis=-1)
    conv2 = conv2d_relu(merge1, w[1], b[1], keep_prob_, is_training)
    conv3 = conv2d_relu(conv2, w[2], b[2], keep_prob_, is_training)
    merge3 = tf.concat([conv2, conv3], axis=-1)
    conv4 = conv2d_relu(merge3, w[3], b[3], keep_prob_, is_training)
    conv5 = conv2d_relu(conv4, w[4], b[4], keep_prob_, is_training)
    merge5 = tf.concat([conv4, conv5], axis=-1)
    conv6 = conv2d_relu(merge5, w[5], b[5], keep_prob_, is_training)
    conv7 = conv2d_relu(conv6, w[6], b[6], keep_prob_, is_training)
    merge7 = tf.concat([conv6, conv7], axis=-1)
    conv8 = conv2d_relu(merge7, w[7], b[7], keep_prob_, is_training)
    # new_features = tf.concat([conv1, conv2, conv3, conv4], axis=-1)
    return conv8


def attention_gate(input, att_input, w, b, keep_prob_):
    g1 = conv2d(input, w[0], b[0], keep_prob_)
    x1 = conv2d(att_input, w[1], b[1], keep_prob_)
    net = tf.add(g1, x1)
    net = tf.nn.relu(net)
    net = conv2d(net, w[2], b[2], keep_prob_)
    net = tf.nn.sigmoid(net)
    # net = tf.concat([att_tensor, net], axis=-1)
    net = net * att_input
    return net


def pam_model(input, h, w, filters):
    proj_query = tf.layers.conv2d(inputs=input, filters=filters // 8, kernel_size=1, padding='SAME')
    proj_query = tf.transpose(tf.reshape(proj_query, [-1, h * w, filters // 8]), (0, 2, 1))
    proj_key = tf.layers.conv2d(inputs=input, filters=filters // 8, kernel_size=1, padding='SAME')
    proj_key = tf.reshape(proj_key, [-1, h * w, filters // 8])
    energy = tf.matmul(proj_key, proj_query)
    attention = tf.nn.softmax(energy, dim=-1)
    proj_value = tf.layers.conv2d(inputs=input, filters=filters, kernel_size=1, padding='SAME')
    proj_value = tf.reshape(proj_value, [-1, h * w, filters])
    out = tf.matmul(attention, proj_value)
    out = tf.reshape(out, [-1, h, w, filters])
    gamma = tf.Variable(initial_value=tf.constant(0.0, shape=(1,)), name='pam_gamma')
    out = gamma * out + input
    return out


def cam_model(input, h, w, filters):
    proj_query = tf.reshape(input, [-1, h * w, filters])
    proj_key = tf.reshape(input, [-1, h * w, filters])
    proj_key = tf.transpose(proj_key, [0, 2, 1])
    energy = tf.matmul(proj_key, proj_query)
    attention = tf.nn.softmax(energy, dim=-1)
    proj_value = tf.reshape(input, [-1, h * w, filters])
    out = tf.matmul(proj_value, attention)
    out = tf.reshape(out, [-1, h, w, filters])
    gamma = tf.Variable(initial_value=tf.constant(0.0, shape=(1,)), name='cam_gamma')
    out = gamma * out + input
    return out


def da_head(input, h, w, filters, keep_prob, is_training):
    reduce_conv = conv2d_bn_relu_nb(input, filters, keep_prob, is_training)
    pam = pam_model(reduce_conv, h, w, filters)
    pam = conv2d_bn_relu_nb(pam, filters, keep_prob, is_training)
    pam = tf.layers.conv2d(pam, filters, 3, padding='same', use_bias=False, kernel_initializer='he_normal')
    cam = cam_model(reduce_conv, h, w, filters)
    cam = conv2d_bn_relu_nb(cam, filters, keep_prob, is_training)
    cam = tf.layers.conv2d(cam, filters, 3, padding='same', use_bias=False, kernel_initializer='he_normal')
    feature_sum = tf.add(pam, cam)
    out = tf.layers.batch_normalization(tf.nn.dropout(feature_sum, keep_prob), training=is_training)
    return out


def pa_head(input, h, w, filters, keep_prob, is_training):
    reduce_conv = conv2d_bn_relu_nb(input, filters, keep_prob, is_training)
    pam = pam_model(reduce_conv, h, w, filters)
    pam = conv2d_bn_relu_nb(pam, filters, keep_prob, is_training)
    pam = tf.layers.conv2d(pam, filters, 3, padding='same', use_bias=False, kernel_initializer='he_normal')
    out = tf.layers.batch_normalization(tf.nn.dropout(pam, keep_prob), training=is_training)
    return out


def ca_head(input, h, w, filters, keep_prob, is_training):
    reduce_conv = conv2d_bn_relu_nb(input, filters, keep_prob, is_training)
    cam = cam_model(reduce_conv, h, w, filters)
    cam = conv2d_bn_relu_nb(cam, filters, keep_prob, is_training)
    cam = tf.layers.conv2d(cam, filters, 3, padding='same', use_bias=False, kernel_initializer='he_normal')
    out = tf.layers.batch_normalization(tf.nn.dropout(cam, keep_prob), training=is_training)
    return out