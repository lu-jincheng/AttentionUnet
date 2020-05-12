import tensorflow as tf
import logging
from layer import *


def create_unet(x, keep_prob, channels, layers=5, n_class=4, features_root=16, filter_size=3, pool_size=2,
                is_training=True):
    """
    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """

    logging.info(
        "Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{"
        "pool_size}".format(
            layers=layers,
            features=features_root,
            filter_size=filter_size,
            pool_size=pool_size))

    # Placeholder for the input image
    with tf.name_scope("preprocessing"):
        nx = tf.shape(x)[1]
        ny = tf.shape(x)[2]
        x_image = tf.reshape(x, tf.stack([-1, nx, ny, channels]))
        in_node = x_image

    w1, w2, b1, b2 = calc_down_weight_bias(0, filter_size, features_root)
    w3, w4, b3, b4 = calc_down_weight_bias(1, filter_size, features_root)
    w5, w6, b5, b6 = calc_down_weight_bias(2, filter_size, features_root)
    w7, w8, b7, b8 = calc_down_weight_bias(3, filter_size, features_root)
    w9, w10, b9, b10 = calc_down_weight_bias(4, filter_size, features_root)

    wd10, bd10, w11, w12, b11, b12 = calc_up_weight_bias(3, filter_size, pool_size, features_root)
    wd12, bd12, w13, w14, b13, b14 = calc_up_weight_bias(2, filter_size, pool_size, features_root)
    wd14, bd14, w15, w16, b15, b16 = calc_up_weight_bias(1, filter_size, pool_size, features_root)
    wd16, bd16, w17, w18, b17, b18 = calc_up_weight_bias(0, filter_size, pool_size, features_root)

    # Down layers
    with tf.name_scope('conv1'):
        conv1 = conv2d_relu(in_node, w1, b1, keep_prob, is_training)
    with tf.name_scope('conv2'):
        conv2 = conv2d_relu(conv1, w2, b2, keep_prob, is_training)
    pool2 = max_pool(conv2, pool_size)

    with tf.name_scope('conv3'):
        conv3 = conv2d_relu(pool2, w3, b3, keep_prob, is_training)
    with tf.name_scope('conv4'):
        conv4 = conv2d_relu(conv3, w4, b4, keep_prob, is_training)
    pool4 = max_pool(conv4, pool_size)

    with tf.name_scope('conv5'):
        conv5 = conv2d_relu(pool4, w5, b5, keep_prob, is_training)
    with tf.name_scope('conv6'):
        conv6 = conv2d_relu(conv5, w6, b6, keep_prob, is_training)
    pool6 = max_pool(conv6, pool_size)

    with tf.name_scope('conv7'):
        conv7 = conv2d_relu(pool6, w7, b7, keep_prob, is_training)
    with tf.name_scope('conv8'):
        conv8 = conv2d_relu(conv7, w8, b8, keep_prob, is_training)
    pool8 = max_pool(conv8, pool_size)

    with tf.name_scope('conv9'):
        conv9 = conv2d_relu(pool8, w9, b9, keep_prob, is_training)
    with tf.name_scope('conv10'):
        conv10 = conv2d_relu(conv9, w10, b10, keep_prob, is_training)

    # Up layers
    up10 = tf.nn.relu(deconv2d(conv10, wd10, pool_size) + bd10)
    merge10 = crop_and_concat(conv8, up10)
    with tf.name_scope('conv11'):
        conv11 = conv2d_relu(merge10, w11, b11, keep_prob, is_training)
    with tf.name_scope('conv12'):
        conv12 = conv2d_relu(conv11, w12, b12, keep_prob, is_training)

    up12 = tf.nn.relu(deconv2d(conv12, wd12, pool_size) + bd12)
    merge12 = crop_and_concat(conv6, up12)
    with tf.name_scope('conv13'):
        conv13 = conv2d_relu(merge12, w13, b13, keep_prob, is_training)
    with tf.name_scope('conv14'):
        conv14 = conv2d_relu(conv13, w14, b14, keep_prob, is_training)

    up14 = tf.nn.relu(deconv2d(conv14, wd14, pool_size) + bd14)
    merge14 = crop_and_concat(conv4, up14)
    with tf.name_scope('conv15'):
        conv15 = conv2d_relu(merge14, w15, b15, keep_prob, is_training)
    with tf.name_scope('conv16'):
        conv16 = conv2d_relu(conv15, w16, b16, keep_prob, is_training)

    up16 = tf.nn.relu(deconv2d(conv16, wd16, pool_size) + bd16)
    merge16 = crop_and_concat(conv2, up16)
    with tf.name_scope('conv17'):
        conv17 = conv2d_relu(merge16, w17, b17, keep_prob, is_training)
    with tf.name_scope('conv18'):
        conv18 = conv2d_relu(conv17, w18, b18, keep_prob, is_training)

    # Output Map
    stddev = np.sqrt(2 / (filter_size ** 2 * 128))
    weight = weight_variable([1, 1, features_root, n_class], stddev)
    bias = bias_variable([n_class], name="bias")
    conv19 = conv2d(conv18, weight, bias, tf.constant(1.0))
    output_map = tf.nn.sigmoid(conv19)

    variables = [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18,
                 b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18]

    return output_map, variables


def create_attention_gate_unet(x, keep_prob, channels, layers=5, n_class=4, features_root=16, filter_size=3, pool_size=2,
                               is_training=True):
    """
    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """

    logging.info(
        "Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{"
        "pool_size}".format(
            layers=layers,
            features=features_root,
            filter_size=filter_size,
            pool_size=pool_size))

    # Placeholder for the input image
    with tf.name_scope("preprocessing"):
        nx = tf.shape(x)[1]
        ny = tf.shape(x)[2]
        x_image = tf.reshape(x, tf.stack([-1, nx, ny, channels]))
        in_node = x_image

    w1, w2, b1, b2 = calc_down_weight_bias(0, filter_size, features_root)
    w3, w4, b3, b4 = calc_down_weight_bias(1, filter_size, features_root)
    w5, w6, b5, b6 = calc_down_weight_bias(2, filter_size, features_root)
    w7, w8, b7, b8 = calc_down_weight_bias(3, filter_size, features_root)
    w9, w10, b9, b10 = calc_down_weight_bias(4, filter_size, features_root)

    wd10, bd10, w11, w12, b11, b12, wa10, ba10 = calc_au_up_weight_bias(3, filter_size, pool_size, features_root)
    wd12, bd12, w13, w14, b13, b14, wa12, ba12 = calc_au_up_weight_bias(2, filter_size, pool_size, features_root)
    wd14, bd14, w15, w16, b15, b16, wa14, ba14 = calc_au_up_weight_bias(1, filter_size, pool_size, features_root)
    wd16, bd16, w17, w18, b17, b18, wa16, ba16 = calc_au_up_weight_bias(0, filter_size, pool_size, features_root)

    # Down layers
    with tf.name_scope('conv1'):
        conv1 = conv2d_relu(in_node, w1, b1, keep_prob, is_training)
    with tf.name_scope('conv2'):
        conv2 = conv2d_relu(conv1, w2, b2, keep_prob, is_training)
    pool2 = max_pool(conv2, pool_size)

    with tf.name_scope('conv3'):
        conv3 = conv2d_relu(pool2, w3, b3, keep_prob, is_training)
    with tf.name_scope('conv4'):
        conv4 = conv2d_relu(conv3, w4, b4, keep_prob, is_training)
    pool4 = max_pool(conv4, pool_size)

    with tf.name_scope('conv5'):
        conv5 = conv2d_relu(pool4, w5, b5, keep_prob, is_training)
    with tf.name_scope('conv6'):
        conv6 = conv2d_relu(conv5, w6, b6, keep_prob, is_training)
    pool6 = max_pool(conv6, pool_size)

    with tf.name_scope('conv7'):
        conv7 = conv2d_relu(pool6, w7, b7, keep_prob, is_training)
    with tf.name_scope('conv8'):
        conv8 = conv2d_relu(conv7, w8, b8, keep_prob, is_training)
    pool8 = max_pool(conv8, pool_size)

    with tf.name_scope('conv9'):
        conv9 = conv2d_relu(pool8, w9, b9, keep_prob, is_training)
    with tf.name_scope('conv10'):
        conv10 = conv2d_relu(conv9, w10, b10, keep_prob, is_training)

    # Up layers
    up10 = tf.nn.relu(deconv2d(conv10, wd10, pool_size) + bd10)
    att10 = attention_gate(conv8, up10, wa10, ba10, keep_prob)
    merge10 = crop_and_concat(conv8, att10)
    conv11 = conv2d_relu(merge10, w11, b11, keep_prob, is_training)
    conv12 = conv2d_relu(conv11, w12, b12, keep_prob, is_training)

    up12 = tf.nn.relu(deconv2d(conv12, wd12, pool_size) + bd12)
    att12 = attention_gate(conv6, up12, wa12, ba12, keep_prob)
    merge12 = crop_and_concat(conv6, att12)
    conv13 = conv2d_relu(merge12, w13, b13, keep_prob, is_training)
    conv14 = conv2d_relu(conv13, w14, b14, keep_prob, is_training)

    up14 = tf.nn.relu(deconv2d(conv14, wd14, pool_size) + bd14)
    att14 = attention_gate(conv4, up14, wa14, ba14, keep_prob)
    merge14 = crop_and_concat(conv4, att14)
    conv15 = conv2d_relu(merge14, w15, b15, keep_prob, is_training)
    conv16 = conv2d_relu(conv15, w16, b16, keep_prob, is_training)

    up16 = tf.nn.relu(deconv2d(conv16, wd16, pool_size) + bd16)
    att16 = attention_gate(conv2, up16, wa16, ba16, keep_prob)
    merge16 = crop_and_concat(conv2, att16)
    conv17 = conv2d_relu(merge16, w17, b17, keep_prob, is_training)
    conv18 = conv2d_relu(conv17, w18, b18, keep_prob, is_training)

    # Output Map
    with tf.name_scope('output_map'):
        stddev = np.sqrt(2 / (filter_size ** 2 * 128))
        weight = weight_variable([1, 1, features_root, n_class], stddev)
        bias = bias_variable([n_class], name="bias")
        conv19 = conv2d(conv18, weight, bias, tf.constant(1.0))
        output_map = tf.nn.sigmoid(conv19)

        variables = [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18,
                     b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18]
        variables = variables + wa10 + wa12 + wa14 + wa16 + ba10 + ba12 + ba14 + ba16

        return output_map, variables


def create_resnet(x, keep_prob, channels, layers, n_class=4, features_root=16, filter_size=3, pool_size=2,
                  is_training=True):
    """
    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """

    logging.info(
        "Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{"
        "pool_size}".format(
            layers=layers,
            features=features_root,
            filter_size=filter_size,
            pool_size=pool_size))

    # Placeholder for the input image
    with tf.name_scope("preprocessing"):
        nx = tf.shape(x)[1]
        ny = tf.shape(x)[2]
        x_image = tf.reshape(x, tf.stack([-1, nx, ny, channels]))
        in_node = x_image

    w1, b1 = calc_res_down_weight_bias(0, filter_size, features_root)
    w2, b2 = calc_res_down_weight_bias(1, filter_size, features_root)
    w3, b3 = calc_res_down_weight_bias(2, filter_size, features_root)
    w4, b4 = calc_res_down_weight_bias(3, filter_size, features_root)
    w5, b5 = calc_res_down_weight_bias(4, filter_size, features_root)

    wd6, bd6, w6, b6 = calc_nsc_upblock_weight_bias(3, filter_size, pool_size, features_root)
    wd7, bd7, w7, b7 = calc_nsc_upblock_weight_bias(2, filter_size, pool_size, features_root)
    wd8, bd8, w8, b8 = calc_nsc_upblock_weight_bias(1, filter_size, pool_size, features_root)
    wd9, bd9, w9, b9 = calc_nsc_upblock_weight_bias(0, filter_size, pool_size, features_root)

    # Down layers
    res1 = residual_block(in_node, w1, b1, keep_prob, is_training)
    pool1 = max_pool(res1, pool_size)

    res2 = residual_block(pool1, w2, b2, keep_prob, is_training)
    pool2 = max_pool(res2, pool_size)

    res3 = residual_block(pool2, w3, b3, keep_prob, is_training)
    pool3 = max_pool(res3, pool_size)

    res4 = residual_block(pool3, w4, b4, keep_prob, is_training)
    pool4 = max_pool(res4, pool_size)

    res5 = residual_block(pool4, w5, b5, keep_prob, is_training)

    # Up layers
    up6 = tf.nn.relu(deconv2d(res5, wd6, pool_size) + bd6)
    conv6 = basic_block(up6, w6, b6, keep_prob, is_training)

    up7 = tf.nn.relu(deconv2d(conv6, wd7, pool_size) + bd7)
    conv7 = basic_block(up7, w7, b7, keep_prob, is_training)

    up8 = tf.nn.relu(deconv2d(conv7, wd8, pool_size) + bd8)
    conv8 = basic_block(up8, w8, b8, keep_prob, is_training)

    up9 = tf.nn.relu(deconv2d(conv8, wd9, pool_size) + bd9)
    conv9 = basic_block(up9, w9, b9, keep_prob, is_training)

    # Output Map
    with tf.name_scope('output_map'):
        stddev = np.sqrt(2 / (filter_size ** 2 * 128))
        weight = weight_variable([1, 1, features_root, n_class], stddev)
        bias = bias_variable([n_class], name="bias")
        conv_out = conv2d(conv9, weight, bias, tf.constant(1.0))
        output_map = tf.nn.sigmoid(conv_out)

    variables = w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8 + w9 + b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9

    return output_map, variables


def create_res_unet(x, keep_prob, channels, layers, n_class=4, features_root=16, filter_size=3, pool_size=2,
                    is_training=True):
    """
    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """

    logging.info(
        "Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{"
        "pool_size}".format(
            layers=layers,
            features=features_root,
            filter_size=filter_size,
            pool_size=pool_size))

    # Placeholder for the input image
    with tf.name_scope("preprocessing"):
        nx = tf.shape(x)[1]
        ny = tf.shape(x)[2]
        x_image = tf.reshape(x, tf.stack([-1, nx, ny, channels]))
        in_node = x_image
        # batch_size = tf.shape(x_image)[0]

    w1, b1 = calc_res_down_weight_bias(0, filter_size, features_root)
    w2, b2 = calc_res_down_weight_bias(1, filter_size, features_root)
    w3, b3 = calc_res_down_weight_bias(2, filter_size, features_root)
    w4, b4 = calc_res_down_weight_bias(3, filter_size, features_root)
    w5, b5 = calc_res_down_weight_bias(4, filter_size, features_root)

    wd6, bd6, w6, b6 = calc_upblock_weight_bias(3, filter_size, pool_size, features_root)
    wd7, bd7, w7, b7 = calc_upblock_weight_bias(2, filter_size, pool_size, features_root)
    wd8, bd8, w8, b8 = calc_upblock_weight_bias(1, filter_size, pool_size, features_root)
    wd9, bd9, w9, b9 = calc_upblock_weight_bias(0, filter_size, pool_size, features_root)

    # Down layers
    res1 = residual_block(in_node, w1, b1, keep_prob, is_training)
    pool1 = max_pool(res1, pool_size)

    res2 = residual_block(pool1, w2, b2, keep_prob, is_training)
    pool2 = max_pool(res2, pool_size)

    res3 = residual_block(pool2, w3, b3, keep_prob, is_training)
    pool3 = max_pool(res3, pool_size)

    res4 = residual_block(pool3, w4, b4, keep_prob, is_training)
    pool4 = max_pool(res4, pool_size)

    res5 = residual_block(pool4, w5, b5, keep_prob, is_training)

    # Up layers
    up6 = tf.nn.relu(deconv2d(res5, wd6, pool_size) + bd6)
    merge6 = crop_and_concat(res4, up6)
    conv6 = basic_block(merge6, w6, b6, keep_prob, is_training)

    up7 = tf.nn.relu(deconv2d(conv6, wd7, pool_size) + bd7)
    merge7 = crop_and_concat(res3, up7)
    conv7 = basic_block(merge7, w7, b7, keep_prob, is_training)

    up8 = tf.nn.relu(deconv2d(conv7, wd8, pool_size) + bd8)
    merge8 = crop_and_concat(res2, up8)
    conv8 = basic_block(merge8, w8, b8, keep_prob, is_training)

    up9 = tf.nn.relu(deconv2d(conv8, wd9, pool_size) + bd9)
    merge9 = crop_and_concat(res1, up9)
    conv9 = basic_block(merge9, w9, b9, keep_prob, is_training)

    # Output Map
    stddev = np.sqrt(2 / (filter_size ** 2 * 128))
    weight = weight_variable([1, 1, features_root, n_class], stddev)
    bias = bias_variable([n_class], name="bias")
    conv_out = conv2d(conv9, weight, bias, tf.constant(1.0))
    output_map = tf.nn.sigmoid(conv_out)

    variables = w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8 + w9 + b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9

    return output_map, variables


def create_densenet(x, keep_prob, channels, layers, n_class=4, features_root=16, filter_size=3, pool_size=2,
                    is_training=True):
    """
    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """

    logging.info(
        "Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{"
        "pool_size}".format(
            layers=layers,
            features=features_root,
            filter_size=filter_size,
            pool_size=pool_size))

    # Placeholder for the input image
    with tf.name_scope("preprocessing"):
        nx = tf.shape(x)[1]
        ny = tf.shape(x)[2]
        x_image = tf.reshape(x, tf.stack([-1, nx, ny, channels]))
        in_node = x_image

    wc1, bc1, w1, b1 = calc_dense_down_weight_bias(0, filter_size, features_root)
    wc2, bc2, w2, b2 = calc_dense_down_weight_bias(1, filter_size, features_root)
    wc3, bc3, w3, b3 = calc_dense_down_weight_bias(2, filter_size, features_root)
    wc4, bc4, w4, b4 = calc_dense_down_weight_bias(3, filter_size, features_root)
    wc5, bc5, w5, b5 = calc_dense_down_weight_bias(4, filter_size, features_root)

    wd6, bd6, w6, b6 = calc_nsc_upblock_weight_bias(3, filter_size, pool_size, features_root)
    wd7, bd7, w7, b7 = calc_nsc_upblock_weight_bias(2, filter_size, pool_size, features_root)
    wd8, bd8, w8, b8 = calc_nsc_upblock_weight_bias(1, filter_size, pool_size, features_root)
    wd9, bd9, w9, b9 = calc_nsc_upblock_weight_bias(0, filter_size, pool_size, features_root)

    # Down layer
    conv1 = conv2d_relu(in_node, wc1, bc1, keep_prob, is_training)
    dense1 = dense_block(conv1, w1, b1, keep_prob, is_training)
    pool1 = max_pool(dense1, pool_size)

    conv2 = conv2d_relu(pool1, wc2, bc2, keep_prob, is_training)
    dense2 = dense_block(conv2, w2, b2, keep_prob, is_training)
    pool2 = max_pool(dense2, pool_size)

    conv3 = conv2d_relu(pool2, wc3, bc3, keep_prob, is_training)
    dense3 = dense_block(conv3, w3, b3, keep_prob, is_training)
    pool3 = max_pool(dense3, pool_size)

    conv4 = conv2d_relu(pool3, wc4, bc4, keep_prob, is_training)
    dense4 = dense_block(conv4, w4, b4, keep_prob, is_training)
    pool4 = max_pool(dense4, pool_size)

    conv5 = conv2d_relu(pool4, wc5, bc5, keep_prob, is_training)
    dense5 = dense_block(conv5, w5, b5, keep_prob, is_training)

    # Up layers
    up6 = tf.nn.relu(deconv2d(dense5, wd6, pool_size) + bd6)
    conv6 = basic_block(up6, w6, b6, keep_prob, is_training)

    up7 = tf.nn.relu(deconv2d(conv6, wd7, pool_size) + bd7)
    conv7 = basic_block(up7, w7, b7, keep_prob, is_training)

    up8 = tf.nn.relu(deconv2d(conv7, wd8, pool_size) + bd8)
    conv8 = basic_block(up8, w8, b8, keep_prob, is_training)

    up9 = tf.nn.relu(deconv2d(conv8, wd9, pool_size) + bd9)
    conv9 = basic_block(up9, w9, b9, keep_prob, is_training)

    # Output Map
    stddev = np.sqrt(2 / (filter_size ** 2 * 128))
    weight = weight_variable([1, 1, features_root, n_class], stddev)
    bias = bias_variable([n_class], name="bias")
    conv_out = conv2d(conv9, weight, bias, tf.constant(1.0))
    output_map = tf.nn.sigmoid(conv_out)

    variables = w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8 + w9 + b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9

    return output_map, variables


def create_dense_unet(x, keep_prob, channels, layers, n_class=4, features_root=16, filter_size=3, pool_size=2,
                      is_training=True):
    """
    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """

    logging.info(
        "Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{"
        "pool_size}".format(
            layers=layers,
            features=features_root,
            filter_size=filter_size,
            pool_size=pool_size))

    # Placeholder for the input image
    with tf.name_scope("preprocessing"):
        nx = tf.shape(x)[1]
        ny = tf.shape(x)[2]
        x_image = tf.reshape(x, tf.stack([-1, nx, ny, channels]))
        in_node = x_image

    wc1, bc1, w1, b1 = calc_dense_down_weight_bias(0, filter_size, features_root)
    wc2, bc2, w2, b2 = calc_dense_down_weight_bias(1, filter_size, features_root)
    wc3, bc3, w3, b3 = calc_dense_down_weight_bias(2, filter_size, features_root)
    wc4, bc4, w4, b4 = calc_dense_down_weight_bias(3, filter_size, features_root)
    wc5, bc5, w5, b5 = calc_dense_down_weight_bias(4, filter_size, features_root)

    wd6, bd6, w6, b6 = calc_upblock_weight_bias(3, filter_size, pool_size, features_root)
    wd7, bd7, w7, b7 = calc_upblock_weight_bias(2, filter_size, pool_size, features_root)
    wd8, bd8, w8, b8 = calc_upblock_weight_bias(1, filter_size, pool_size, features_root)
    wd9, bd9, w9, b9 = calc_upblock_weight_bias(0, filter_size, pool_size, features_root)

    # Down layer

    conv1 = conv2d_relu(in_node, wc1, bc1, keep_prob, is_training)
    dense1 = dense_block(conv1, w1, b1, keep_prob, is_training)
    pool1 = max_pool(dense1, pool_size)

    conv2 = conv2d_relu(pool1, wc2, bc2, keep_prob, is_training)
    dense2 = dense_block(conv2, w2, b2, keep_prob, is_training)
    pool2 = max_pool(dense2, pool_size)

    conv3 = conv2d_relu(pool2, wc3, bc3, keep_prob, is_training)
    dense3 = dense_block(conv3, w3, b3, keep_prob, is_training)
    pool3 = max_pool(dense3, pool_size)

    conv4 = conv2d_relu(pool3, wc4, bc4, keep_prob, is_training)
    dense4 = dense_block(conv4, w4, b4, keep_prob, is_training)
    pool4 = max_pool(dense4, pool_size)

    conv5 = conv2d_relu(pool4, wc5, bc5, keep_prob, is_training)
    dense5 = dense_block(conv5, w5, b5, keep_prob, is_training)

    # Up layers
    up6 = tf.nn.relu(deconv2d(dense5, wd6, pool_size) + bd6)
    merge6 = crop_and_concat(dense4, up6)
    conv6 = basic_block(merge6, w6, b6, keep_prob, is_training)

    up7 = tf.nn.relu(deconv2d(conv6, wd7, pool_size) + bd7)
    merge7 = crop_and_concat(dense3, up7)
    conv7 = basic_block(merge7, w7, b7, keep_prob, is_training)

    up8 = tf.nn.relu(deconv2d(conv7, wd8, pool_size) + bd8)
    merge8 = crop_and_concat(dense2, up8)
    conv8 = basic_block(merge8, w8, b8, keep_prob, is_training)

    up9 = tf.nn.relu(deconv2d(conv8, wd9, pool_size) + bd9)
    merge9 = crop_and_concat(dense1, up9)
    conv9 = basic_block(merge9, w9, b9, keep_prob, is_training)

    # Output Map
    stddev = np.sqrt(2 / (filter_size ** 2 * 128))
    weight = weight_variable([1, 1, features_root, n_class], stddev)
    bias = bias_variable([n_class], name="bias")
    conv_out = conv2d(conv9, weight, bias, tf.constant(1.0))
    output_map = tf.nn.sigmoid(conv_out)

    variables = w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8 + w9 + b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9

    return output_map, variables


def create_attention_gate_res_unet(x, keep_prob, channels, layers, n_class=4, features_root=16, filter_size=3, pool_size=2,
                                   is_training=True):
    """
    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """

    logging.info(
        "Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{"
        "pool_size}".format(
            layers=layers,
            features=features_root,
            filter_size=filter_size,
            pool_size=pool_size))

    # Placeholder for the input image
    with tf.name_scope("preprocessing"):
        nx = tf.shape(x)[1]
        ny = tf.shape(x)[2]
        x_image = tf.reshape(x, tf.stack([-1, nx, ny, channels]))
        in_node = x_image

    w1, b1 = calc_res_down_weight_bias(0, filter_size, features_root)
    w2, b2 = calc_res_down_weight_bias(1, filter_size, features_root)
    w3, b3 = calc_res_down_weight_bias(2, filter_size, features_root)
    w4, b4 = calc_res_down_weight_bias(3, filter_size, features_root)
    w5, b5 = calc_res_down_weight_bias(4, filter_size, features_root)

    wd6, bd6, w6, b6 = calc_upblock_weight_bias(3, filter_size, pool_size, features_root)
    wa6, ba6 = calc_att_weight_bias(3, filter_size, features_root)
    wd7, bd7, w7, b7 = calc_upblock_weight_bias(2, filter_size, pool_size, features_root)
    wa7, ba7 = calc_att_weight_bias(2, filter_size, features_root)
    wd8, bd8, w8, b8 = calc_upblock_weight_bias(1, filter_size, pool_size, features_root)
    wa8, ba8 = calc_att_weight_bias(1, filter_size, features_root)
    wd9, bd9, w9, b9 = calc_upblock_weight_bias(0, filter_size, pool_size, features_root)
    wa9, ba9 = calc_att_weight_bias(0, filter_size, features_root)

    # Down layers
    res1 = residual_block(in_node, w1, b1, keep_prob, is_training)
    pool1 = max_pool(res1, pool_size)

    res2 = residual_block(pool1, w2, b2, keep_prob, is_training)
    pool2 = max_pool(res2, pool_size)

    res3 = residual_block(pool2, w3, b3, keep_prob, is_training)
    pool3 = max_pool(res3, pool_size)

    res4 = residual_block(pool3, w4, b4, keep_prob, is_training)
    pool4 = max_pool(res4, pool_size)

    res5 = residual_block(pool4, w5, b5, keep_prob, is_training)

    # Up layers
    up6 = tf.nn.relu(deconv2d(res5, wd6, pool_size) + bd6)
    att6 = attention_gate(res4, up6, wa6, ba6, keep_prob)
    merge6 = crop_and_concat(res4, att6)
    conv6 = basic_block(merge6, w6, b6, keep_prob, is_training)

    up7 = tf.nn.relu(deconv2d(conv6, wd7, pool_size) + bd7)
    att7 = attention_gate(res3, up7, wa7, ba7, keep_prob)
    merge7 = crop_and_concat(res3, att7)
    conv7 = basic_block(merge7, w7, b7, keep_prob, is_training)

    up8 = tf.nn.relu(deconv2d(conv7, wd8, pool_size) + bd8)
    att8 = attention_gate(res2, up8, wa8, ba8, keep_prob)
    merge8 = crop_and_concat(res2, att8)
    conv8 = basic_block(merge8, w8, b8, keep_prob, is_training)

    up9 = tf.nn.relu(deconv2d(conv8, wd9, pool_size) + bd9)
    att9 = attention_gate(res1, up9, wa9, ba9, keep_prob)
    merge9 = crop_and_concat(res1, att9)
    conv9 = basic_block(merge9, w9, b9, keep_prob, is_training)

    # Output Map
    stddev = np.sqrt(2 / (filter_size ** 2 * 128))
    weight = weight_variable([1, 1, features_root, n_class], stddev)
    bias = bias_variable([n_class], name="bias")
    conv_out = conv2d(conv9, weight, bias, tf.constant(1.0))
    output_map = tf.nn.sigmoid(conv_out)

    variables = w1 + w2 + w3 + w4 + w5 + w6 + wa6 + w7 + wa7 + w8 + wa8 + w9 + wa9 \
                + b1 + b2 + b3 + b4 + b5 + b6 + ba9 + b7 + ba7 + b8 + ba8 + b9 + ba9

    return output_map, variables


def create_se_res_unet(x, keep_prob, channels, layers, n_class=4, features_root=16, filter_size=3, pool_size=2,
                       is_training=True):
    """
    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """

    logging.info(
        "Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{"
        "pool_size}".format(
            layers=layers,
            features=features_root,
            filter_size=filter_size,
            pool_size=pool_size))

    # Placeholder for the input image
    with tf.name_scope("preprocessing"):
        nx = tf.shape(x)[1]
        ny = tf.shape(x)[2]
        x_image = tf.reshape(x, tf.stack([-1, nx, ny, channels]))
        in_node = x_image
        # batch_size = tf.shape(x_image)[0]

    w1, b1 = calc_res_down_weight_bias(0, filter_size, features_root)
    w2, b2 = calc_res_down_weight_bias(1, filter_size, features_root)
    w3, b3 = calc_res_down_weight_bias(2, filter_size, features_root)
    w4, b4 = calc_res_down_weight_bias(3, filter_size, features_root)
    w5, b5 = calc_res_down_weight_bias(4, filter_size, features_root)

    wd6, bd6, w6, b6 = calc_SE_up_weight_bias(3, filter_size, pool_size, features_root)
    wd7, bd7, w7, b7 = calc_SE_up_weight_bias(2, filter_size, pool_size, features_root)
    wd8, bd8, w8, b8 = calc_SE_up_weight_bias(1, filter_size, pool_size, features_root)
    wd9, bd9, w9, b9 = calc_SE_up_weight_bias(0, filter_size, pool_size, features_root)

    # Down layers
    se1 = SE_down_block(in_node, w1, b1, 64, keep_prob, is_training)
    pool1 = max_pool(se1, pool_size)

    se2 = SE_down_block(pool1, w2, b2, 128, keep_prob, is_training)
    pool2 = max_pool(se2, pool_size)

    se3 = SE_down_block(pool2, w3, b3, 256, keep_prob, is_training)
    pool3 = max_pool(se3, pool_size)

    se4 = SE_down_block(pool3, w4, b4, 512, keep_prob, is_training)
    pool4 = max_pool(se4, pool_size)

    se5 = SE_down_block(pool4, w5, b5, 1024, keep_prob, is_training)

    # Up layers
    up6 = tf.nn.relu(deconv2d(se5, wd6, pool_size) + bd6)
    merge6 = crop_and_concat(se4, up6)
    se6 = SE_up_block(merge6, w6, b6, 512, keep_prob, is_training)

    up7 = tf.nn.relu(deconv2d(se6, wd7, pool_size) + bd7)
    merge7 = crop_and_concat(se3, up7)
    se7 = SE_up_block(merge7, w7, b7, 256, keep_prob, is_training)

    up8 = tf.nn.relu(deconv2d(se7, wd8, pool_size) + bd8)
    merge8 = crop_and_concat(se2, up8)
    se8 = SE_up_block(merge8, w8, b8, 128, keep_prob, is_training)

    up9 = tf.nn.relu(deconv2d(se8, wd9, pool_size) + bd9)
    merge9 = crop_and_concat(se1, up9)
    se9 = SE_up_block(merge9, w9, b9, 64, keep_prob, is_training)

    # Output Map
    stddev = np.sqrt(2 / (filter_size ** 2 * 128))
    weight = weight_variable([1, 1, features_root, n_class], stddev)
    bias = bias_variable([n_class], name="bias")
    conv_out = conv2d(se9, weight, bias, tf.constant(1.0))
    output_map = tf.nn.sigmoid(conv_out)

    variables = w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8 + w9 + b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9

    return output_map, variables


def create_res_da_unet(x, keep_prob, channels, layers, n_class=4, features_root=16, filter_size=3, pool_size=2,
                       is_training=True, attention_model='dual'):
    """
    :param attention_model:
    :param is_training:
    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    """

    logging.info(
        "Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{"
        "pool_size}".format(
            layers=layers,
            features=features_root,
            filter_size=filter_size,
            pool_size=pool_size))

    # Placeholder for the input image
    with tf.name_scope("preprocessing"):
        nx = tf.shape(x)[1]
        ny = tf.shape(x)[2]
        x_image = tf.reshape(x, tf.stack([-1, nx, ny, channels]))
        in_node = x_image
        # batch_size = tf.shape(x_image)[0]

    w1, b1 = calc_res_down_weight_bias(0, filter_size, features_root)
    w2, b2 = calc_res_down_weight_bias(1, filter_size, features_root)
    w3, b3 = calc_res_down_weight_bias(2, filter_size, features_root)
    w4, b4 = calc_res_down_weight_bias(3, filter_size, features_root)
    w5, b5 = calc_res_down_weight_bias(4, filter_size, features_root)

    wd6, bd6, w6, b6 = calc_upblock_weight_bias(3, filter_size, pool_size, features_root)
    wd7, bd7, w7, b7 = calc_upblock_weight_bias(2, filter_size, pool_size, features_root)
    wd8, bd8, w8, b8 = calc_upblock_weight_bias(1, filter_size, pool_size, features_root)
    wd9, bd9, w9, b9 = calc_upblock_weight_bias(0, filter_size, pool_size, features_root)

    # Down layers
    res1 = residual_block(in_node, w1, b1, keep_prob, is_training)
    pool1 = max_pool(res1, pool_size)

    res2 = residual_block(pool1, w2, b2, keep_prob, is_training)
    pool2 = max_pool(res2, pool_size)

    res3 = residual_block(pool2, w3, b3, keep_prob, is_training)
    pool3 = max_pool(res3, pool_size)

    res4 = residual_block(pool3, w4, b4, keep_prob, is_training)
    pool4 = max_pool(res4, pool_size)

    res5 = residual_block(pool4, w5, b5, keep_prob, is_training)

    # DA Head
    if attention_model == 'dual':
        head = da_head(res5, 16, 16, 1024, keep_prob, is_training)
    elif attention_model == 'channel-wise':
        head = ca_head(res5, 16, 16, 1024, keep_prob, is_training)
    elif attention_model == 'spatial':
        head = pa_head(res5, 16, 16, 1024, keep_prob, is_training)
    # Up layers
    up6 = tf.nn.relu(deconv2d(head, wd6, pool_size) + bd6)
    merge6 = crop_and_concat(res4, up6)
    conv6 = basic_block(merge6, w6, b6, keep_prob, is_training)

    up7 = tf.nn.relu(deconv2d(conv6, wd7, pool_size) + bd7)
    merge7 = crop_and_concat(res3, up7)
    conv7 = basic_block(merge7, w7, b7, keep_prob, is_training)

    up8 = tf.nn.relu(deconv2d(conv7, wd8, pool_size) + bd8)
    merge8 = crop_and_concat(res2, up8)
    conv8 = basic_block(merge8, w8, b8, keep_prob, is_training)

    up9 = tf.nn.relu(deconv2d(conv8, wd9, pool_size) + bd9)
    merge9 = crop_and_concat(res1, up9)
    conv9 = basic_block(merge9, w9, b9, keep_prob, is_training)

    # Output Map
    stddev = np.sqrt(2 / (filter_size ** 2 * 128))
    weight = weight_variable([1, 1, features_root, n_class], stddev)
    bias = bias_variable([n_class], name="bias")
    conv_out = conv2d(conv9, weight, bias, tf.constant(1.0))
    output_map = tf.nn.sigmoid(conv_out)

    variables = w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8 + w9 + b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9

    return output_map, variables


