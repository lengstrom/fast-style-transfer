import tensorflow as tf, pdb

WEIGHTS_INIT_STDEV = .1


def net(image):
    conv1 = _conv_layer(image, 32, 9, 1)
    conv2 = _conv_layer(conv1, 64, 3, 2)
    conv3 = _conv_layer(conv2, 128, 3, 2)
    resid1 = _residual_block(conv3, 128, 128, 3, 3)
    resid2 = _residual_block(resid1, 128, 128, 3, 3)
    resid3 = _residual_block(resid2, 128, 128, 3, 3)
    resid4 = _residual_block(resid3, 128, 128, 3, 3)
    resid5 = _residual_block(resid4, 128, 128, 3, 3)
    conv_t1 = _conv_tranpose_layer(resid5, 64, 3, 2)
    conv_t2 = _conv_tranpose_layer(conv_t1, 32, 3, 2)
    conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False)
    preds = tf.nn.tanh(conv_t3) * 150 + 255./2
    return preds


def tiny_net(image):
    conv1 = _conv_layer(image, 8, 5, 1)
    conv2 = _conv_layer(conv1, 16, 3, 2)
    conv3 = _conv_layer(conv2, 16, 3, 2)
    resid1 = _residual_block(conv3, 16, 16, 3, 3)
    resid2 = _residual_block(resid1, 16, 16, 3, 3)
    conv_t1 = _conv_tranpose_layer(resid2, 16, 3, 2)
    conv_t2 = _conv_tranpose_layer(conv_t1, 8, 3, 2)
    conv_t3 = _conv_layer(conv_t2, 3, 5, 1, relu=False)
    preds = tf.nn.tanh(conv_t3) * 150 + 127.5
    return preds


def _conv_layer(net, num_filters, filter_size, strides, relu=True):
    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
    net = _instance_norm(net)
    if relu:
        net = tf.nn.relu(net)

    return net


def _depthwise_conv_layer(net, num_filters, filter_size, strides, relu=True):
    weights_init = _depthwise_conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    net = tf.nn.depthwise_conv2d(net, weights_init, strides_shape, padding='SAME')
    net = _instance_norm(net)
    if relu:
        net = tf.nn.relu(net)
    return net


def _conv_tranpose_layer(net, num_filters, filter_size, strides):
    weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)

    batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
    input_shape = tf.shape(net)
    new_shape = [input_shape[0], input_shape[1]*strides, input_shape[2]*strides, num_filters]
    tf_shape = tf.stack(new_shape)
    strides_shape = [1, strides, strides, 1]

    net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
    net.set_shape([batch_size if batch_size else None, int(rows * strides) if rows else None,
                   int(cols * strides) if cols else None, num_filters])
    net = _instance_norm(net)
    return tf.nn.relu(net)


def _residual_block(net, out_channel_0=128, out_channel_1=128, filter_size_0=3, filter_size_1=3):
    tmp = _conv_layer(net, out_channel_0, filter_size_0, 1)
    return net + _conv_layer(tmp, out_channel_1, filter_size_1, 1, relu=False)


def _depthwise_residual_block(net, out_channel_0=64, out_channel_1=64, filter_size_0=3, filter_size_1=3):
    tmp = _depthwise_conv_layer(net, out_channel_0, filter_size_0, 1)
    return net + _depthwise_conv_layer(tmp, out_channel_1, filter_size_1, 1, relu=False)


def _instance_norm(net, train=True):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift


def _conv_init_vars(net, out_channels, filter_size, transpose=False):
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)
    return weights_init


def _depthwise_conv_init_vars(net, out_channels, filter_size, transpose=False):
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, int(out_channels / in_channels)]
    else:
        weights_shape = [filter_size, filter_size, out_channels, int(in_channels / out_channels)]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)
    return weights_init
