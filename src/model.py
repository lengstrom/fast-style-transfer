import os
import functools
import vgg
import tensorflow as tf
import transform
from utils import get_img

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'


# pre-compute style features
def pre_compute_style_features(vgg_path, style_image_path):
    style_image = tf.constant(get_img(style_image_path), dtype=tf.float32)
    style_image = tf.expand_dims(style_image, axis=0)

    style_image_pre = vgg.preprocess(style_image)

    net = vgg.net(vgg_path, style_image_pre)

    style_features = {}
    for layer in STYLE_LAYERS:
        features = net[layer]
        new_shape = (-1, features.shape[3].value)
        features = tf.reshape(features, new_shape)
        gram = tf.matmul(tf.transpose(features), features) / tf.size(features, out_type=tf.float32)
        style_features[layer] = gram
    return style_features


def input_fn(file_dir, batch_size, num_epochs, shared_name):
    # queue with the file names that can be shared amongst workers during training
    filenames = tf.train.match_filenames_once(os.path.join(file_dir, '*.jpg'))
    filename_queue = tf.train.string_input_producer(filenames, shared_name=shared_name, num_epochs=num_epochs,
                                                    shuffle=True)
    image_reader = tf.WholeFileReader()
    _, image_file = image_reader.read(filename_queue)
    image = tf.image.decode_jpeg(image_file, channels=3)
    image = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(image, axis=0), size=(256, 256)))
    min_after_dequeue = batch_size * 10
    capacity = min_after_dequeue + 3 * batch_size
    batch = tf.train.shuffle_batch([image], batch_size=batch_size, capacity=capacity,
                                   min_after_dequeue=min_after_dequeue, allow_smaller_final_batch=False)
    return batch


def model_fn(train_batch, content_weight, style_weight, tv_weight, vgg_path, style_features, batch_size, learning_rate):

    global_step = tf.train.get_or_create_global_step()

    batch_shape = (batch_size, 256, 256, 3)

    X_pre = vgg.preprocess(train_batch)

    # precompute content features
    content_features = {}
    content_net = vgg.net(vgg_path, X_pre)
    content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

    preds = transform.net(train_batch / 255.0)
    preds_pre = vgg.preprocess(preds)

    net = vgg.net(vgg_path, preds_pre)

    content_size = _tensor_size(content_features[CONTENT_LAYER]) * batch_size
    assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])
    content_loss = content_weight * (2 * tf.nn.l2_loss(
        net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size
                                     )

    style_losses = []
    for style_layer in STYLE_LAYERS:
        layer = net[style_layer]
        bs, height, width, filters = map(lambda i: i.value, layer.get_shape())
        size = height * width * filters
        feats = tf.reshape(layer, (bs, height * width, filters))
        feats_T = tf.transpose(feats, perm=[0, 2, 1])
        grams = tf.matmul(feats_T, feats) / size
        style_gram = style_features[style_layer]
        style_losses.append(2 * tf.nn.l2_loss(grams - style_gram) / tf.size(style_gram, out_type=tf.float32))

    style_loss = style_weight * functools.reduce(tf.add, style_losses) / batch_size

    # total variation denoising
    tv_y_size = _tensor_size(preds[:, 1:, :, :])
    tv_x_size = _tensor_size(preds[:, :, 1:, :])
    y_tv = tf.nn.l2_loss(preds[:, 1:, :, :] - preds[:, :batch_shape[1] - 1, :, :])
    x_tv = tf.nn.l2_loss(preds[:, :, 1:, :] - preds[:, :, :batch_shape[2] - 1, :])
    tv_loss = tv_weight * 2 * (x_tv / tv_x_size + y_tv / tv_y_size) / batch_size

    loss = content_loss + style_loss + tv_loss

    # overall loss
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    return train_step, global_step, style_loss, content_loss, tv_loss, loss


def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)
