import json
import os
import time
import tensorflow as tf
from argparse import ArgumentParser
from utils import exists
import random
import model
from EvaluationHook import EvaluationHook

CONTENT_WEIGHT = 7.5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 2e2
LEARNING_RATE = 1e-3
NUM_EPOCHS = 2
CHECKPOINT_ITERATIONS = 2000
VGG_PATH = 'data/imagenet-vgg-verydeep-19.mat'
TRAIN_PATH = 'data/train2014'
BATCH_SIZE = 4


def run(target, cluster_spec, is_chief, job_dir, content_weight, style_path, style_weight, tv_weight, vgg_path, train_path,
        test_image_path, epochs, checkpoint_iterations, batch_size, learning_rate, log_iterations=10):
    # pre compute style features first
    with tf.Graph().as_default():
        with tf.device(tf.train.replica_device_setter()):
            tf_style_features = model.pre_compute_style_features(vgg_path, style_path)
        with tf.Session() as sess:
            style_features = sess.run(tf_style_features)
    # reset the default graph to save RAM (the vgg network is quite heavy)
    tf.reset_default_graph()

    # training hooks
    checkpoint_dir = os.path.join(job_dir, 'checkpoints')
    hooks = list()
    if is_chief and test_image_path:
        hooks.append(EvaluationHook(test_image_path, checkpoint_dir, os.path.join(job_dir, 'test_image_results')))

    with tf.Graph().as_default():
        with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):

            batch = model.input_fn(train_path, batch_size, epochs, 'train_queue')

            tensors = model.model_fn(batch, content_weight, style_weight, tv_weight, vgg_path, style_features,
                                     batch_size, learning_rate)
            if is_chief:
                hooks.append(tf.train.CheckpointSaverHook(checkpoint_dir=checkpoint_dir,
                                                          save_steps=checkpoint_iterations))

        with tf.train.MonitoredTrainingSession(master=target,
                                               is_chief=is_chief,
                                               checkpoint_dir=job_dir,
                                               hooks=hooks,
                                               save_checkpoint_secs=None,
                                               save_summaries_steps=None,
                                               log_step_count_steps=log_iterations) as sess:
            uid = random.randint(1, 100)
            tf.logging.info('UID: %s' % uid)
            while not sess.should_stop():
                start_time = time.time()
                # train step
                # tup = sess.run([train_step, style_loss, content_loss, tv_loss, loss, global_step])
                _, global_step, style_loss, content_loss, tv_loss, loss = sess.run(tensors)
                end_time = time.time()
                delta_time = end_time - start_time

                if global_step % log_iterations == 0:
                    msg = 'Global Step: %s, UID: %s' % (global_step, uid) + '\n' + \
                          'Batch Time: %s, Total Loss: %s' % (delta_time, loss) + '\n' + \
                          'Style: %s, Content:%s, Tv: %s' % (style_loss, content_loss, tv_loss)

                    tf.logging.info(msg)

        tf.logging.info('The thread %s was successfully closed' % uid)


def build_parser():
    parser = ArgumentParser()

    parser.add_argument('--job-dir', type=str,
                        dest='job_dir', help='GCS or local dir for checkpoints, exports, and summaries.'
                                             ' Use an existing directory to load a trained model,'
                                             ' or a new directory to retrain',
                        metavar='JOB_DIR', required=True)

    parser.add_argument('--style-path', type=str,
                        dest='style_path', help='style image path',
                        metavar='STYLE_PATH', required=True)

    parser.add_argument('--train-path', type=str,
                        dest='train_path', help='path to training images folder',
                        metavar='TRAIN_PATH', default=TRAIN_PATH)

    parser.add_argument('--test-image-path', type=str,
                        dest='test_image_path', help='test image path',
                        metavar='TEST_IMAGE_PATH', default=False)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='num epochs',
                        metavar='EPOCHS', default=NUM_EPOCHS)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--checkpoint-iterations', type=int,
                        dest='checkpoint_iterations', help='checkpoint frequency',
                        metavar='CHECKPOINT_ITERATIONS',
                        default=CHECKPOINT_ITERATIONS)

    parser.add_argument('--vgg-path', type=str,
                        dest='vgg_path',
                        help='path to VGG19 network (default %(default)s)',
                        metavar='VGG_PATH', default=VGG_PATH)

    parser.add_argument('--content-weight', type=float,
                        dest='content_weight',
                        help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)

    parser.add_argument('--style-weight', type=float,
                        dest='style_weight',
                        help='style weight (default %(default)s)',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)

    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight',
                        help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)

    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)
    parser.add_argument('--verbosity',
                        choices=[
                            'DEBUG',
                            'ERROR',
                            'FATAL',
                            'INFO',
                            'WARN'
                        ],
                        default='INFO',
                        help='Set logging verbosity')

    return parser


def check_opts(opts):
    exists(opts.job_dir, "checkpoint dir not found!")
    exists(opts.style_path, "style path not found!")
    exists(opts.train_path, "train path not found!")
    if opts.test_image_path:
        exists(opts.test_image_path, "test img not found!")
    exists(opts.vgg_path, "vgg network data not found!")
    assert opts.epochs > 0
    assert opts.batch_size > 0
    assert opts.checkpoint_iterations > 0
    assert opts.content_weight >= 0
    assert opts.style_weight >= 0
    assert opts.tv_weight >= 0
    assert opts.learning_rate >= 0


def dispatch(*args, **kwargs):
    """Parse TF_CONFIG to cluster_spec and call run() method
  TF_CONFIG environment variable is available when running using
  gcloud either locally or on cloud. It has all the information required
  to create a ClusterSpec which is important for running distributed code.
  """

    tf_config = os.environ.get('TF_CONFIG')
    # If TF_CONFIG is not available run local
    if not tf_config:
        return run(target='', cluster_spec=None, is_chief=True, *args, **kwargs)

    tf_config_json = json.loads(tf_config)

    cluster = tf_config_json.get('cluster')
    job_name = tf_config_json.get('task', {}).get('type')
    task_index = tf_config_json.get('task', {}).get('index')

    # If cluster information is empty run local
    if job_name is None or task_index is None:
        return run(target='', cluster_spec=None, is_chief=True, *args, **kwargs)

    cluster_spec = tf.train.ClusterSpec(cluster)
    server = tf.train.Server(cluster_spec,
                             job_name=job_name,
                             task_index=task_index)

    if job_name == 'ps':
        server.join()
        return
    elif job_name in ['master', 'worker']:
        return run(server.target, cluster_spec, is_chief=(job_name == 'master'), *args, **kwargs)


if __name__ == "__main__":

    parser = build_parser()
    options, unknown = parser.parse_known_args()
    check_opts(options)

    # Set python level verbosity
    tf.logging.set_verbosity(options.verbosity)
    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf.logging.__dict__[options.verbosity] / 10)
    del options.verbosity

    if unknown:
        tf.logging.warn('Unknown arguments: {}'.format(unknown))

    dispatch(**options.__dict__)
