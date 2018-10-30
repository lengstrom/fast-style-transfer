from __future__ import print_function
import sys, os, pdb
sys.path.insert(0, 'src')
import numpy as np, scipy.misc 
from optimize import optimize
from argparse import ArgumentParser
from utils import save_img, get_img, exists, list_files
import evaluate

CONTENT_WEIGHT = 7.5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 2e2

LEARNING_RATE = 1e-3
NUM_EPOCHS = 3
CHECKPOINT_DIR = 'train/data/ckpts'
LOG_DIR = 'train/log'
CHECKPOINT_ITERATIONS = 10
VGG_PATH = 'train/data/imagenet-vgg-verydeep-19.mat'
TRAIN_PATH = 'train/data/train2014'
BATCH_SIZE = 1
DEVICE = '/cpu:0'
FRAC_GPU = 1

NUM_STEPS = 3

STEP_CONTENT_WEIGHTS = '1, 1.5, 7.5'
STEP_STYLE_WEIGHTS = '0, 10, 100'
STEP_TV_WEIGHTS = '0, 20, 200'

MODEL_PATH = 'train/data/out_models/output_graph.pb'
OUTPUT_NODE_NAME = 'add_22'

STYLE_PATH = 'train/data/styles/la_muse.jpg'


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=str,
                        dest='checkpoint_dir', help='dir to save checkpoint in',
                        metavar='CHECKPOINT_DIR', default=CHECKPOINT_DIR)

    parser.add_argument('--log-dir', type=str,
                        dest='log_dir', help='dir to save logs in',
                        metavar='LOG_DIR', default=LOG_DIR)

    parser.add_argument('--style', type=str,
                        dest='style', help='style image path',
                        metavar='STYLE', default=STYLE_PATH)

    parser.add_argument('--train-path', type=str,
                        dest='train_path', help='path to training images folder',
                        metavar='TRAIN_PATH', default=TRAIN_PATH)

    parser.add_argument('--test', type=str,
                        dest='test', help='test image path',
                        metavar='TEST', default=False)

    parser.add_argument('--test-dir', type=str,
                        dest='test_dir', help='test image save dir',
                        metavar='TEST_DIR', default=False)

    parser.add_argument('--model-path', type=str,
                        dest='model_path', help='path to save pb file',
                        metavar='MODEL_PATH', default=MODEL_PATH)

    parser.add_argument('--output-node-name', type=str,
                        dest='output_node_name', help='output node name',
                        metavar='OUTPUT_NODE_NAME', default=OUTPUT_NODE_NAME)

    parser.add_argument('--use-tiny-net', dest='use_tiny_net', action='store_true',
                        help='use tiny net defined in transform.py or not',
                        default=False)

    parser.add_argument('--slow', dest='slow', action='store_true',
                        help='gatys\' approach (for debugging, not supported)',
                        default=False)

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

    parser.add_argument('--steps', type=int,
                        dest='steps', help='num steps for trainning',
                        metavar='STEPS', default=NUM_STEPS)

    parser.add_argument("--step-content-weights",
                        type=str,
                        dest='step_content_weights',
                        default=STEP_CONTENT_WEIGHTS,
                        help="step content weights.")

    parser.add_argument("--step-style-weights",
                        type=str,
                        dest='step_style_weights',
                        default=STEP_STYLE_WEIGHTS,
                        help="step style weights.")

    parser.add_argument("--step-tv-weights",
                        type=str,
                        dest='step_tv_weights',
                        default=STEP_TV_WEIGHTS,
                        help="step tv weights.")

    return parser


def parse_float_array_from_str(ints_str):
    return [float(int_str) for int_str in ints_str.split(',')]


def check_opts(opts):
    exists(opts.checkpoint_dir, "checkpoint dir not found!")
    exists(opts.log_dir, "log dir not found!")
    exists(opts.style, "style path not found!")
    exists(opts.train_path, "train path not found!")
    if opts.test or opts.test_dir:
        exists(opts.test, "test img not found!")
        exists(opts.test_dir, "test directory not found!")
    exists(opts.vgg_path, "vgg network data not found!")
    assert opts.epochs > 0
    assert opts.batch_size > 0
    assert opts.checkpoint_iterations > 0
    assert os.path.exists(opts.vgg_path)
    assert opts.content_weight >= 0
    assert opts.style_weight >= 0
    assert opts.tv_weight >= 0
    assert opts.learning_rate >= 0


def _get_files(img_dir):
    files = list_files(img_dir)
    return [os.path.join(img_dir,x) for x in files]

    
def main():
    parser = build_parser()
    options = parser.parse_args()
    check_opts(options)

    style_target = get_img(options.style)
    if not options.slow:
        content_targets = _get_files(options.train_path)
    elif options.test:
        content_targets = [options.test]

    epochs = 0
    step_epochs = options.epochs / options.steps
    for n in range(options.steps):
        if options.steps > 1:
            step_content_weights = parse_float_array_from_str(options.step_content_weights)
            step_style_weights = parse_float_array_from_str(options.step_style_weights)
            step_tv_weights = parse_float_array_from_str(options.step_tv_weights)
            assert len(step_content_weights) == options.steps
            assert len(step_style_weights) == options.steps
            assert len(step_tv_weights) == options.steps
            content_weight = step_content_weights[n]
            style_weight = step_style_weights[n]
            tv_weight = step_tv_weights[n]
        else:
            content_weight = options.content_weight
            style_weight = options.style_weight
            tv_weight = options.tv_weight

        epochs = epochs + step_epochs
        if epochs > options.epochs:
            epochs = options.epochs

        kwargs = {
            "slow": options.slow,
            "epochs": epochs,
            "print_iterations": options.checkpoint_iterations,
            "batch_size": options.batch_size,
            "save_path": os.path.abspath(options.checkpoint_dir),
            "log_dir": os.path.abspath(options.log_dir),
            "learning_rate": options.learning_rate,
            "use_tiny_net": options.use_tiny_net,
        }

        if options.slow:
            if options.epochs < 10:
                kwargs['epochs'] = 1000
            if options.learning_rate < 1:
                kwargs['learning_rate'] = 1e1

        args = [
            content_targets,
            style_target,
            content_weight,
            style_weight,
            tv_weight,
            options.vgg_path
        ]

        for preds, losses, i, epoch in optimize(*args, **kwargs):
            style_loss, content_loss, tv_loss, loss = losses

            print('Epoch %d, Iteration: %d, Loss: %s' % (epoch, i, loss))
            to_print = (style_loss, content_loss, tv_loss)
            print('style: %s, content:%s, tv: %s' % to_print)
            if options.test:
                assert options.test_dir != False
                preds_path = '%s/%s_%s.png' % (options.test_dir,epoch,i)
                if not options.slow:
                    evaluate.ffwd_to_img(options.test,
                                         preds_path,
                                         options.checkpoint_dir,
                                         options.model_path,
                                         use_tiny_net=options.use_tiny_net)
                else:
                    save_img(preds_path, img)
    ckpt_dir = options.checkpoint_dir
    cmd_text = 'python evaluate.py --checkpoint %s ...' % ckpt_dir
    print("Training complete. For evaluation:\n    `%s`" % cmd_text)


if __name__ == '__main__':
    main()
