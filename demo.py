import sys
sys.path.insert(0, 'src')
import transform, numpy as np, vgg, pdb, os
import scipy.misc
import tensorflow as tf
from PIL import Image
import riseml.server
from utils import save_img, get_img, exists, list_files
from argparse import ArgumentParser
from collections import defaultdict
from io import BytesIO
import time

DEVICE = '/gpu:0'

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        dest='checkpoint_dir',
                        help='dir or .ckpt file to load checkpoint from',
                        metavar='CHECKPOINT', required=True)

    parser.add_argument('--device', type=str,
                        dest='device',help='device to perform compute on',
                        metavar='DEVICE', default=DEVICE)

    parser.add_argument('--input_width', type=int,
                        dest='input_width',help='input image width in pixels',
                        metavar='DEVICE', default=320)

    parser.add_argument('--input_height', type=int,
                        dest='input_height',help='input image height in pixels',
                        metavar='DEVICE', default=240)

    return parser

def check_opts(opts):
    exists(opts.checkpoint_dir, 'Checkpoint not found!')

def main():
    parser = build_parser()
    opts = parser.parse_args()
    check_opts(opts)
    batch_size = 1
    img_shape = (opts.input_height, opts.input_width, 3)
    g = tf.Graph()
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device(DEVICE), \
            tf.Session(config=soft_config) as sess:
        batch_shape = (batch_size,) + img_shape
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')

        preds = transform.net(img_placeholder)
        saver = tf.train.Saver()
        if os.path.isdir(opts.checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(opts.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, opts.checkpoint_dir)

        def transfer_style(input_image):
            input_image = Image.open(BytesIO(input_image))
            image = np.asarray(input_image.convert('RGB'), dtype=np.float32)
            X = np.zeros(batch_shape, dtype=np.float32)
            X[0] = image
            _preds = sess.run(preds, feed_dict={img_placeholder: X})
            img = np.clip(_preds[0], 0, 255).astype(np.uint8)
            result = Image.fromarray(img)
            output_image = BytesIO()
            result.save(output_image, format='JPEG')
            return output_image.getvalue()
        riseml.server.serve(transfer_style, port=os.environ.get('PORT'))


if __name__ == '__main__':
    main()
