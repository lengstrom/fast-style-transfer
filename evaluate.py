from __future__ import print_function
import sys
sys.path.insert(0, 'src')
import transform, numpy as np, vgg, pdb, os
import scipy.misc
import tensorflow as tf
from utils import save_img, get_img, exists, list_files
from argparse import ArgumentParser
from collections import defaultdict
import time
import json
import subprocess
import numpy
from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.video.io.ffmpeg_writer as ffmpeg_writer

BATCH_SIZE = 1
DEVICE = '/cpu:0'

# IN_PATH = 'train/data/inputs/'
# OUT_PATH = 'train/data/outputs/'
# CKPT_PATH = 'train/data/ckpts/'
# MODEL_PATH = 'train/data/out_models/output_graph.pb'
OUTPUT_NODE_NAME = 'add_22'


def ffwd_video(path_in, path_out, checkpoint_dir, device_t='/gpu:0', batch_size=4):
    video_clip = VideoFileClip(path_in, audio=False)
    video_writer = ffmpeg_writer.FFMPEG_VideoWriter(path_out, video_clip.size, video_clip.fps, codec="libx264",
                                                    preset="medium", bitrate="2000k",
                                                    audiofile=path_in, threads=None,
                                                    ffmpeg_params=None)

    g = tf.Graph()
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device(device_t), \
            tf.Session(config=soft_config) as sess:
        batch_shape = (batch_size, video_clip.size[1], video_clip.size[0], 3)
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')

        preds = transform.net(img_placeholder)
        saver = tf.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)

        X = np.zeros(batch_shape, dtype=np.float32)

        def style_and_write(count):
            for i in range(count, batch_size):
                X[i] = X[count - 1]  # Use last frame to fill X
            _preds = sess.run(preds, feed_dict={img_placeholder: X})
            for i in range(0, count):
                video_writer.write_frame(np.clip(_preds[i], 0, 255).astype(np.uint8))

        frame_count = 0  # The frame count that written to X
        for frame in video_clip.iter_frames():
            X[frame_count] = frame
            frame_count += 1
            if frame_count == batch_size:
                style_and_write(frame_count)
                frame_count = 0

        if frame_count != 0:
            style_and_write(frame_count)

        video_writer.close()


# get img_shape
def ffwd(data_in,
         paths_out,
         checkpoint_dir,
         model_path,
         output_node_name,
         device_t='/cpu:0',
         batch_size=1,
         is_tiny_net=False):
    assert len(paths_out) > 0
    is_paths = type(data_in[0]) == str
    if is_paths:
        assert len(data_in) == len(paths_out)
        img_shape = get_img(data_in[0]).shape
    else:
        assert data_in.size[0] == len(paths_out)
        img_shape = X[0].shape

    g = tf.Graph()
    batch_size = min(len(paths_out), batch_size)
    curr_num = 0
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device(device_t), \
            tf.Session(config=soft_config) as sess:
        batch_shape = (batch_size,) + img_shape
        input_shape = (None, None, None, 3)
        img_placeholder = tf.placeholder(tf.float32, shape=input_shape,
                                         name='img_placeholder')
        if is_tiny_net:
            preds = transform.tiny_net(img_placeholder)
        else:
            preds = transform.net(img_placeholder)
        print(preds)
        saver = tf.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                out_node_names = [output_node_name]
                frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                                sess.graph_def,
                                                                                out_node_names)
                with open(model_path, 'wb') as f:
                    f.write(frozen_graph_def.SerializeToString())
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)

        num_iters = int(len(paths_out)/batch_size)
        for i in range(num_iters):
            pos = i * batch_size
            curr_batch_out = paths_out[pos:pos+batch_size]
            if is_paths:
                curr_batch_in = data_in[pos:pos+batch_size]
                X = np.zeros(batch_shape, dtype=np.float32)
                for j, path_in in enumerate(curr_batch_in):
                    img = get_img(path_in)
                    assert img.shape == img_shape, \
                        'Images have different dimensions. ' +  \
                        'Resize images or use --allow-different-dimensions.'
                    X[j] = img
            else:
                X = data_in[pos:pos+batch_size]

            _preds = sess.run(preds, feed_dict={img_placeholder:X})
            for j, path_out in enumerate(curr_batch_out):
                save_img(path_out, _preds[j])
                
        remaining_in = data_in[num_iters*batch_size:]
        remaining_out = paths_out[num_iters*batch_size:]
    if len(remaining_in) > 0:
        ffwd(remaining_in,
             remaining_out,
             checkpoint_dir,
             model_path,
             output_node_name,
             device_t=device_t,
             batch_size=1,
             is_tiny_net=is_tiny_net)


def ffwd_to_img(in_path,
                out_path,
                checkpoint_dir,
                model_path=MODEL_PATH,
                output_node_name=OUTPUT_NODE_NAME,
                device='/cpu:0',
                use_tiny_net=False,):
    paths_in, paths_out = [in_path], [out_path]
    ffwd(paths_in,
         paths_out,
         checkpoint_dir,
         model_path,
         output_node_name,
         batch_size=1,
         device_t=device,
         is_tiny_net=use_tiny_net)


def ffwd_different_dimensions(in_path,
                              out_path,
                              checkpoint_dir,
                              model_path,
                              output_node_name,
                              device_t=DEVICE,
                              batch_size=1,
                              use_tiny_net=False):
    in_path_of_shape = defaultdict(list)
    out_path_of_shape = defaultdict(list)
    for i in range(len(in_path)):
        in_image = in_path[i]
        out_image = out_path[i]
        shape = "%dx%dx%d" % get_img(in_image).shape
        in_path_of_shape[shape].append(in_image)
        out_path_of_shape[shape].append(out_image)
    for shape in in_path_of_shape:
        print('Processing images of shape %s' % shape)
        ffwd(in_path_of_shape[shape],
             out_path_of_shape[shape],
             checkpoint_dir,
             model_path,
             output_node_name,
             device_t,
             batch_size,
             is_tiny_net=use_tiny_net)


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        dest='checkpoint_dir',
                        help='dir or .ckpt file to load checkpoint from',
                        metavar='CHECKPOINT', required=True)
    #                    default=CKPT_PATH)

    parser.add_argument('--in-path', type=str,
                        dest='in_path',help='dir or file to transform',
                        metavar='IN_PATH', required=True)
    #                    default=IN_PATH)

    help_out = 'destination (dir or file) of transformed file or files'
    parser.add_argument('--out-path', type=str,
                        dest='out_path', help=help_out, metavar='OUT_PATH',
                        required=True)
    #                    default=OUT_PATH)

    help_model = 'destination (dir or file) of transformed model'
    parser.add_argument('--model-path', type=str,
                        dest='model_path',
                        help=help_model, metavar='MODEL_PATH',
                        required=True)
    #                    default=MODEL_PATH)

    parser.add_argument('--device', type=str,
                        dest='device',help='device to perform compute on',
                        metavar='DEVICE', default=DEVICE)

    parser.add_argument('--output-node-name', type=str,
                        dest='output_node_name',help='output node name',
                        metavar='OUTPUT_NODE_NAME',
                        default=OUTPUT_NODE_NAME)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size',help='batch size for feedforwarding',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--allow-different-dimensions', action='store_true',
                        dest='allow_different_dimensions', 
                        help='allow different image dimensions')

    parser.add_argument('--use-tiny-net', action='store_true',
                        dest='use_tiny_net',
                        help='use tiny net')

    return parser


def check_opts(opts):
    exists(opts.checkpoint_dir, 'Checkpoint not found!')
    exists(opts.in_path, 'In path not found!')
    if os.path.isdir(opts.out_path):
        exists(opts.out_path, 'out dir not found!')
        assert opts.batch_size > 0


def main():
    parser = build_parser()
    opts = parser.parse_args()
    check_opts(opts)

    if not os.path.isdir(opts.in_path):
        if os.path.exists(opts.out_path) and os.path.isdir(opts.out_path):
            out_path = \
                    os.path.join(opts.out_path,os.path.basename(opts.in_path))
        else:
            out_path = opts.out_path

        ffwd_to_img(opts.in_path, out_path, opts.checkpoint_dir,
                    opts.model_path, opts.output_node_name,
                    device=opts.device, use_tiny_net=opts.use_tiny_net)
    else:
        files = list_files(opts.in_path)
        full_in = [os.path.join(opts.in_path,x) for x in files]
        full_out = [os.path.join(opts.out_path,x) for x in files]
        if opts.allow_different_dimensions:
            ffwd_different_dimensions(full_in,
                                      full_out,
                                      opts.checkpoint_dir,
                                      opts.model_path,
                                      opts.output_node_name,
                                      device_t=opts.device,
                                      batch_size=opts.batch_size,
                                      use_tiny_net=opts.use_tiny_net)
        else :
            ffwd(full_in,
                 full_out,
                 opts.checkpoint_dir,
                 opts.model_path,
                 opts.output_node_name,
                 device_t=opts.device,
                 batch_size=opts.batch_size,
                 is_tiny_net=opts.use_tiny_net)


if __name__ == '__main__':
    main()
