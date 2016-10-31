from argparse import ArgumentParser
import sys
sys.path.insert(0, 'src')
import os, random, subprocess, evaluate, shutil
from utils import exists, list_files
import pdb

TMP_DIR = '.fns_frames_%s/' % random.randint(0,99999)
DEVICE = '/gpu:0'
BATCH_SIZE = 4

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        dest='checkpoint', help='checkpoint directory or .ckpt file',
                        metavar='CHECKPOINT', required=True)

    parser.add_argument('--in-path', type=str,
                        dest='in_path', help='in video path',
                        metavar='IN_PATH', required=True)
    
    parser.add_argument('--out-path', type=str,
                        dest='out', help='path to save processed video to',
                        metavar='OUT', required=True)
    
    parser.add_argument('--tmp-dir', type=str, dest='tmp_dir',
                        help='tmp dir for processing', metavar='TMP_DIR',
                        default=TMP_DIR)

    parser.add_argument('--device', type=str, dest='device',
                        help='device for eval. CPU discouraged. ex: \'/gpu:0\'',
                        metavar='DEVICE', default=DEVICE)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size',help='batch size for eval. default 4.',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)
    return parser

def check_opts(opts):
    exists(opts.checkpoint)
    exists(opts.out)

def main():
    parser = build_parser()
    opts = parser.parse_args()
    
    in_dir = os.path.join(opts.tmp_dir, 'in')
    out_dir = os.path.join(opts.tmp_dir, 'out')
    if not os.path.exists(in_dir):
        os.makedirs(in_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    in_args = [
        'ffmpeg',
        '-i', opts.in_path,
        '%s/frame_%%d.png' % in_dir
    ]

    subprocess.call(" ".join(in_args), shell=True)
    base_names = list_files(in_dir)
    in_files = map(lambda x: os.path.join(in_dir, x), base_names)
    out_files = map(lambda x: os.path.join(out_dir, x), base_names)
    evaluate.ffwd(in_files, out_files, opts.checkpoint, device_t=opts.device,
                  batch_size=opts.batch_size)
    fr = 30 # wtf
    out_args = [
        'ffmpeg',
        '-i', '%s/frame_%%d.png' % out_dir,
        '-f', 'mp4',
        '-q:v', '0',
        '-vcodec', 'mpeg4',
        '-r', str(fr),
        opts.out
    ]

    subprocess.call(" ".join(out_args), shell=True)
    print 'Video at: %s' % opts.out
    shutil.rmtree(opts.tmp_dir)
 
if __name__ == '__main__':
    main()


