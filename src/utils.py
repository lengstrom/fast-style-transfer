import tensorflow as tf
import scipy.misc
import numpy as np
from tensorflow.python.lib.io import file_io
from StringIO import StringIO


# the problem with this is that you don't get the shape
def read_jpeg(src):
    filename_queue = tf.train.string_input_producer([src])
    reader = tf.WholeFileReader()
    _, image_file = reader.read(filename_queue)
    image = tf.image.decode_jpeg(image_file, channels=3)
    return image


def get_img(src, img_size=False):
    f = StringIO(file_io.read_file_to_string(src))
    img = scipy.misc.imread(f, mode='RGB')  # misc.imresize(, (256, 256, 3))
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img, img, img))
    if img_size:
        img = scipy.misc.imresize(img, img_size)
    return img


def exists(p, msg):
    assert file_io.file_exists(p), msg


def list_files(in_path):
    file_io.walk(in_path)
    files = []
    for (dirpath, dirnames, filenames) in file_io.walk(in_path):
        files.extend(filenames)
        break

    return files


def save_img(path, image):
    file_io.write_string_to_file(path, image)
