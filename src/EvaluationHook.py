import tensorflow as tf
import threading
import transform
import os
from utils import get_img, save_img
from tensorflow.python.lib.io import file_io


class EvaluationHook(tf.train.SessionRunHook):
    def __init__(self, image_path, checkpoint_dir, result_image_dir, eval_every=1):

        self.image_path = image_path
        self.checkpoint_dir = checkpoint_dir
        self.result_image_dir = result_image_dir
        self.eval_every = eval_every  # eval every x checkpoints
        self.latest_checkpoint = None
        self.checkpoints_since_eval = 0
        self.graph = tf.Graph()

        # the image is being stored in memory at this point so don't use this type of code with multiple images
        with self.graph.as_default():
            img = get_img(self.image_path)
            tf_img = tf.constant(img, dtype=tf.float32)
            tf_img = tf.expand_dims(tf_img, axis=0)
            tf_pred = tf.squeeze(transform.net(tf_img))
            tf_pred = tf.cast(tf.clip_by_value(tf_pred, 0, 255), dtype=tf.uint8)
            self.global_step = tf.train.get_or_create_global_step()
            self.jpeg_image = tf.image.encode_jpeg(tf_pred)
            self.saver = tf.train.Saver()

        # MonitoredTrainingSession runs hooks in background threads
        # and it doesn't wait for the thread from the last session.run()
        # call to terminate to invoke the next hook, hence locks.
        self._eval_lock = threading.Lock()
        self._checkpoint_lock = threading.Lock()

    def after_run(self, run_context, run_values):
        # Always check for new checkpoints in case a single evaluation
        # takes longer than checkpoint frequency and _eval_every is >1
        self._update_latest_checkpoint()

        if self._eval_lock.acquire(False):
            try:
                if self.checkpoints_since_eval >= self.eval_every:
                    self.checkpoints_since_eval = 0
                    self._run_eval()
            finally:
                self._eval_lock.release()

    def _update_latest_checkpoint(self):
        """Update the latest checkpoint file created in the output dir."""
        if self._checkpoint_lock.acquire(False):
            try:
                latest = tf.train.latest_checkpoint(self.checkpoint_dir)
                if not latest == self.latest_checkpoint:
                    self.checkpoints_since_eval += 1
                    self.latest_checkpoint = latest
            finally:
                self._checkpoint_lock.release()

    def end(self, session):
        """Called at then end of session to make sure we always evaluate."""
        self._update_latest_checkpoint()

        with self._eval_lock:
            self._run_eval()

    def _run_eval(self):

        with tf.Session(graph=self.graph) as sess:

            if file_io.is_directory(self.checkpoint_dir):
                ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    self.saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    raise Warning("No checkpoint found... Not going to save result image")
            else:
                self.saver.restore(sess, self.checkpoint_dir)

            image, global_step = sess.run([self.jpeg_image, self.global_step])

            result_image_path = os.path.join(self.result_image_dir, 'result_image_' + str(global_step) + '.jpg')
            if not file_io.is_directory(self.result_image_dir):
                file_io.create_dir(self.result_image_dir)
            save_img(result_image_path, image)
            tf.logging.info('Saved image for global step %s in %s' % (global_step, result_image_path))





