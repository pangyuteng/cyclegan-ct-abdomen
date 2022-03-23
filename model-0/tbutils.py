
# ref 
# https://keras.io/guides/customizing_what_happens_in_fit
# https://gist.github.com/joelthchao/ef6caa586b647c3c032a4f84d52e3a11
# https://stackoverflow.com/questions/43784921/how-to-display-custom-images-in-tensorboard-using-keras
# https://www.tensorflow.org/api_docs/python/tf/summary/image
# https://www.tensorflow.org/tensorboard/migrate

import os
import sys
import random
import numpy as np
import tensorflow as tf

class ImageSummaryCallback(tf.keras.callbacks.Callback):
    def __init__(self, logdir):
        super(ImageSummaryCallback, self).__init__()
        self.logdir = logdir
        self.file_writer = tf.summary.create_file_writer(self.logdir)
        self.count = 0
    def on_epoch_end(self, epoch, logs=None, mydict=None):
        imgA = mydict['img'][:3,].astype(np.uint8)
        imgB = mydict['img'][3:,].astype(np.uint8)
        with self.file_writer.as_default():
            tf.summary.image("ImageA", imgA, step=self.count)
            tf.summary.image("ImageB", imgB, step=self.count)
            self.file_writer.flush()
            self.count+=1

class MetricSummaryCallback(tf.keras.callbacks.Callback):
    def __init__(self, logdir):
        super(MetricSummaryCallback, self).__init__()
        self.logdir = logdir
        self.file_writer = tf.summary.create_file_writer(self.logdir)
        self.count = 0
    def on_epoch_end(self, epoch, logs=None, mydict=None):        
        with self.file_writer.as_default():
            for name, value in mydict.items():
                tf.summary.scalar(name, value, step=self.count)
                self.file_writer.flush()
                self.count+=1
