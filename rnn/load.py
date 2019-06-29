""" Load Shakespearian dataset """

from __future__ import absolute_import, division, print_function

import numpy as np
import os
import time

import tensorflow as tf
tf.enable_eager_execution()


class Load:
    """ Load is responsible for downloading the Shakespeare text from TF """

    source_url = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'

    def get_path_to_file(self):
        """
        Returns the path to the Shakespeare text
        :return: string - Shakespeare text
        """
        return tf.keras.utils.get_file('shakespeare.txt', self.source_url)

    def get_text(self):
        """ Returns """
        return open(self.get_path_to_file(), 'rb').read().decode(encoding='utf-8')
