""" Load Shakespearian dataset """

import tensorflow as tf


class Load:
    """ Load is responsible for downloading the Shakespeare text from TF """

    @staticmethod
    def get_path_to_file():
        """
        Returns the path to the Shakespeare text
        :return: string - Shakespeare path
        """
        source_url = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
        return tf.keras.utils.get_file('shakespeare.txt', source_url)

    @staticmethod
    def get_text():
        """
        Returns the loaded Shakespeare text
        :return: string - Shakespeare text
        """
        path = Load.get_path_to_file()
        return open(path, 'rb').read().decode(encoding='utf-8')

    @staticmethod
    def get_vocabulary():
        """
        Returns the vocabulary/unique characters of the text
        :return: set - vocabulary
        """
        return sorted(set(Load.get_text()))
