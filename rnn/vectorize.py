""" Holds the Vectorize class """

import numpy as np

import rnn


class Vectorize:
    """ Vectorize is responsible for tokenizing/vectorizing the Shakespeare text """

    @staticmethod
    def get_characters_to_index():
        """
        Maps characters to alphabet indices
        :return: dict - {char: index}
        """
        vocabulary = rnn.Load.get_vocabulary()
        return {u: i for i, u in enumerate(vocabulary)}

    @staticmethod
    def get_index_to_characters():
        """
        Maps indices to alphabet characters
        :return: dict - {index: char}
        """
        vocabulary = rnn.Load.get_vocabulary()
        return np.array(vocabulary)
