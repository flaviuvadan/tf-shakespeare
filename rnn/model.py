""" Model file """

import functools
import tensorflow as tf

from rnn import Data


class Model:
    """ Model class """

    @staticmethod
    def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
        """
        Get the Shakespeare model
        :param vocab_size: int - number of unique characters in the training text
        :param embedding_dim: input layer dimension
        :param rnn_units: number of RNN units
        :param batch_size: number of training examples per batch
        :return: model - linear stack of layers
        """
        r_nn = functools.partial(tf.keras.layers.GRU,  # Gated Recurrent Unit
                                 recurrent_activation='sigmoid')
        return tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                      batch_input_shape=[batch_size, None]),
            r_nn(rnn_units,
                 return_sequences=True,
                 recurrent_initializer='glorot_uniform',
                 stateful=True),
            tf.keras.layers.Dense(vocab_size)
        ])
