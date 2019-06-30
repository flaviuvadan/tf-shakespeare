""" Model file """

import os

import functools
import tensorflow as tf

from rnn import Data, Load


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
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                      batch_input_shape=[batch_size, None]),
            r_nn(rnn_units,
                 return_sequences=True,
                 recurrent_initializer='glorot_uniform',
                 stateful=True),
            tf.keras.layers.Dense(vocab_size)
        ])
        model.compile(tf.train.AdamOptimizer(),
                      loss=Model.loss)
        return model

    @staticmethod
    def loss(labels, logits):
        """
        Return the loss associated with a prediction. "Measures the probability error in discrete classification tasks
        in which the classes are mutually exclusive"
        :param labels: actual values
        :param logits: predicted values
        :return: categorical cross-entropy between a given label and the predicted values
        """
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

    @staticmethod
    def train_model(model, epochs=3):
        """ Train the given model for the given epochs, should be generated via build_model() """
        checkpoint_callback = Model.get_checkpoint_callback()
        examples_per_epoch = len(Load.get_text()) // 100 // 64
        model.fit(Data.get_training_dataset().repeat(),
                  epochs=epochs,
                  steps_per_epoch=examples_per_epoch,
                  callbacks=[checkpoint_callback])

    @staticmethod
    def get_checkpoint_callback():
        """ Returns the checkpoints for saving the model after each training epoch """
        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True)

    @staticmethod
    def get_model_from_checkpoint():
        """ Returns a model from the 3rd checkpoint """
        checkpoint_dir = './training_checkpoints'
        model = Model.build_model(65, 256, 1024, 1)
        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        model.build(tf.TensorShape([1, None]))
        return model

