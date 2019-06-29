""" Holds the Data class """

import tensorflow as tf

import rnn


class Data:
    """ Train holds functions responsible for producing training examples from Shakespearian text """

    @staticmethod
    def get_sequences():
        """
        Returns batch sequences of the training text
        :return: [sequences]
        """
        seq_length = 100
        text_as_int = rnn.Vectorize.get_text_as_int()
        char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
        return char_dataset.batch(seq_length + 1, drop_remainder=True)

    @staticmethod
    def split_input_target(chunk):
        """
        Takes a chunk of a sequence and splits it according to a source and a target
        :param chunk: string - chunk of sequence
        :return: tuple - (input, target) e.g chunk = hello => (hell, ello)
        """
        input_txt = chunk[:-1]
        target_txt = chunk[1:]
        return input_txt, target_txt

    @staticmethod
    def get_dataset():
        """
        Returns the training dataset
        :return: [(input, target)]
        """
        sequences = Data.get_sequences()
        return sequences.map(Data.split_input_target)

    @staticmethod
    def get_training_batches():
        """
        Get a training batch
        :return: []
        """
        batch_size = 64
        # examples_per_epoch = len(rnn.Load.get_text())
        buffer_size = 10000  # buffer size to shuffle the dataset
        dataset = Data.get_dataset()
        return dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
