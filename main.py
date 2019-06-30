""" Main file for running the model """

import tensorflow as tf

from rnn import Data, Model

if __name__ == '__main__':
    model = Model.build_model(65, 256, 1024, 64)
    for input_example_batch, target_example_batch in Data.get_training_dataset().take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

        si = tf.random.multinomial(example_batch_predictions[0], num_samples=1)
        sampled_indices = tf.squeeze(si, axis=-1).numpy()
        print(sampled_indices)
        print(Model.loss(target_example_batch, example_batch_predictions))
        print('Prediction shape: ', example_batch_predictions.shape, " # (batch size, seq len, vocab size)")
        print('scalar loss:      ', example_batch_predictions.numpy().mean())
        Model.train_model(model)
