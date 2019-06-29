# tf-shakespeare
TensorFlow Shakespearian RNN practice

## Network shape

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (64, None, 256)           16640     
_________________________________________________________________
gru (GRU)                    (64, None, 1024)          3935232   
_________________________________________________________________
dense (Dense)                (64, None, 65)            66625     
=================================================================
Total params: 4,018,497
Trainable params: 4,018,497
Non-trainable params: 0
_________________________________________________________________
```

#### References
- [TensorFlor text-generation tutorial](https://www.tensorflow.org/tutorials/sequences/text_generation)
- [Andrej Karpathy's blog](https://github.com/flaviuvadan/tf-shakespeare.githttp://karpathy.github.io/2015/05/21/rnn-effectiveness/)