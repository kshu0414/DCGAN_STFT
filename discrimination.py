# Implementation of Minibatch Discrimination introduced in paper Improved Techniques for Training GANs
# Code copied from https://github.com/keras-team/keras/pull/3677/files with modification
# initializations -> initializers

from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
import tensorflow as tf
from keras.optimizers import Adam
import numpy as np


class MinibatchDiscrimination(Layer):
    """Concatenates to each sample information about how different the input
    features for that sample are from features of other samples in the same
    minibatch, as described in Salimans et. al. (2016). Useful for preventing
    GANs from collapsing to a single output. When using this layer, generated
    samples and reference samples should be in separate batches.
    # Example
    ```python
        # apply a convolution 1d of length 3 to a sequence with 10 timesteps,
        # with 64 output filters
        model = Sequential()
        model.add(Convolution1D(64, 3, border_mode='same', input_shape=(10, 32)))
        # now model.output_shape == (None, 10, 64)
        # flatten the output so it can be fed into a minibatch discrimination layer
        model.add(Flatten())
        # now model.output_shape == (None, 640)
        # add the minibatch discrimination layer
        model.add(MinibatchDiscrimination(5, 3))
        # now model.output_shape = (None, 645)
    ```
    # Arguments
        nb_kernels: Number of discrimination kernels to use
            (dimensionality concatenated to output).
        kernel_dim: The dimensionality of the space where closeness of samples
            is calculated.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        weights: list of numpy arrays to set as initial weights.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        input_dim: Number of channels/dimensions in the input.
            Either this argument or the keyword argument `input_shape`must be
            provided when using this layer as the first layer in a model.
    # Input shape
        2D tensor with shape: `(samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(samples, input_dim + nb_kernels)`.
    # References
        - [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
    """

    def __init__(self, nb_kernels, kernel_dim, input_dim, **kwargs):
        super(MinibatchDiscrimination, self).__init__(**kwargs)
        self.nb_kernels = nb_kernels
        self.kernel_dim = kernel_dim
        self.W = self.add_weight(shape=(self.nb_kernels, input_dim, self.kernel_dim),
                             initializer='glorot_uniform',
                             trainable=True,
                             name='minibatch_discrimination')

    def call(self, x, mask=None):
        activation = K.reshape(K.dot(x, self.W), (-1, self.nb_kernels, self.kernel_dim))
        diffs = K.expand_dims(activation, 3) - K.expand_dims(K.permute_dimensions(activation, [1, 2, 0]), 0)
        abs_diffs = K.sum(K.abs(diffs), axis=2)
        minibatch_features = K.sum(K.exp(-abs_diffs), axis=2)
        output = K.concatenate([x, minibatch_features], axis=1)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], input_shape[1]+self.nb_kernels)

def temp_model():
    input = Input(shape=(180,))
    features = MinibatchDiscrimination(nb_kernels=3,kernel_dim=16,input_dim=180)(input)
    output = Dense(1,activation=None)(features)
    curr_model = Model(inputs=input,outputs=output)
    return(curr_model)

if __name__ == "__main__":
    
    model = temp_model()
    print(model.summary())
    model.compile(
        loss = 'mean_squared_error',
        optimizer = Adam()
    )

    x = np.random.uniform(0,1,(20,180))
    y = np.random.uniform(0,1,(20,1))
    model.train_on_batch(x,y)
