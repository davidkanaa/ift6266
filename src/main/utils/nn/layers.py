import tensorflow as tf
from tensorflow.python.layers.layers import batch_normalization, dropout
from tensorflow.python.layers.layers import conv2d, conv2d_transpose, dense


# Note: kernel_size must be a tuple of 1 integer, same for stride ! weird
def convolution2d(inputs,
                  n_output_filters,
                  kernel_shape,
                  strides,
                  is_training,
                  padding="SAME",
                  dilation=(1, 1),
                  batch_norm=True,
                  activation_fn=tf.nn.relu,
                  filters_initializer=tf.random_normal_initializer(stddev=0.01),
                  name="conv2d"):
    with tf.variable_scope(name):
        # Apply convolution
        h = conv2d(inputs=inputs, filters=n_output_filters, kernel_size=kernel_shape, strides=strides,
                   padding=padding, dilation_rate=dilation, kernel_initializer=filters_initializer)

        # if batch_norm, apply batch normalisation
        if batch_norm:
            h = batch_normalization(inputs=h,
                                    training=is_training, trainable=True,
                                    scale=True, center=True)

        # Apply non-linearity
        outputs = activation_fn(h) if activation_fn is not None else h

    return outputs


def convolution2d_transpose(inputs,
                            n_output_filters,
                            kernel_shape,
                            strides,
                            is_training,
                            padding="SAME",
                            batch_norm=True,
                            activation_fn=tf.nn.relu,
                            filters_initializer=tf.random_normal_initializer(stddev=0.01),
                            name="conv2d-transposed"):
    with tf.variable_scope(name):
        # Apply convolution
        h = conv2d_transpose(inputs=inputs, filters=n_output_filters, kernel_size=kernel_shape, strides=strides,
                             padding=padding, kernel_initializer=filters_initializer)

        # if batch_norm, apply batch normalisation
        if batch_norm:
            h = batch_normalization(inputs=h,
                                    training=is_training, trainable=True,
                                    scale=True, center=True)

        # Apply non-linearity
        outputs = activation_fn(h) if activation_fn is not None else h

    return outputs


def linear(inputs,
           n_output_units,
           is_training,
           batch_norm=True,
           activation_fn=tf.nn.relu,
           weights_initializer=tf.random_normal_initializer(stddev=0.01),
           name="linear"):
    with tf.variable_scope(name):
        # Apply convolution
        h = dense(inputs=inputs, units=n_output_units,
                  kernel_initializer=weights_initializer, use_bias=False)

        # if batch_norm, apply batch normalisation
        if batch_norm:
            h = batch_normalization(inputs=h,
                                    training=is_training, trainable=True,
                                    scale=True, center=True)

        # Apply non-linearity
        outputs = activation_fn(h) if activation_fn is not None else h

    return outputs


def residual_conv2d(inputs,
                    n_layers,
                    n_output_filters,
                    kernel_shape,
                    strides,
                    is_training,
                    activation_fn=tf.nn.relu,
                    name="residual-conv2d"):
    with tf.variable_scope(name):
        x = inputs
        h = inputs
        for i in range(n_layers - 1):
            h = convolution2d(inputs=h,
                              n_output_filters=n_output_filters,
                              kernel_shape=kernel_shape,
                              strides=strides,
                              is_training=is_training,
                              activation_fn=activation_fn,
                              name="res-layer{}".format(i + 1))
        h = convolution2d(inputs=h,
                          n_output_filters=n_output_filters,
                          kernel_shape=kernel_shape,
                          strides=strides,
                          is_training=is_training,
                          activation_fn=lambda v: v,
                          name="res-layer{}".format(n_layers))
        outputs = activation_fn(h + x) if activation_fn is not None else h + x
    return outputs


def residual_deconv2d(inputs,
                      n_layers,
                      n_output_filters,
                      kernel_shape,
                      strides,
                      is_training,
                      activation_fn=tf.nn.relu,
                      name="residual-conv2d-transposed"):
    with tf.variable_scope(name):
        x = inputs
        h = inputs
        for i in range(n_layers - 1):
            h = convolution2d_transpose(inputs=h,
                                        n_output_filters=n_output_filters,
                                        kernel_shape=kernel_shape,
                                        strides=strides,
                                        is_training=is_training,
                                        activation_fn=activation_fn,
                                        name="res-layer{}".format(i + 1))
        h = convolution2d_transpose(inputs=h,
                                    n_output_filters=n_output_filters,
                                    kernel_shape=kernel_shape,
                                    strides=strides,
                                    is_training=is_training,
                                    activation_fn=lambda v: v,
                                    name="res-layer{}".format(n_layers))
        outputs = activation_fn(h + x) if activation_fn is not None else h + x
    return outputs
