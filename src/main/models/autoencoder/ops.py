import tensorflow as tf

from src.main.utils.nn.layers import convolution2d, convolution2d_transpose as deconv2d, linear


def encode(value, filters_shapes, filters_strides, is_training, activation=tf.nn.relu, name="autoencoder-encode"):
    """

    Args:
        value:            Input `Tensor` of shape `[batch_size, in_height, in_width, n_in_channels]`.
                          A batch of images.
        filters_shapes:   A `list` of `int`. 1-D of length 3.
                          Like : `[kernel_height, kernel_width, n_filter_out_channels]`.
                          The shapes of convolution filters.
        filters_strides:  A `list` of `int`. 1-D of length 2.
                          Like : `[stride_h, stride_w]`.
                          The shapes of convolution filters.
        is_training:
        activation:       Activation function for the layers (default: relu).
        name:             A scope name for this set of layers (optional, default: "autoencoder-encode").

    Returns:              The input tensor encoded and the shapes of the inputs at each layer.

    """
    with tf.variable_scope(name):
        #
        v = value

        #
        n_layers = len(filters_shapes)
        for i in range(n_layers):  # accessing the filters forward
            # Determine kernel size
            height, width, n_channels = filters_shapes[i][0], filters_shapes[i][1], filters_shapes[i][2]

            # Determine convolution strides
            strides = filters_strides[i]

            # Apply convolution + batch normalisation + activation function
            v = convolution2d(inputs=v, kernel_shape=[height, width], n_output_filters=n_channels,
                              strides=strides, activation_fn=activation, is_training=is_training,
                              name="conv-layer-{}".format(i + 1))

    return v


def decode(value, filters_shapes, filters_strides, is_training, activation=tf.nn.relu, name="autoencoder-decode"):
    """

    Args:
        value:            Input `Tensor` [of encoded images].
        filters_shapes:   A `list` of `int`. 1-D of length 4.
                          Like : `[filter_height, filter_width, n_filter_in_channels, n_filter_out_channels]`.
                          The shapes of convolution filters.
        filters_strides:  A `list` of `int`. 1-D of length 4.
                          Like : `[filter_height, filter_width, n_filter_in_channels, n_filter_out_channels]`.
                          The shapes of convolution filters.
        is_training:
        activation:       Activation function for the layers (default: relu).
        name:             A scope name for this set of layers (optional, default: "autoencoder-decode").

    Returns:              The decoded input tensor.

    """
    with tf.variable_scope(name):
        v = value

        #
        n_filters = len(filters_shapes)
        for i in range(n_filters - 1, -1, -1):  # accessing the filters backward

            with tf.variable_scope("layer-{}".format(n_filters - i - 1)):
                # Determine kernel size
                height, width, n_channels = filters_shapes[i][0], filters_shapes[i][1], filters_shapes[i][2]

                # Determine convolution strides
                strides = filters_strides[i]

                # Apply convolution + batch normalisation + activation function
                v = deconv2d(inputs=v, kernel_size=[height, width], n_output_filters=n_channels,
                             strides=strides, activation=activation, is_training=is_training,
                             name="deconv-layer-{}".format(i + 1))

    return v


def dense(value, sizes, is_training, activation=tf.nn.relu, name="dense"):
    """

    Args:
        value:            A `Tensor`.
        sizes:            The sizes (number of units) of all densely connected layers in this scope.
        is_training:
        activation:       Activation function for the layers (default: relu).
        name:             A scope name for this set of layers (optional, default: "dense").

    Returns:              A `Tensor`.

    """
    with tf.variable_scope(name):
        v = value

        for i in range(len(sizes)):
            with tf.variable_scope("layer-{}".format(i)):
                v = linear(inputs=value, n_output_units=sizes[i], is_training=is_training, activation_fn=activation)

    return v
