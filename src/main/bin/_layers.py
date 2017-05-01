import tensorflow as tf
from tensorflow.python.ops.nn import moments, batch_normalization


def batch_norm(inputs, epsilon=.01, convolution=False, name=None):
    """

    Args:
        inputs:
        epsilon:
        convolution:
        name:

    Returns:

    """
    with tf.variable_scope(name):
        # Determine size for `gamma` and `beta`
        size = inputs.get_shape().as_list()[3]

        # Create scale parameters
        gamma = tf.get_variable(name="scales", shape=[1, 1, 1, size], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.1))

        # Create biases (a.k.a. offsets)
        beta = tf.get_variable(name="offsets", shape=[1, 1, 1, size], dtype=tf.float32,
                               initializer=tf.constant_initializer(0.1))

        # Apply batch normalisation
        if convolution:
            mean, variance = moments(x=inputs, axes=[0, 1, 2])
        else:
            mean, variance = moments(x=inputs, axes=[0])

        output = batch_normalization(x=inputs, mean=mean, variance=variance, scale=gamma, offset=beta,
                                     variance_epsilon=epsilon)

    return output


def convolution2d(inputs,
                  n_output_filters,
                  kernel_size,
                  strides,
                  padding="SAME",
                  bias=True,
                  activation=tf.nn.relu,
                  name=None):
    """

    Args:
        inputs:
        n_output_filters:
        kernel_size:
        strides:
        padding:
        bias:
        activation:
        name:

    Returns:

    """
    with tf.variable_scope(name):
        # Determine kernel shape
        n_input_channels = inputs.get_shape().as_list()[3]
        kernel_shape = kernel_size + [n_input_channels, n_output_filters]

        # Determine full strides
        strides_ = [1, 1] + strides

        # Create filter parameters
        W = tf.get_variable(name="filters", shape=kernel_shape, dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.01))
        # Apply convolution filters (a.k.a. kernel)
        outputs = tf.nn.conv2d(input=inputs, filter=W, strides=strides_, padding=padding,
                               use_cudnn_on_gpu=True)

        # if bias == True, create biases and add to output
        if bias:
            b = tf.get_variable(name="biases", shape=[1, 1, 1, n_output_filters], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.1))
            outputs += b

        # Apply non-linearity
        outputs = activation(outputs)

    return outputs


def convolution2d_transpose(inputs,
                            n_output_filters,
                            kernel_size,
                            strides,
                            padding="SAME",
                            bias=True,
                            activation=tf.nn.relu,
                            name=None):
    """

    Args:
        inputs:
        n_output_filters:
        kernel_size:
        strides:
        padding:
        bias:
        activation:
        name:

    Returns:

    """
    with tf.variable_scope(name):
        # Determine kernel shape
        n_input_channels = inputs.get_shape().as_list()[3]
        kernel_shape = kernel_size + [n_input_channels, n_output_filters]

        # Create filter parameters
        W = tf.get_variable(name="filters", shape=kernel_shape, dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.01))

        ###

        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]

        h_axis, w_axis, c_axis = 1, 2, 3

        height, width = inputs_shape[h_axis], inputs_shape[w_axis]
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = strides

        def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
            if isinstance(dim_size, tf.Tensor):
                dim_size = tf.multiply(dim_size, stride_size)
            elif dim_size is not None:
                dim_size *= stride_size

            if padding == "VALID" and dim_size is not None:
                dim_size += max(kernel_size - stride_size, 0)
            return dim_size

        # Infer the dynamic output shape:
        out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
        out_width = get_deconv_dim(width, stride_w, kernel_w, padding)

        output_shape = (batch_size, out_height, out_width, n_output_filters)

        # Determine full strides
        strides = [1, stride_h, stride_w, 1]

        output_shape_tensor = tf.stack(output_shape)

        # Apply convolution filters (a.k.a. kernel)
        outputs = tf.nn.conv2d_transpose(
            value=inputs,
            filter=W,
            output_shape=output_shape_tensor,
            strides=strides,
            padding=padding)

        # Infer the static output shape:
        out_shape = inputs.get_shape().as_list()
        out_shape[c_axis] = n_output_filters
        out_shape[h_axis] = get_deconv_dim(
            out_shape[h_axis], stride_h, kernel_h, padding)
        out_shape[w_axis] = get_deconv_dim(
            out_shape[w_axis], stride_w, kernel_w, padding)
        outputs.set_shape(out_shape)

        # if bias == True, create biases and add to output
        if bias:
            b = tf.get_variable(name="biases", shape=[1, 1, 1, n_output_filters], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.1))
            outputs += b

        # Apply non-linearity
        outputs = activation(outputs)

    return outputs


def dense(inputs,
          n_units,
          bias=True,
          activation=tf.nn.relu,
          name=None):
    """

    Args:
        inputs:
        n_units:
        bias:
        activation:
        name:

    Returns:

    """
    with tf.variable_scope(name):
        # Determine number of input units (inputs size)
        n_in = tf.size(inputs)

        # filter parameters
        W = tf.get_variable(name="weights", shape=[n_in, n_units], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.01))
        # apply filters
        outputs = tf.matmul(inputs, W)

        # if bias == True, create biases and add to output
        if bias:
            b = tf.get_variable(name="biases", shape=[1, n_units], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.1))
            outputs += b

        # Apply non-linearity
        outputs = activation(outputs)

    return outputs


def conv2d_with_batch_norm(inputs,
                           n_output_filters,
                           filter_size,
                           strides,
                           padding="SAME",
                           activation=tf.nn.relu,
                           name=None):
    """

    Args:
        inputs:
        n_output_filters:
        filter_size:
        strides:
        padding:
        activation:
        name:

    Returns:

    """
    with tf.variable_scope(name):
        # Apply convolution without biases
        a = convolution2d(inputs=inputs, n_output_filters=n_output_filters, kernel_size=filter_size, strides=strides,
                          padding=padding, bias=False)

        # Apply batch normalisation
        h = batch_norm(inputs=a, convolution=True)

        # Apply non-linearity
        outputs = activation(h)

    return outputs


def deconv2d_with_batch_norm(inputs,
                             n_output_filters,
                             filter_size,
                             strides,
                             padding="SAME",
                             activation=tf.nn.relu,
                             name=None):
    """

    Args:
        inputs:
        n_output_filters:
        filter_size:
        strides:
        padding:
        activation:
        name:

    Returns:

    """
    with tf.variable_scope(name):
        # Apply convolution without biases
        a = convolution2d_transpose(inputs=inputs, n_output_filters=n_output_filters, kernel_size=filter_size,
                                    strides=strides, padding=padding, bias=False)

        # Apply batch normalisation
        h = batch_norm(inputs=a, convolution=True)

        # Apply non-linearity
        outputs = activation(h)

    return outputs


def dense_with_batchnorm(inputs,
                         n_units,
                         activation=tf.nn.relu,
                         name=None):
    with tf.variable_scope(name):
        # Apply dense layer
        a = dense(inputs=inputs, n_units=n_units, bias=False)

        # Apply batch normalisation
        h = batch_norm(inputs=a, convolution=True)

        # Apply non-linearity
        outputs = activation(h)

    return outputs
