"""
This is a basic implementation of DRAW : Recurrent Network For Image Generation (http://arxiv.org/pdf/1502.04623v2.pdf).

Notes:
    - There are various candidates for the read/write attention mechanism (attention networks): soft, hard, mixtures, gaussian (default).
    - Adding depth, residual blocks ... may help.
    - Adding an Adversarial component (discriminator) ... may help.
"""

import tensorflow as tf
from tensorflow.contrib import rnn

from src.main.utils.nn.layers import linear


def get_filterbank_matrices(image_shape, gx, gy, patch_size, delta, sigma2, epsilon=1e-4, name="get-attention-filters"):
    with tf.variable_scope(name):
        # location indices in the patch
        patch_indices = tf.cast(tf.range(patch_size), tf.float32)

        # compute the mean location of the filters
        mu_x = gx + (patch_indices - .5 * (patch_size + 1)) * delta
        mu_y = gy + (patch_indices - .5 * (patch_size + 1)) * delta

        # location indices in the input image
        a = tf.cast(tf.range(image_shape[0]), tf.float32)
        b = tf.cast(tf.range(image_shape[1]), tf.float32)

        # reshape everything to match [patch_size, image_dim]
        # for later matrix multiplication with items of shape [batch_size, number_of_channels, patch_size, image_dim]
        mu_x = tf.reshape(mu_x, [-1, 1])
        mu_y = tf.reshape(mu_y, [-1, 1])
        a = tf.reshape(a, [1, -1])
        b = tf.reshape(b, [1, -1])

        # compute the filters
        Fx = tf.exp(-tf.square(a - mu_x) / (2 * sigma2))
        Fy = tf.exp(-tf.square(b - mu_y) / (2 * sigma2))

        # normalise the filters over the image dims
        Fx = Fx / tf.maximum(tf.reduce_sum(Fx, axis=2, keep_dims=True), epsilon)
        Fy = Fy / tf.maximum(tf.reduce_sum(Fy, axis=2, keep_dims=True), epsilon)

    return Fx, Fy


def get_grid(image_shape, patch_size, gx, gy, delta, name="get-grid"):
    with tf.variable_scope(name):
        A, B = image_shape
        gx = .5 * (A + 1) * (gx + 1)
        gy = .5 * (B + 1) * (gy + 1)
        delta = (max(A, B) - 1) * delta / (patch_size - 1)
    return gx, gy, delta


def get_attention_params(image_shape, patch_size, state, is_training, name="get-attention-params"):
    with tf.variable_scope(name):
        # compute core attention params
        params = linear(inputs=state, n_output_units=5, is_training=is_training, activation_fn=None)
        gx, gy, log_sigma2, log_delta, log_gamma = tf.split(params, 5, 1)
        delta = tf.exp(log_delta)
        sigma2 = tf.exp(log_sigma2)
        gamma = tf.exp(log_gamma)

        # compute grid then attention filters
        gx, gy, delta = get_grid(image_shape=image_shape, patch_size=patch_size, gx=gx, gy=gy, delta=delta)
        Fx, Fy = get_filterbank_matrices(image_shape=image_shape, patch_size=patch_size, gx=gx, gy=gy, delta=delta,
                                         sigma2=sigma2)
    return Fx, Fy, gamma


def get_glimpse(inputs, Fx, Fy, gamma, name="get-glimpse"):
    with tf.variable_scope(name):
        Fy = tf.transpose(Fy, perm=[0, 1, 3, 2])
        glimpse = apply_attention(inputs, Fx, Fy, gamma)
    return glimpse


def project_glimpse(inputs, Fx, Fy, gamma, name="project-glimpse"):
    with tf.variable_scope(name):
        Fx = tf.transpose(Fx, perm=[0, 1, 3, 2])
        gamma = tf.inv(gamma)
        u = apply_attention(inputs, Fx, Fy, gamma)
    return u


def apply_attention(inputs, Fx, Fy, gamma, name="apply-attention"):
    with tf.variable_scope(name):
        u = tf.matmul(Fx, tf.matmul(inputs, Fy))
        u = tf.scalar_mul(scalar=gamma, x=u)
    return u


def read(image, error, state, image_shape, patch_size, is_training, name="read"):
    with tf.variable_scope(name):
        # get attention window params
        Fx, Fy, gamma = get_attention_params(image_shape=image_shape, patch_size=patch_size, state=state,
                                             is_training=is_training)

        # apply attention filters
        x = get_glimpse(inputs=image, Fx=Fx, Fy=Fy, gamma=gamma)
        x_ = get_glimpse(inputs=error, Fx=Fx, Fy=Fy, gamma=gamma)

    return tf.concat([x, x_], axis=1)


def write(state, canvas_shape, patch_size, is_training, name="write"):
    with tf.variable_scope(name):
        # get writer glimpse
        # it is supposed to have the shape [patch_size, patch_size]
        # we'll see if it does and if it doesn't we may apply another linear transformation to meet the requirements
        w = linear(state, n_output_units=patch_size, is_training=is_training, activation_fn=None)

        # get attention window params
        Fx, Fy, gamma = get_attention_params(image_shape=canvas_shape, patch_size=patch_size, state=state,
                                             is_training=is_training)

        # project the glimpse to canvas
        w = project_glimpse(inputs=w, Fx=Fx, Fy=Fy, gamma=gamma)

    return w


encoder_size = 256
lstm_encoder = rnn.LSTMCell(num_units=encoder_size)


def encode(inputs, state, name="encode"):
    with tf.variable_scope(name):
        outputs = lstm_encoder(inputs, state)
    return outputs


decoder_size = 256
lstm_decocer = rnn.LSTMCell(num_units=decoder_size)


def decode(inputs, state, name="decode"):
    with tf.variable_scope(name):
        outputs = lstm_decocer(inputs, state)
    return outputs


def sample(state, size, is_training, name="sample"):
    with tf.variable_scope(name):
        mu = linear(inputs=state, n_output_units=size, is_training=is_training, activation_fn=None)
        log_sigma = linear(inputs=state, n_output_units=size, is_training=is_training, activation_fn=None)
        sigma = tf.exp(log_sigma)
        b = state.get_shape().as_list()[0]
        e = tf.random_normal([b, size])
    return mu + sigma * e, mu, log_sigma, sigma


class DRAW(object):
    def __init__(self, image_shape, reader_patch, writer_patch):
        pass

    def build(self):
        pass
