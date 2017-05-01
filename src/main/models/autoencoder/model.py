import numpy as np
import tensorflow as tf

from src.main.utils.nn.layers import convolution2d, convolution2d_transpose as deconv2d, linear
from src.main.utils.nn.layers import residual_conv2d, residual_deconv2d as residual_deconv2d
from src.main.utils.nn.losses import l2_loss, adversarial_loss


def model(inputs, is_training, name="model"):
    with tf.variable_scope(name):
        # residual block 1
        # number of residual layers :     6,
        # kernel size               : 32x32,
        # number of feature maps    :   128,
        # activation function       :   elu
        h1 = residual_conv2d(inputs=inputs,
                             n_layers=1,
                             n_output_filters=3,
                             kernel_shape=[5, 5],
                             strides=[1, 1],
                             is_training=is_training,
                             activation_fn=tf.nn.elu,
                             name="res-block-1")
        outputs = h1
        # end of residual block

        # down sampling convolution layer
        outputs = convolution2d(inputs=outputs,
                                n_output_filters=128,
                                kernel_shape=[5, 5],
                                strides=[2, 2],
                                is_training=is_training,
                                activation_fn=tf.nn.elu,
                                name="down-sampling-1")
        #

        # residual block 2
        # number of residual layers :     6,
        # kernel size               : 16x16,
        # number of feature maps    :   128,
        # activation function       :   elu
        h2 = residual_conv2d(inputs=outputs,
                             n_layers=1,
                             n_output_filters=128,
                             kernel_shape=[5, 5],
                             strides=[1, 1],
                             is_training=is_training,
                             activation_fn=tf.nn.elu,
                             name="res-block-2")
        outputs = h2
        # end of residual block

        # down sampling convolution layer
        outputs = convolution2d(inputs=outputs,
                                n_output_filters=256,
                                kernel_shape=[5, 5],
                                strides=[2, 2],
                                is_training=is_training,
                                activation_fn=tf.nn.elu,
                                name="down-sampling-2")
        #

        # residual block 3
        # number of residual layers :     6,
        # kernel size               :   8x8,
        # number of feature maps    :   128,
        # activation function       :   elu
        h3 = residual_conv2d(inputs=outputs,
                             n_layers=1,
                             n_output_filters=256,
                             kernel_shape=[5, 5],
                             strides=[1, 1],
                             is_training=is_training,
                             activation_fn=tf.nn.elu,
                             name="res-block-3")
        outputs = h3

        # down sampling convolution layer
        outputs = convolution2d(inputs=outputs,
                                n_output_filters=512,
                                kernel_shape=[5, 5],
                                strides=[2, 2],
                                is_training=is_training,
                                activation_fn=tf.nn.elu,
                                name="down-sampling-3")
        #

        # residual block 4
        # number of residual layers :     6,
        # kernel size               :   4x4,
        # number of feature maps    :   128,
        # activation function       :   elu
        h4 = residual_conv2d(inputs=outputs,
                             n_layers=1,
                             n_output_filters=512,
                             kernel_shape=[5, 5],
                             strides=[1, 1],
                             is_training=is_training,
                             activation_fn=tf.nn.elu,
                             name="res-block-4")
        outputs = h4
        # end of residual block

        #

        # dense block
        # now channel-wise
        _, dim1, dim2, dim3 = outputs.get_shape().as_list()
        ndim = dim1 * dim2
        o = []
        for i in range(dim3):
            x = tf.reshape(outputs[:, :, :,  i], [-1, ndim])
            x = linear(inputs=x,
                       n_output_units=ndim,
                       is_training=is_training,
                       activation_fn=tf.nn.elu,
                       name="dense-{}".format(i))
            x = tf.reshape(x, shape=[-1, dim1, dim2])
            o.append(x)
        outputs = tf.stack(o, axis=-1)

        #

        # residual block 5
        # number of residual layers :     6,
        # kernel size               :   4x4,
        # number of feature maps    :   128,
        # activation function       :   elu
        outputs = outputs + h4
        h5 = residual_deconv2d(inputs=outputs,
                               n_layers=1,
                               n_output_filters=512,
                               kernel_shape=[5, 5],
                               strides=[1, 1],
                               is_training=is_training,
                               activation_fn=tf.nn.elu,
                               name="res-block-5")
        outputs = h5
        # end of residual block

        # up sampling convolution layer
        outputs = deconv2d(inputs=outputs,
                           n_output_filters=256,
                           kernel_shape=[5, 5],
                           strides=[2, 2],
                           is_training=is_training,
                           activation_fn=tf.nn.elu,
                           name="up-sampling-1")
        #

        #  residual block 6
        # number of residual layers :     6,
        # kernel size               :   8x8,
        # number of feature maps    :   128,
        # activation function       :   elu
        outputs = outputs + h3
        h6 = residual_deconv2d(inputs=outputs,
                               n_layers=1,
                               n_output_filters=256,
                               kernel_shape=[5, 5],
                               strides=[1, 1],
                               is_training=is_training,
                               activation_fn=tf.nn.elu,
                               name="res-block-6")
        outputs = h6
        # end of residual block

        # up sampling convolution layer
        outputs = deconv2d(inputs=outputs,
                           n_output_filters=128,
                           kernel_shape=[5, 5],
                           strides=[2, 2],
                           is_training=is_training,
                           activation_fn=tf.nn.elu,
                           name="up-sampling-2")
        #

        # residual block 7
        # number of residual layers :     6,
        # kernel size               : 16x16,
        # number of feature maps    :   128,
        # activation function       :   elu
        outputs = outputs + h2
        h7 = residual_deconv2d(inputs=outputs,
                               n_layers=1,
                               n_output_filters=128,
                               kernel_shape=[5, 5],
                               strides=[1, 1],
                               is_training=is_training,
                               activation_fn=tf.nn.elu,
                               name="res-block-7")
        outputs = h7
        # end of residual block

        # convolution layer
        outputs = deconv2d(inputs=outputs,
                           n_output_filters=3,
                           kernel_shape=[5, 5],
                           strides=[1, 1],
                           is_training=is_training,
                           activation_fn=tf.nn.elu,
                           name="deconv")

        # residual block 8
        # number of residual layers :     6,
        # kernel size               : 16x16,
        # number of feature maps    :   128,
        # activation function       :   elu
        outputs = outputs
        h8 = residual_deconv2d(inputs=outputs,
                               n_layers=1,
                               n_output_filters=3,
                               kernel_shape=[5, 5],
                               strides=[1, 1],
                               is_training=is_training,
                               activation_fn=tf.nn.elu,
                               name="res-block-8")
        outputs = h8
        # end of residual block

    return outputs


class Autoencoder(object):
    def __init__(self, images_shape, name="auto-encoder"):
        # model's scope name
        self._name = name

        #
        self._images_shape = images_shape

        #
        self._session = tf.Session()

    def _build(self):

        #
        self._is_training = tf.placeholder(dtype=tf.bool, shape=[])

        # inputs
        self._x = tf.placeholder(dtype=tf.float32, shape=[None] + self._images_shape)

        # targets
        self._t = tf.placeholder(dtype=tf.float32, shape=[None] + self._images_shape)

        # output
        y = model(inputs=self._x, is_training=self._is_training, name=self._name)

        # loss
        o = tf.pad(y, paddings=[[0, 0], [16, 16], [16, 16], [0, 0]], mode="CONSTANT")  # padding of image_dim / 4
        self._loss = l2_loss(targets=self._t, y=o)

    def train(self, datasets, batch_size=20, max_epoch=50, optimizer=tf.train.AdamOptimizer, clip_coef=1):

        # build model
        self._build()

        # gradient optimization
        grads_and_vars = optimizer().compute_gradients(self._loss)
        grads_and_vars = [(tf.clip_by_norm(t=grad, clip_norm=clip_coef), var) for grad, var in grads_and_vars]
        train_step = optimizer().apply_gradients(grads_and_vars=grads_and_vars)

        #
        self._session.run(tf.global_variables_initializer())

        # set summaries
        self._summaries(location=".", batch_size=batch_size)

        #
        n_train_batches = np.ceil(datasets.train.images.shape[0] / batch_size)
        n_valid_batches = np.ceil(datasets.validation.images.shape[0] / (2 * batch_size))

        #
        validation_loss = np.infty

        for epoch in np.arange(1, max_epoch + 1):

            # on training samples
            for batch_index in np.arange(n_train_batches):
                train_batch_xs, train_batch_ts = datasets.train.next_batch(batch_size)
                train_batch_xs = train_batch_xs.reshape([-1] + self._images_shape)
                train_loss, op, _ = self._session.run(fetches=[self._loss, self._op, train_step],
                                                      feed_dict={self._x: train_batch_xs,
                                                                 self._t: train_batch_ts,
                                                                 self._is_training: True})
                self._train_writer.add_summary(op)

            # on validation samples
            validation_losses = []
            for batch_index in np.arange(n_valid_batches):
                valid_batch_xs, valid_batch_ts = datasets.validation.next_batch(2 * batch_size)
                valid_batch_xs = valid_batch_xs.reshape([-1] + self._images_shape)
                valid_loss, op = self._session.run(fetches=[self._loss, self._op],
                                                   feed_dict={self._x: valid_batch_xs,
                                                              self._t: valid_batch_ts,
                                                              self._is_training: False})
                self._valid_writer.add_summary(op)

                validation_losses.append(valid_loss)
            if validation_loss < np.mean(validation_losses):
                self.save()
            validation_loss = np.mean(validation_losses)

        # close the writers
        self._train_writer.close(), self._valid_writer.close()

    def apply(self, inputs):
        inputs = inputs.reshape([-1] + self._images_shape)
        output = self._session.run(fetches=self._y, feed_dict={self._x: inputs})
        return output

    def save(self, location=None):
        path = "./mvs/" if location is None else location
        saver = tf.train.Saver()
        saver.save(self._session, path)

    def load(self, location=None):
        path = "." if location is None else location
        self._session = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self._session, path)

    def _summaries(self, location, batch_size):

        #
        tf.summary.image(name="target-image", tensor=self._t, max_outputs=batch_size)
        tf.summary.image(name="input-image", tensor=self._x, max_outputs=batch_size)

        #
        tf.summary.scalar(name="l2-reconstruction-loss", tensor=self._loss)

        self._op = tf.summary.merge_all()
        self._train_writer = tf.summary.FileWriter("{}/train/".format(location), self._session.graph)
        self._valid_writer = tf.summary.FileWriter("{}/validation/".format(location), self._session.graph)
