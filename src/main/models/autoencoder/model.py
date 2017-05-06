import numpy as np
import tensorflow as tf

from src.main.utils.nn.layers import convolution2d as conv2d, convolution2d_transpose as deconv2d, linear
from src.main.utils.nn.layers import residual_conv2d, residual_deconv2d as residual_deconv2d
from src.main.utils.nn.losses import l2_loss, cross_entropy_loss


# def model(inputs, is_training, name="model"):
#     with tf.variable_scope(name):
#         # residual block 1
#         # number of residual layers :     6,
#         # kernel size               : 32x32,
#         # number of feature maps    :   128,
#         # activation function       :   elu
#         h1 = residual_conv2d(inputs=inputs,
#                              n_layers=1,
#                              n_output_filters=3,
#                              kernel_shape=[5, 5],
#                              strides=[1, 1],
#                              is_training=is_training,
#                              activation_fn=tf.nn.elu,
#                              name="res-block-1")
#         outputs = h1
#         # end of residual block
#
#         # down sampling convolution layer
#         outputs = convolution2d(inputs=outputs,
#                                 n_output_filters=128,
#                                 kernel_shape=[5, 5],
#                                 strides=[2, 2],
#                                 is_training=is_training,
#                                 activation_fn=tf.nn.elu,
#                                 name="down-sampling-1")
#         #
#
#         # residual block 2
#         # number of residual layers :     6,
#         # kernel size               : 16x16,
#         # number of feature maps    :   128,
#         # activation function       :   elu
#         h2 = residual_conv2d(inputs=outputs,
#                              n_layers=1,
#                              n_output_filters=128,
#                              kernel_shape=[5, 5],
#                              strides=[1, 1],
#                              is_training=is_training,
#                              activation_fn=tf.nn.elu,
#                              name="res-block-2")
#         outputs = h2
#         # end of residual block
#
#         # down sampling convolution layer
#         outputs = convolution2d(inputs=outputs,
#                                 n_output_filters=256,
#                                 kernel_shape=[5, 5],
#                                 strides=[2, 2],
#                                 is_training=is_training,
#                                 activation_fn=tf.nn.elu,
#                                 name="down-sampling-2")
#         #
#
#         # residual block 3
#         # number of residual layers :     6,
#         # kernel size               :   8x8,
#         # number of feature maps    :   128,
#         # activation function       :   elu
#         h3 = residual_conv2d(inputs=outputs,
#                              n_layers=1,
#                              n_output_filters=256,
#                              kernel_shape=[5, 5],
#                              strides=[1, 1],
#                              is_training=is_training,
#                              activation_fn=tf.nn.elu,
#                              name="res-block-3")
#         outputs = h3
#
#         # down sampling convolution layer
#         outputs = convolution2d(inputs=outputs,
#                                 n_output_filters=512,
#                                 kernel_shape=[5, 5],
#                                 strides=[2, 2],
#                                 is_training=is_training,
#                                 activation_fn=tf.nn.elu,
#                                 name="down-sampling-3")
#         #
#
#         # residual block 4
#         # number of residual layers :     6,
#         # kernel size               :   4x4,
#         # number of feature maps    :   128,
#         # activation function       :   elu
#         h4 = residual_conv2d(inputs=outputs,
#                              n_layers=1,
#                              n_output_filters=512,
#                              kernel_shape=[5, 5],
#                              strides=[1, 1],
#                              is_training=is_training,
#                              activation_fn=tf.nn.elu,
#                              name="res-block-4")
#         outputs = h4
#         # end of residual block
#
#         #
#
#         # dense block
#         # now channel-wise
#         _, dim1, dim2, dim3 = outputs.get_shape().as_list()
#         ndim = dim1 * dim2
#         o = []
#         for i in range(dim3):
#             x = tf.reshape(outputs[:, :, :,  i], [-1, ndim])
#             x = linear(inputs=x,
#                        n_output_units=ndim,
#                        is_training=is_training,
#                        activation_fn=tf.nn.elu,
#                        name="dense-{}".format(i))
#             x = tf.reshape(x, shape=[-1, dim1, dim2])
#             o.append(x)
#         outputs = tf.stack(o, axis=-1)
#
#         #
#
#         # residual block 5
#         # number of residual layers :     6,
#         # kernel size               :   4x4,
#         # number of feature maps    :   128,
#         # activation function       :   elu
#         outputs = outputs + h4
#         h5 = residual_deconv2d(inputs=outputs,
#                                n_layers=1,
#                                n_output_filters=512,
#                                kernel_shape=[5, 5],
#                                strides=[1, 1],
#                                is_training=is_training,
#                                activation_fn=tf.nn.elu,
#                                name="res-block-5")
#         outputs = h5
#         # end of residual block
#
#         # up sampling convolution layer
#         outputs = deconv2d(inputs=outputs,
#                            n_output_filters=256,
#                            kernel_shape=[5, 5],
#                            strides=[2, 2],
#                            is_training=is_training,
#                            activation_fn=tf.nn.elu,
#                            name="up-sampling-1")
#         #
#
#         #  residual block 6
#         # number of residual layers :     6,
#         # kernel size               :   8x8,
#         # number of feature maps    :   128,
#         # activation function       :   elu
#         outputs = outputs + h3
#         h6 = residual_deconv2d(inputs=outputs,
#                                n_layers=1,
#                                n_output_filters=256,
#                                kernel_shape=[5, 5],
#                                strides=[1, 1],
#                                is_training=is_training,
#                                activation_fn=tf.nn.elu,
#                                name="res-block-6")
#         outputs = h6
#         # end of residual block
#
#         # up sampling convolution layer
#         outputs = deconv2d(inputs=outputs,
#                            n_output_filters=128,
#                            kernel_shape=[5, 5],
#                            strides=[2, 2],
#                            is_training=is_training,
#                            activation_fn=tf.nn.elu,
#                            name="up-sampling-2")
#         #
#
#         # residual block 7
#         # number of residual layers :     6,
#         # kernel size               : 16x16,
#         # number of feature maps    :   128,
#         # activation function       :   elu
#         outputs = outputs + h2
#         h7 = residual_deconv2d(inputs=outputs,
#                                n_layers=1,
#                                n_output_filters=128,
#                                kernel_shape=[5, 5],
#                                strides=[1, 1],
#                                is_training=is_training,
#                                activation_fn=tf.nn.elu,
#                                name="res-block-7")
#         outputs = h7
#         # end of residual block
#
#         # convolution layer
#         outputs = deconv2d(inputs=outputs,
#                            n_output_filters=3,
#                            kernel_shape=[5, 5],
#                            strides=[1, 1],
#                            is_training=is_training,
#                            activation_fn=tf.nn.elu,
#                            name="deconv")
#
#         # residual block 8
#         # number of residual layers :     6,
#         # kernel size               : 16x16,
#         # number of feature maps    :   128,
#         # activation function       :   elu
#         outputs = outputs
#         h8 = residual_deconv2d(inputs=outputs,
#                                n_layers=1,
#                                n_output_filters=3,
#                                kernel_shape=[5, 5],
#                                strides=[1, 1],
#                                is_training=is_training,
#                                activation_fn=tf.nn.elu,
#                                name="res-block-8")
#         outputs = h8
#         # end of residual block
#
#     return outputs


class Autoencoder(object):
    def __init__(self, images_shape, name="DCGAN"):
        self._images_shape = images_shape

        self._name = name

        #
        self._built = False
        self._status = {"discriminator": False,
                        "encoder": False,
                        "decoder": False}

        #
        self._session = tf.Session()

    def _build(self):

        #
        self._is_training = tf.placeholder(dtype=tf.bool, shape=[])

        # inputs:
        # - images (masked, here central area masked)
        self._inputs = tf.placeholder(dtype=tf.float32, shape=[None] + self._images_shape)

        # targets:
        # - images (originals, without mask over central area)
        self._targets = tf.placeholder(dtype=tf.float32, shape=[None] + self._images_shape)

        # generate fake images
        fake_images = self._forward_pass(images=self._inputs)
        self._outputs = fake_images

        # discriminate real images, fake images, wrong images (wrong image-to-embeddings relationships [shuffled])
        logits_real = self._discriminator(is_training=self._is_training,
                                          images=self._targets)
        logits_fake = self._discriminator(is_training=self._is_training,
                                          images=fake_images)

        # losses:
        # - l2 reconstruction loss
        # - adversarial losses :
        #       > generator loss
        #       > discriminator loss
        self._losses = dict()
        self._losses["l2"] = l2_loss(targets=self._targets, y=fake_images)
        self._losses["generator"] = cross_entropy_loss(logits=logits_fake,
                                                       targets=tf.ones_like(logits_fake))
        d_real = cross_entropy_loss(logits=logits_real, targets=tf.ones_like(logits_real))
        d_fake = cross_entropy_loss(logits=logits_fake, targets=tf.zeros_like(logits_fake))
        self._losses["discriminator"] = d_real + d_fake

        #
        self._vars = dict()
        vars = tf.trainable_variables()

        self._vars["discriminator"] = [var for var in vars if "discriminator" in var.name]
        self._vars["generator"] = [var for var in vars if "encoder" in var.name or "decoder" in var.name]

    def build(self):
        self._build()
        return self

    def _forward_pass(self, images):
        outputs = self._model(is_training=self._is_training, images=images)
        outputs = tf.pad(outputs, paddings=[[0, 0], [16, 16], [16, 16], [0, 0]],
                         mode="CONSTANT")  # padding of image_dim / 4
        outputs = images + outputs  # add borders to padded outputs
        return outputs

    def _model(self, is_training, images):
        with tf.variable_scope("encoder") as scope:
            if self._status.get("encoder"):
                scope.reuse_variables()

            # pass the image through a sequence of convolution layers

            # 64x64x3 --> 32x32x32
            h0 = conv2d(inputs=images,
                        is_training=is_training,
                        n_output_filters=32, kernel_shape=[5, 5], strides=[2, 2],
                        name="conv-0")
            # 32x32x32 --> 16x16x64
            h1 = conv2d(inputs=h0,
                        is_training=is_training,
                        n_output_filters=32 * 2, kernel_shape=[5, 5], strides=[2, 2],
                        name="conv-1")
            # 16x16x64 --> 8x8x128
            h2 = conv2d(inputs=h1,
                        is_training=is_training,
                        n_output_filters=32 * 4, kernel_shape=[5, 5], strides=[2, 2],
                        name="conv-2")
            # 8x8x128 --> 4x4x256
            h3 = conv2d(inputs=h2,
                        is_training=is_training,
                        n_output_filters=32 * 8, kernel_shape=[5, 5], strides=[2, 2],
                        name="conv-3")

            #
            # h3 = tf.reshape(h3, [-1, 4 * 4 * 512])

            # code = linear(inputs=h3,
            #               is_training=is_training,
            #               n_output_units=4 * 4 * 64 * 8,
            #               name="code")

            # dense block
            # now channel-wise
            _, dim1, dim2, dim3 = h3.get_shape().as_list()
            ndim = dim1 * dim2
            o = []
            for i in range(dim3):
                x = tf.reshape(h3[:, :, :, i], [-1, ndim])
                x = linear(inputs=x,
                           n_output_units=ndim,
                           is_training=is_training,
                           activation_fn=tf.nn.elu,
                           name="dense-{}".format(i))
                x = tf.reshape(x, shape=[-1, dim1, dim2])
                o.append(x)
            code = tf.stack(o, axis=-1, name="code")

            #
            # code = h3

            if not self._status.get("encoder"):
                self._status["encoder"] = True

        with tf.variable_scope("decoder") as scope:
            if self._status.get("decoder"):
                scope.reuse_variables()

            z = code

            # pass through deconvolution layers
            strides_ = [2, 2]

            # 4x4x512 --> 8x8x256
            h0_ = deconv2d(inputs=z,
                           is_training=is_training,
                           n_output_filters=32 * 4, kernel_shape=[5, 5], strides=strides_,
                           name="deconv-0") + h2
            # 8x8x256 --> 16x16x128
            h1_ = deconv2d(inputs=h0_,
                           is_training=is_training,
                           n_output_filters=32 * 2, kernel_shape=[5, 5], strides=strides_,
                           name="deconv-1") + h1
            # 16x16x128 --> 32x32x64
            h2_ = deconv2d(inputs=h1_,
                           is_training=is_training,
                           n_output_filters=32, kernel_shape=[5, 5], strides=strides_,
                           name="deconv-2") + h0

            # 32x32x64 --> 32x32x3
            outputs = deconv2d(inputs=h2_,
                               is_training=is_training,
                               n_output_filters=3, kernel_shape=[5, 5], strides=[1, 1],
                               activation_fn=tf.nn.tanh,
                               name="deconv-3")

            if not self._status.get("decoder"):
                self._status["decoder"] = True

            return outputs

    def _discriminator(self, images, is_training):
        with tf.variable_scope("discriminator") as scope:
            if self._status.get("discriminator"):
                scope.reuse_variables()

            # pass the image through a sequence of convolution layers

            # 64x64x3 --> 32x32x32
            h0_ = conv2d(inputs=images,
                         is_training=is_training,
                         n_output_filters=32, kernel_shape=[5, 5], strides=[2, 2],
                         name="conv-0")
            # 32x32x32 --> 16x16x64
            h1 = conv2d(inputs=h0_,
                        is_training=is_training,
                        n_output_filters=32 * 2, kernel_shape=[5, 5], strides=[2, 2],
                        name="conv-1")
            # 16x16x64 --> 8x8x128
            h2 = conv2d(inputs=h1,
                        is_training=is_training,
                        n_output_filters=32 * 4, kernel_shape=[5, 5], strides=[2, 2],
                        name="conv-2")
            # 8x8x128 --> 4x4x256
            h3 = conv2d(inputs=h2,
                        is_training=is_training,
                        n_output_filters=32 * 8, kernel_shape=[5, 5], strides=[2, 2],
                        name="conv-3")
            #

            # reshape for linear units
            _, dim1, dim2, dim3 = h3.get_shape().as_list()
            h3 = tf.reshape(h3, [-1, dim1 * dim2 * dim3])

            logits = linear(inputs=h3,
                            is_training=is_training,
                            n_output_units=1,
                            activation_fn=None,
                            name="logits")

            # apply sigmoid
            # they are currently not required
            # outputs = tf.nn.sigmoid(logits)

            if not self._status.get("discriminator"):
                self._status["discriminator"] = True

            return logits

    def train(self, datasets, batch_size=20, max_epoch=50, optimizer=tf.train.AdamOptimizer, clip_coef=1):

        # build model if not already built
        if not self._built:
            self._build()

        # gradient optimization
        alpha = 0.2
        gen_op = self._optimizer((1 - alpha) * self._losses.get("l2") + alpha * self._losses.get("generator"),
                                 clip_coef, optimizer)
        disc_op = self._optimizer(self._losses.get("discriminator"), clip_coef, optimizer)

        #
        self._session.run(tf.global_variables_initializer())

        # set summaries
        self._summaries(location=".", batch_size=batch_size)

        #
        n_train_batches = np.ceil(datasets.train.n_examples / batch_size)
        n_valid_batches = np.ceil(datasets.validation.n_examples / batch_size)

        for epoch in np.arange(1, max_epoch + 1):

            # on training samples
            for batch_index in np.arange(n_train_batches):
                train_batch_xs, train_batch_ts, _ = datasets.train.next_batch(batch_size)
                train_batch_xs = train_batch_xs.reshape([-1] + self._images_shape)

                # update discriminator
                op, d_loss, _ = self._session.run(fetches=[self._op,
                                                           self._losses.get("discriminator"),
                                                           disc_op],
                                                  feed_dict={self._inputs: train_batch_xs,
                                                             self._targets: train_batch_ts,
                                                             self._is_training: True})

                # update generator (twice to make sure the discriminator's loss does not reach 0
                for k in range(2):
                    op, g_loss, _ = self._session.run(fetches=[self._op,
                                                               self._losses.get("generator"),
                                                               gen_op],
                                                      feed_dict={self._inputs: train_batch_xs,
                                                                 self._targets: train_batch_ts,
                                                                 self._is_training: True})

                self._train_writer.add_summary(op, epoch)

            # on validation samples
            for batch_index in np.arange(n_valid_batches):
                valid_batch_xs, valid_batch_ts, _ = datasets.validation.next_batch(2 * batch_size)
                valid_batch_xs = valid_batch_xs.reshape([-1] + self._images_shape)

                # discriminator
                op, d_loss, _ = self._session.run(fetches=[self._op,
                                                           self._losses.get("discriminator"),
                                                           disc_op],
                                                  feed_dict={self._inputs: valid_batch_xs,
                                                             self._targets: valid_batch_ts,
                                                             self._is_training: False})

                # generator
                op, g_loss, _ = self._session.run(fetches=[self._op,
                                                           self._losses.get("generator"),
                                                           gen_op],
                                                  feed_dict={self._inputs: valid_batch_xs,
                                                             self._targets: valid_batch_ts,
                                                             self._is_training: False})
                self._valid_writer.add_summary(op, epoch)

            if epoch % 5 == 0:
                self.save(location="./mvs/gan/embed_epoch_{}.ckpt".format(epoch))

        # close the writers
        self._train_writer.close(), self._valid_writer.close()

    def _optimizer(self, loss, clip_coef, optimizer):
        grads_and_vars = optimizer().compute_gradients(loss)
        grads_and_vars = [(tf.clip_by_norm(t=grad, clip_norm=clip_coef), var) for grad, var in grads_and_vars]
        train_step = optimizer().apply_gradients(grads_and_vars=grads_and_vars)
        # train_step = optimizer().minimize(loss=loss)
        return train_step

    def apply(self, images):
        # build the model if not built
        if not self._built:
            self._build()
        imgs = images.reshape([-1] + self._images_shape)
        outputs = self._session.run(fetches=self._outputs, feed_dict={self._inputs: imgs,
                                                                      self._is_training: False})
        return outputs

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
        tf.summary.image(name="target-image", tensor=self._targets, max_outputs=batch_size)
        tf.summary.image(name="input-image", tensor=self._inputs, max_outputs=batch_size)
        tf.summary.image(name="output-image", tensor=self._outputs, max_outputs=batch_size)

        #
        tf.summary.scalar(name="l2-reconstruction-loss", tensor=self._losses.get("l2"))
        tf.summary.scalar(name="discriminator-loss", tensor=self._losses.get("discriminator"))
        tf.summary.scalar(name="generator-loss", tensor=self._losses.get("generator"))

        self._op = tf.summary.merge_all()
        self._train_writer = tf.summary.FileWriter("{}/train/".format(location), self._session.graph)
        self._valid_writer = tf.summary.FileWriter("{}/validation/".format(location), self._session.graph)


class Autoencoder_Emb(object):
    def __init__(self, images_shape, embeddings_length=1024, name="auto-encoder"):
        self._images_shape = images_shape
        self._embeddings_length = embeddings_length

        self._name = name

        #
        self._built = False
        self._status = {"discriminator": False,
                        "encoder": False,
                        "decoder": False}

        #
        self._session = tf.Session()

    def _build(self):

        #
        self._is_training = tf.placeholder(dtype=tf.bool, shape=[])

        # inputs:
        # - images (masked, here central area masked)
        # - captions embeddings
        self._inputs = tf.placeholder(dtype=tf.float32, shape=[None] + self._images_shape)
        self._embeddings = tf.placeholder(dtype=tf.float32,
                                          shape=[None, self._embeddings_length])

        # targets:
        # - images (originals, without mask over central area)
        self._targets = tf.placeholder(dtype=tf.float32, shape=[None] + self._images_shape)

        # generate fake images
        fake_images = self._forward_pass(images=self._inputs, embeddings=self._embeddings)
        self._outputs = fake_images

        # discriminate real images, fake images, wrong images (wrong image-to-embeddings relationships [shuffled])
        logits_real = self._discriminator(is_training=self._is_training,
                                          images=self._targets,
                                          embeddings=self._embeddings)
        logits_fake = self._discriminator(is_training=self._is_training,
                                          images=fake_images,
                                          embeddings=self._embeddings)
        logits_wrong = self._discriminator(is_training=self._is_training,
                                           images=tf.random_shuffle(self._targets),
                                           embeddings=self._embeddings)

        # losses:
        # - l2 reconstruction loss
        # - adversarial losses :
        #       > generator loss
        #       > discriminator loss
        self._losses = dict()
        self._losses["l2"] = l2_loss(targets=self._targets, y=fake_images)
        self._losses["generator"] = cross_entropy_loss(logits=logits_fake,
                                                       targets=tf.ones_like(logits_fake))
        d_real = cross_entropy_loss(logits=logits_real, targets=tf.ones_like(logits_real))
        d_fake = cross_entropy_loss(logits=logits_fake, targets=tf.zeros_like(logits_fake))
        d_wrong = cross_entropy_loss(logits=logits_wrong, targets=tf.zeros_like(logits_wrong))
        self._losses["discriminator"] = d_real + d_fake + d_wrong

        #
        self._vars = dict()
        vars = tf.trainable_variables()

        self._vars["discriminator"] = [var for var in vars if "discriminator" in var.name]
        self._vars["generator"] = [var for var in vars if "encoder" in var.name or "decoder" in var.name]

    def build(self):
        self._build()
        return self

    def _forward_pass(self, images, embeddings):
        outputs = self._model(is_training=self._is_training, images=images, embeddings=embeddings)
        outputs = tf.pad(outputs, paddings=[[0, 0], [16, 16], [16, 16], [0, 0]],
                         mode="CONSTANT")  # padding of image_dim / 4
        outputs = images + outputs  # add borders to padded outputs
        return outputs

    def _model(self, is_training, images, embeddings):
        with tf.variable_scope("encoder") as scope:
            if self._status.get("encoder"):
                scope.reuse_variables()

            # pass the image through a sequence of convolution layers

            # 64x64x3 --> 32x32x32
            h0 = conv2d(inputs=images,
                        is_training=is_training,
                        n_output_filters=32, kernel_shape=[5, 5], strides=[2, 2],
                        name="conv-0")
            # 32x32x32 --> 16x16x64
            h1 = conv2d(inputs=h0,
                        is_training=is_training,
                        n_output_filters=32 * 2, kernel_shape=[5, 5], strides=[2, 2],
                        name="conv-1")
            # 16x16x64 --> 8x8x128
            h2 = conv2d(inputs=h1,
                        is_training=is_training,
                        n_output_filters=32 * 4, kernel_shape=[5, 5], strides=[2, 2],
                        name="conv-2")
            # 8x8x128 --> 4x4x256
            h3 = conv2d(inputs=h2,
                        is_training=is_training,
                        n_output_filters=32 * 8, kernel_shape=[5, 5], strides=[2, 2],
                        name="conv-3")

            #
            # h3 = tf.reshape(h3, [-1, 4 * 4 * 512])

            # code = linear(inputs=h3,
            #               is_training=is_training,
            #               n_output_units=4 * 4 * 64 * 8,
            #               name="code")

            # dense block
            # now channel-wise
            _, dim1, dim2, dim3 = h3.get_shape().as_list()
            ndim = dim1 * dim2
            o = []
            for i in range(dim3):
                x = tf.reshape(h3[:, :, :, i], [-1, ndim])
                x = linear(inputs=x,
                           n_output_units=ndim,
                           is_training=is_training,
                           activation_fn=tf.nn.elu,
                           name="dense-{}".format(i))
                x = tf.reshape(x, shape=[-1, dim1, dim2])
                o.append(x)
            code = tf.stack(o, axis=-1, name="code")

            #
            # code = h3

            if not self._status.get("encoder"):
                self._status["encoder"] = True

        with tf.variable_scope("decoder") as scope:
            if self._status.get("decoder"):
                scope.reuse_variables()

            # z = code

            # reduce embeddings
            embs = linear(inputs=embeddings,
                          is_training=is_training,
                          n_output_units=256,
                          name="reduce-embeddings")
            embs = tf.reshape(embs, [-1, 1, 1, 256])
            embs = tf.tile(embs, [1, 4, 4, 1])

            z = tf.concat([code, embs], axis=-1)
            # z = tf.reshape(z, [-1, 4, 4, 256 * 2])

            # pass through deconvolution layers
            strides_ = [2, 2]

            # 4x4x256 --> 8x8x128
            h0_ = deconv2d(inputs=z,
                           is_training=is_training,
                           n_output_filters=32 * 4, kernel_shape=[5, 5], strides=strides_,
                           name="deconv-0") + h2
            # 8x8x128 --> 16x16x64
            h1_ = deconv2d(inputs=h0_,
                           is_training=is_training,
                           n_output_filters=32 * 2, kernel_shape=[5, 5], strides=strides_,
                           name="deconv-1") + h1
            # 16x16x64 --> 32x32x32
            h2_ = deconv2d(inputs=h1_,
                           is_training=is_training,
                           n_output_filters=32, kernel_shape=[5, 5], strides=strides_,
                           name="deconv-2") + h0

            # 32x32x32 --> 32x32x3
            outputs = deconv2d(inputs=h2_,
                               is_training=is_training,
                               n_output_filters=3, kernel_shape=[5, 5], strides=[1, 1],
                               activation_fn=tf.nn.tanh,
                               name="deconv-3")

            if not self._status.get("decoder"):
                self._status["decoder"] = True

            return outputs

    def _discriminator(self, images, embeddings, is_training):
        with tf.variable_scope("discriminator") as scope:
            if self._status.get("discriminator"):
                scope.reuse_variables()

            # pass the image through a sequence of convolution layers

            # 64x64x3 --> 32x32x32
            h0_ = conv2d(inputs=images,
                         is_training=is_training,
                         n_output_filters=32, kernel_shape=[5, 5], strides=[2, 2],
                         name="conv-0")
            # 32x32x32 --> 16x16x64
            h1 = conv2d(inputs=h0_,
                        is_training=is_training,
                        n_output_filters=32 * 2, kernel_shape=[5, 5], strides=[2, 2],
                        name="conv-1")
            # 16x16x64 --> 8x8x128
            h2 = conv2d(inputs=h1,
                        is_training=is_training,
                        n_output_filters=32 * 4, kernel_shape=[5, 5], strides=[2, 2],
                        name="conv-2")
            # 8x8x128 --> 4x4x256
            h3 = conv2d(inputs=h2,
                        is_training=is_training,
                        n_output_filters=32 * 8, kernel_shape=[5, 5], strides=[2, 2],
                        name="conv-3")
            #


            # reduce embeddings to lower representation
            # then reshape and tile them to get a [1, 4, 4, embs_dim]
            embs = linear(inputs=embeddings,
                          is_training=is_training,
                          n_output_units=256,
                          name="reduce-embeddings")
            embs = tf.reshape(embs, [-1, 1, 1, 256])
            embs = tf.tile(embs, [1, 4, 4, 1])

            # concatenate image and embedding representations
            h3 = tf.concat([h3, embs], axis=-1)

            # reshape for linear units
            _, dim1, dim2, dim3 = h3.get_shape().as_list()
            h3 = tf.reshape(h3, [-1, dim1 * dim2 * dim3])

            logits = linear(inputs=h3,
                            is_training=is_training,
                            n_output_units=1,
                            activation_fn=None,
                            name="logits")

            # apply sigmoid
            # they are currently not required
            # outputs = tf.nn.sigmoid(logits)

            if not self._status.get("discriminator"):
                self._status["discriminator"] = True

            return logits

    def train(self, datasets, batch_size=20, max_epoch=50, optimizer=tf.train.AdamOptimizer, clip_coef=1):

        # build model if not already built
        if not self._built:
            self._build()

        # gradient optimization
        alpha = 0.2
        gen_op = self._optimizer((1 - alpha) * self._losses.get("l2") + alpha * self._losses.get("generator"),
                                 clip_coef, optimizer)
        disc_op = self._optimizer(self._losses.get("discriminator"), clip_coef, optimizer)

        #
        self._session.run(tf.global_variables_initializer())

        # set summaries
        self._summaries(location=".", batch_size=batch_size)

        #
        n_train_batches = np.ceil(datasets.train.n_examples / batch_size)
        n_valid_batches = np.ceil(datasets.validation.n_examples / batch_size)

        for epoch in np.arange(1, max_epoch + 1):

            # on training samples
            for batch_index in np.arange(n_train_batches):
                train_batch_xs, train_batch_ts, train_batch_embs = datasets.train.next_batch(batch_size)
                train_batch_xs = train_batch_xs.reshape([-1] + self._images_shape)

                # update discriminator
                op, d_loss, _ = self._session.run(fetches=[self._op,
                                                           self._losses.get("discriminator"),
                                                           disc_op],
                                                  feed_dict={self._inputs: train_batch_xs,
                                                             self._targets: train_batch_ts,
                                                             self._embeddings: train_batch_embs,
                                                             self._is_training: True})

                # update generator (twice to make sure the discriminator's loss does not reach 0
                for k in range(2):
                    op, g_loss, _ = self._session.run(fetches=[self._op,
                                                               self._losses.get("generator"),
                                                               gen_op],
                                                      feed_dict={self._inputs: train_batch_xs,
                                                                 self._targets: train_batch_ts,
                                                                 self._embeddings: train_batch_embs,
                                                                 self._is_training: True})

                self._train_writer.add_summary(op, epoch)

            # on validation samples
            for batch_index in np.arange(n_valid_batches):
                valid_batch_xs, valid_batch_ts, valid_batch_embs = datasets.validation.next_batch(2 * batch_size)
                valid_batch_xs = valid_batch_xs.reshape([-1] + self._images_shape)

                # discriminator
                op, d_loss, _ = self._session.run(fetches=[self._op,
                                                           self._losses.get("discriminator"),
                                                           disc_op],
                                                  feed_dict={self._inputs: valid_batch_xs,
                                                             self._targets: valid_batch_ts,
                                                             self._embeddings: valid_batch_embs,
                                                             self._is_training: False})

                # generator
                op, g_loss, _ = self._session.run(fetches=[self._op,
                                                           self._losses.get("generator"),
                                                           gen_op],
                                                  feed_dict={self._inputs: valid_batch_xs,
                                                             self._targets: valid_batch_ts,
                                                             self._embeddings: valid_batch_embs,
                                                             self._is_training: False})
                self._valid_writer.add_summary(op, epoch)

            if epoch % 5 == 0:
                self.save(location="./mvs/gan/embed_epoch_{}.ckpt".format(epoch))

        # close the writers
        self._train_writer.close(), self._valid_writer.close()

    def _optimizer(self, loss, clip_coef, optimizer):
        grads_and_vars = optimizer().compute_gradients(loss)
        grads_and_vars = [(tf.clip_by_norm(t=grad, clip_norm=clip_coef), var) for grad, var in grads_and_vars]
        train_step = optimizer().apply_gradients(grads_and_vars=grads_and_vars)
        # train_step = optimizer().minimize(loss=loss)
        return train_step

    def apply(self, images, embeddings):
        # build the model if not built
        if not self._built:
            self._build()
        imgs = images.reshape([-1] + self._images_shape)
        outputs = self._session.run(fetches=self._outputs, feed_dict={self._inputs: imgs,
                                                                      self._embeddings: embeddings,
                                                                      self._is_training: False})
        return outputs

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
        tf.summary.image(name="target-image", tensor=self._targets, max_outputs=batch_size)
        tf.summary.image(name="input-image", tensor=self._inputs, max_outputs=batch_size)
        tf.summary.image(name="output-image", tensor=self._outputs, max_outputs=batch_size)

        #
        tf.summary.scalar(name="l2-reconstruction-loss", tensor=self._losses.get("l2"))
        tf.summary.scalar(name="discriminator-loss", tensor=self._losses.get("discriminator"))
        tf.summary.scalar(name="generator-loss", tensor=self._losses.get("generator"))

        self._op = tf.summary.merge_all()
        self._train_writer = tf.summary.FileWriter("{}/train/".format(location), self._session.graph)
        self._valid_writer = tf.summary.FileWriter("{}/validation/".format(location), self._session.graph)
