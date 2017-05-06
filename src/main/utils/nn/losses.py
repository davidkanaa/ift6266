import tensorflow as tf


def l2_loss(targets, y):
    loss = tf.reduce_mean(tf.squared_difference(targets, y))
    return loss


def cross_entropy_loss(logits, targets):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=targets)
    loss = tf.reduce_mean(loss)
    return loss


def adversarial_loss(logits_real, logits_fake):
    # compute generator loss
    generator_loss = cross_entropy_loss(logits=logits_fake,
                                        targets=tf.ones_like(logits_fake))

    # compute discriminator argument
    discriminator_loss = cross_entropy_loss(logits=logits_real,
                                            targets=tf.ones_like(logits_real))
    discriminator_loss += cross_entropy_loss(logits=logits_fake,
                                             targets=tf.zeros_like(logits_fake))
    return generator_loss, discriminator_loss


def adversarial_loss_with_embeddings(logits_real, logits_fake, logits_wrong):
    # compute generator loss
    generator_loss = cross_entropy_loss(logits=logits_fake,
                                        targets=tf.ones_like(logits_fake))

    # compute discriminator argument
    discriminator_loss = cross_entropy_loss(logits=logits_real,
                                            targets=tf.ones_like(logits_real))
    discriminator_loss += cross_entropy_loss(logits=logits_fake,
                                             targets=tf.zeros_like(logits_fake))
    discriminator_loss += cross_entropy_loss(logits=logits_wrong,
                                             targets=tf.zeros_like(logits_wrong))
    return generator_loss, discriminator_loss
