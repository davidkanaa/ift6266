

def train_epoch(model, saving_each_iter=10):
    nb_train_iter = (len(model.cfg.queue.filename) * model.cfg.queue.nb_examples_per_file) // model.batch_size
    for i in trange(nb_train_iter, leave=False, desc="Training iteration"):
        op = [model.train_fn]
        if i % saving_each_iter == 0:
            op.append(model.merged_summary_op)
        out = model.sess.run(op, feed_dict={model.is_training: True})

        if i % saving_each_iter == 0:
            current_iter = model.sess.run(model.global_step)
            model.summary_writer.add_summary(out[1], global_step=current_iter)

    if not os.path.exists("model"):
        os.makedirs("model")
    current_iter = model.sess.run(model.global_step)
    model.saver.save(model.sess, "model/model", global_step=current_iter)


def train_adversarial_epoch(model, saving_each_iter=100):
    n_train_critic = model.cfg.gan.n_train_critic
    n_train_generator = model.cfg.gan.n_train_generator

    j = 0

    if model.epoch < model.cfg.gan.intense_starting_period:
        n_train_critic = model.cfg.gan.n_train_critic_intense
    for _ in trange(n_train_critic, desc="Train critic", leave=False):
        op = [model.train_dis]
        j += 1
        if j % saving_each_iter == 0:
            op.append(model.merged_summary_op)
        out = model.sess.run(op, feed_dict={model.is_training: True})
        if j % saving_each_iter == 0:
            current_iter = model.sess.run(model.global_step)
            model.summary_writer.add_summary(out[1], global_step=current_iter)

    for _ in trange(n_train_generator, desc="Train generator", leave=False):
        op = [model.train_gen]
        j += 1
        if j % saving_each_iter == 0:
            op.append(model.merged_summary_op)

        out = model.sess.run(op, feed_dict={model.is_training: True})
        if j % saving_each_iter == 0:
            current_iter = model.sess.run(model.global_step)
            model.summary_writer.add_summary(out[1], global_step=current_iter)

    if not os.path.exists("model"):
        os.makedirs("model")
    current_iter = model.sess.run(model.global_step)
    model.saver.save(model.sess, "model/model", global_step=current_iter)


def compute_restart_epoch(model):
    current_step = model.global_step.eval(model.sess)

    if not model.adv_training:
        return current_step // (
            (len(model.cfg.queue.filename) * model.cfg.queue.nb_examples_per_file) // model.batch_size)
    else:
        n_iter_starting = model.cfg.gan.n_train_critic_intense * model.cfg.gan.intense_starting_period \
                          + model.cfg.gan.n_train_generator
        if current_step - n_iter_starting >= 0:
            current_step -= n_iter_starting
            n_iter = model.cfg.gan.n_train_critic + model.cfg.gan.n_train_generator
            return current_step // n_iter
        else:
            return current_step // n_iter_starting