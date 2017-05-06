import glob
import os
import pickle as pkl

import numpy as np
from PIL import Image, ImageChops
from nltk.tokenize import RegexpTokenizer

image_size = 64
# Maximum number of captions to use
SEQ_LENGTH = 40


def get_dict_correspondance(worddict="/home/davidkanaa/Documents/UdeM/ift6266_h17_deep-learning/data/inpainting/worddict.pkl"):
    with open(worddict, 'rb') as fd:
        dictionary = pkl.load(fd)

    word_to_ix = {word: i for i, word in enumerate(dictionary)}
    ix_to_word = {i: word for i, word in enumerate(dictionary)}

    # print(ix_to_word[200]) = plane
    # print(word_to_ix["plane"]) = 200

    return [word_to_ix, ix_to_word]


def get_nb_train(data_path="/home/davidkanaa/Documents/UdeM/ift6266_h17_deep-learning/data/inpainting/train2014/"):
    imgs = glob.glob(data_path + "/*.jpg")
    return len(imgs)


def get_nb_val(data_path="/home/davidkanaa/Documents/UdeM/ift6266_h17_deep-learning/data/inpainting/val2014/"):
    imgs = glob.glob(data_path + "/*.jpg")
    return len(imgs)


def get_train_batch(batch_idx, batch_size,
                    data_path="/home/davidkanaa/Documents/UdeM/ift6266_h17_deep-learning/data/inpainting/train2014/",
                    caption_path="/home/davidkanaa/Documents/UdeM/ift6266_h17_deep-learning/data/inpainting/dict_key_imgID_value_caps_train_and_valid.pkl",
                    active_shift=True, active_rotation=True):
    imgs = glob.glob(data_path + "/*.jpg")
    batch_imgs = imgs[batch_idx * batch_size:(batch_idx + 1) * batch_size]

    input_batch = np.empty((0, 3, image_size, image_size), dtype=np.float32)
    target_batch = np.empty((0, 3, image_size // 2, image_size // 2), dtype=np.float32)

    # Read the caption dictionnary (train + valid)
    with open(caption_path, 'rb') as fd:
        caption_dict = pkl.load(fd)

    # Get the correspondance to create the captions array
    [word_to_index, index_to_word] = get_dict_correspondance()

    vocab_size = len(word_to_index)
    # Shape for a 1D-CNN (batch_size, nb_channel = vocab_size, height = SEQ_LENGTH)
    captions_array = np.zeros((batch_size, vocab_size, SEQ_LENGTH), dtype=np.float32)

    # Liste des mots disponibles
    # for x in dictionary:
    #    print(x)

    # Tokenizer wich remove punctuation
    tokenizer = RegexpTokenizer(r'\w+')

    for i, img_path in enumerate(batch_imgs):

        # treat the caption
        cap_id = os.path.basename(img_path)[:-4]

        caption = caption_dict[cap_id]
        tokenize_caption = []

        for j in range(len(caption)):
            tokenize_caption = tokenize_caption + tokenizer.tokenize(caption[j])

        len_caption = len(tokenize_caption)

        # Create the one hot vector for the current sentence
        for j in range(SEQ_LENGTH):
            # If the sentence is smaller than the sentence size we keep 0
            if j < len_caption:
                word = tokenize_caption[j]
                captions_array[i, word_to_index[word], j] = 1.

        # print(np.sum(captions_array[i])) # Give SEQ_LENGHT most of the time the processing seems correct
        img = Image.open(img_path)

        # Dynamic data augmentation

        # rotation aleatoire (dans un angle de 50 deg)
        if active_rotation:
            random_angle = np.random.uniform(-25, 25)
            img = img.rotate(random_angle)

        # shift aleatoire (de 20% de la taille de l'image maximum)
        if active_shift:
            random_y_shift = np.random.randint(-(image_size // 20), image_size // 20)
            random_x_shift = np.random.randint(-(image_size // 20), image_size // 20)
            img = ImageChops.offset(img, random_x_shift, random_y_shift)

        img_array = np.array(img)

        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:
            input = np.copy(img_array)
            input[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = 0
            target = img_array[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :]
            # transform size to fit our neural network
            input = input.transpose(2, 0, 1)
            input = input.reshape(1, 3, image_size, image_size)
            target = target.transpose(2, 0, 1)
            target = target.reshape(1, 3, image_size // 2, image_size // 2)

            # append to the minibatch
            input_batch = np.append(input, input_batch, axis=0)
            target_batch = np.append(target, target_batch, axis=0)
        else:
            input = np.copy(img_array)
            input[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16] = 0
            target = img_array[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16]

            input = input.reshape(1, 1, image_size, image_size)
            input = np.repeat(input, 3, axis=1)
            target = target.reshape(1, 1, image_size // 2, image_size // 2)
            target = np.repeat(target, 3, axis=1)

            input_batch = np.append(input, input_batch, axis=0)
            target_batch = np.append(target, target_batch, axis=0)

    # We want input in the interval [ - 1,  1 ]
    return [(input_batch / 256) * 2 - 1, (target_batch / 256) * 2 - 1, captions_array]


def get_val_batch(batch_idx, batch_size,
                  data_path="/home/davidkanaa/Documents/UdeM/ift6266_h17_deep-learning/data/inpainting/val2014/",
                  caption_path="/home/davidkanaa/Documents/UdeM/ift6266_h17_deep-learning/data/inpainting/dict_key_imgID_value_caps_train_and_valid.pkl",
                  active_shift=True, active_rotation=True):
    imgs = glob.glob(data_path + "/*.jpg")
    batch_imgs = imgs[batch_idx * batch_size:(batch_idx + 1) * batch_size]

    input_batch = np.empty((0, 3, image_size, image_size), dtype=np.float32)
    target_batch = np.empty((0, 3, image_size // 2, image_size // 2), dtype=np.float32)

    # Read the caption dictionnary (train + valid)
    with open(caption_path, 'rb') as fd:
        caption_dict = pkl.load(fd)

    # Get the correspondance to create the captions array
    [word_to_index, index_to_word] = get_dict_correspondance()

    vocab_size = len(word_to_index);
    # Shape for a 1D-CNN (batch_size, nb_channel = vocab_size, height = SEQ_LENGTH)
    captions_array = np.zeros((batch_size, vocab_size, SEQ_LENGTH), dtype=np.float32)

    # Liste des mots disponibles
    # for x in dictionary:
    #    print(x)

    # Tokenizer wich remove punctuation
    tokenizer = RegexpTokenizer(r'\w+')

    for i, img_path in enumerate(batch_imgs):

        # treat the caption
        cap_id = os.path.basename(img_path)[:-4]

        caption = caption_dict[cap_id]
        tokenize_caption = []

        for j in range(len(caption)):
            tokenize_caption = tokenize_caption + tokenizer.tokenize(caption[j])

        len_caption = len(tokenize_caption)

        # Create the one hot vector for the current sentence
        for j in range(SEQ_LENGTH):
            # If the sentence is smaller than the sentence size we keep 0
            if j < len_caption:
                word = tokenize_caption[j]
                captions_array[i, word_to_index[word], j] = 1.

        # print(np.sum(captions_array[i])) # Give SEQ_LENGHT most of the time the processing seems correct
        img = Image.open(img_path)

        # Dynamic data augmentation

        # rotation aleatoire (dans un angle de 50 deg)
        if active_rotation:
            random_angle = np.random.uniform(-25, 25)
            img = img.rotate(random_angle)

        # shift aleatoire (de 20% de la taille de l'image maximum)
        if active_shift:
            random_y_shift = np.random.randint(-(image_size // 20), image_size // 20)
            random_x_shift = np.random.randint(-(image_size // 20), image_size // 20)
            img = ImageChops.offset(img, random_x_shift, random_y_shift)

        img_array = np.array(img)

        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:
            input = np.copy(img_array)
            input[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = 0
            target = img_array[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :]
            # transform size to fit our neural network
            input = input.transpose(2, 0, 1)
            input = input.reshape(1, 3, image_size, image_size)
            target = target.transpose(2, 0, 1)
            target = target.reshape(1, 3, image_size // 2, image_size // 2)

            # append to the minibatch
            input_batch = np.append(input, input_batch, axis=0)
            target_batch = np.append(target, target_batch, axis=0)
        else:
            input = np.copy(img_array)
            input[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16] = 0
            target = img_array[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16]

            input = input.reshape(1, 1, image_size, image_size)
            input = np.repeat(input, 3, axis=1)
            target = target.reshape(1, 1, image_size // 2, image_size // 2)
            target = np.repeat(target, 3, axis=1)

            input_batch = np.append(input, input_batch, axis=0)
            target_batch = np.append(target, target_batch, axis=0)

    # We want input in the interval [ - 1,  1 ]
    return [(input_batch / 256) * 2 - 1, (target_batch / 256) * 2 - 1, captions_array]


class Trainset(object):
    def __init__(self):

        self._n_examples = get_nb_train()
        self._index = 0

        self._images, self._targets, self._captions = self.next_batch(1)

    @property
    def images(self):
        return self._images

    @property
    def captions(self):
        return self._captions

    @property
    def targets(self):
        return self._targets

    @property
    def n_examples(self):
        return self._n_examples

    def next_batch(self, size):
        batch = get_train_batch(self._index, size)

        if self._index > self._n_examples:
            self._index = 0
        else:
            self._index += 1

        return batch


class Valset(object):
    def __init__(self):

        self._n_examples = get_nb_train()
        self._index = 0

        self._images, self._targets, self._captions = self.next_batch(1)

    @property
    def images(self):
        return self._images

    @property
    def captions(self):
        return self._captions

    @property
    def targets(self):
        return self._targets

    @property
    def n_examples(self):
        return self._n_examples

    def next_batch(self, size):
        batch = get_val_batch(self._index, size)

        if self._index > self._n_examples:
            self._index = 0
        else:
            self._index += 1

        return batch


from collections import namedtuple

Datasets = namedtuple(typename="Datasets", field_names=["train", "validation", "test"])

mscoco = Datasets(train=Trainset(), validation=Valset(), test=None)

if __name__ == '__main__':
    # resize_mscoco()
    # show_examples(5, 10)
    # get_nb_train()
    [data_input, data_target, captions_array] = get_train_batch(1, 10)

    # get_dict_correspondance()
