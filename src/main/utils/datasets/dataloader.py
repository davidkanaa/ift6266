from collections import namedtuple

import numpy as np

import os
import pickle


def mask(images):
    height = images.shape[1]
    width = images.shape[2]
    n_channels = images.shape[3]
    x = images.copy()

    h = height // 4
    w = width // 4
    x[:, h:3 * h, w:3 * w, :] = 0

    return x.reshape([-1, height, width, n_channels])


def load_mscoco(location):
    train_loc = os.path.join(location, "train")
    val_loc = os.path.join(location, "val")

    # training set
    train = Dataset(location=train_loc)

    # validation
    validation = Dataset(location=val_loc)

    return Datasets(train=train, validation=validation, test=None)


class Dataset(object):
    def __init__(self, location):
        self._location = location

        self._examples = os.listdir(location)
        self._n_examples = len(self._examples)
        self._examples = np.random.permutation(self._examples)
        self._index = 0

    @property
    def n_examples(self):
        return self._n_examples

    def next_batch(self, size):
        begin = self._index
        self._index += size
        if self._index > self._n_examples:
            #
            self._examples = np.random.permutation(self._examples)

            #
            begin = 0
            self._index = size
        end = self._index

        images = []
        targets = []
        embeddings = []
        for filename in self._examples[begin:end]:
            filename = os.path.join(self._location, filename)
            data = pickle.load(open(filename, "rb"))
            targets.append(data.get("image"))
            embeddings.append(data.get("embedding"))
            #targets.append(data)

        targets = np.divide(np.stack(targets, axis=0), 255, dtype=np.float32)  # normalise the values of pixels
        images = mask(targets)
        embeddings = np.stack(embeddings)

        return images, targets, embeddings


Datasets = namedtuple(typename="Datasets", field_names=["train", "validation", "test"])
