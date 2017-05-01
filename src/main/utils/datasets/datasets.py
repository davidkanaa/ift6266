from collections import namedtuple

import numpy as np


class Dataset(object):
    def __init__(self, images, targets, captions=None, dtype=None):
        self._images = images
        self._targets = targets
        self._captions = captions

        self._n_examples = images.shape[0]
        self._index = 0

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
        begin = self._index
        self._index += size
        if self._index > self._n_examples:
            #
            perm = np.random.permutation(self._n_examples)
            self._images = self._images[perm]
            self._targets = self._targets[perm]

            #
            begin = 0
            self._index = size
        end = self._index
        return self._images[begin:end], self._targets[begin:end]


Datasets = namedtuple(typename="Datasets", field_names=["train", "validation", "test"])
