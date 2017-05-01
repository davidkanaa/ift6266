import os
import pickle

import numpy as np
from scipy import misc
from tensorflow.python.framework import dtypes

import src.main.utils.term as term
from src.main.utils.datasets.datasets import Dataset, Datasets

# ----------------------------------------------------------------------------------------- #
#  the file names for the datasets
# ----------------------------------------------------------------------------------------- #

TRAIN_IMAGES = "train.pkl"
VALID_IMAGES = "valid.pkl"


# ----------------------------------------------------------------------------------------- #

def mask(images):
    height = images.shape[1]
    width = images.shape[2]
    n_channels = images.shape[3]
    x = images.copy()

    h = height // 4
    w = width // 4
    x[:, h:3 * h, w:3 * w, :] = 0

    return x.reshape([-1, height, width, n_channels])


def pickle_dataset(location, destination, name):
    """
    
    Args:
        location: 
        destination: 
        name: 

    Returns:

    """
    images = []
    file_names = os.listdir(location)
    count = 0
    for filename in file_names:
        images.append(misc.imread(os.path.join(location, filename), mode="RGB"))
        count += 1
        term.progress("Pickling images: {}%".format(count // len(file_names)))

    images = np.asarray(images)

    #
    _name_ = name
    if not os.path.exists(destination):
        os.makedirs(destination)
    f = open(os.path.join(destination, "{}.pkl".format(_name_)), "wb")
    pickle.dump(obj=images, file=f, protocol=pickle.HIGHEST_PROTOCOL)

    term.success("{} images pickled.".format(_name_))


def pickle_datasets(location, destination):
    """
    
    Args:
        location:
        destination: 

    Returns:

    """
    #
    train_dir = os.path.join(location, TRAIN_IMAGES)
    valid_dir = os.path.join(location, VALID_IMAGES)

    #
    pickle_dataset(train_dir, destination, name="train")
    pickle_dataset(valid_dir, destination, name="valid")


def unpickle_dataset(location):
    """
    
    Args:
        location: 

    Returns:

    """
    term.progress("Loading dataset from {} ... ".format(location.name))
    images = pickle.load(location)
    term.success("Done loading {}".format(location.name))

    return images


def unpickle_datasets(location):
    #
    train_loc = open(os.path.join(location, TRAIN_IMAGES), mode="rb")
    valid_loc = open(os.path.join(location, VALID_IMAGES), mode="rb")

    #
    train_images = unpickle_dataset(train_loc)
    valid_images = unpickle_dataset(valid_loc)

    return train_images, valid_images


def read_data_sets(location, test=5000, dtype=dtypes.float32):
    """
    
    Args:
        location:
        test: 
        dtype: 

    Returns:

    """
    with open(os.path.join(location, TRAIN_IMAGES), mode="rb") as f:
        train_images = unpickle_dataset(f)

    with open(os.path.join(location, VALID_IMAGES), mode="rb") as f:
        valid_images = unpickle_dataset(f)

    # training set
    train = Dataset(train_images, mask(train_images), dtype=dtype)

    # validation & test set
    valid_images = np.random.permutation(valid_images)
    valid_images, test_images = valid_images[test:], valid_images[:test]

    validation = Dataset(valid_images, mask(valid_images), dtype=dtype)
    test = Dataset(test_images, mask(test_images), dtype=dtype)

    return Datasets(train=train, validation=validation, test=test)


def load_mscoco(location):
    return read_data_sets(location=location)
