import nltk
import pickle
from collections import OrderedDict
import os
import argparse

import numpy as np
from termcolor import cprint


def text2index(text, dictionary):
    """
    Converts the string `text` into a vector of indices using `dictionary`.
    Args:
        text: 
        dictionary: 

    Returns:

    """
    if not isinstance(text, str):
        raise ValueError("")
    tokens = nltk.tokenize.word_tokenize(text.lower())
    return list(map(lambda x: dictionary.get(x), tokens))


def image_captions2index(captions, dictionary):
    """
    Converts the set of image captions in `captions` to vectors of indices using `dictionary`.
    Args:
        captions: 
        dictionary: 

    Returns:

    """

    # number of captions
    n_captions = np.sum(len(captions.get(img)) for img in captions.keys())
    length = np.max(for s in captions.get() for img in captions.keys())

    #
    counter = 0
    for img in captions.keys():
        vectors = list(map(lambda s: text2index(s, dictionary), captions.get(img)))
        captions[img] = vectors
        counter += len(vectors)
        ratio = 100 * np.divide(counter, n_captions, dtype=np.float32)

        cprint(text="Encoded captions: {} %".format(ratio),
               color="yellow",
               attrs=["bold"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dictionary', type=str, default='Data/sample_captions.txt',
                        help='caption file')
    parser.add_argument('--location', type=str, default='Data/sample_captions.txt',
                        help='caption file')
    parser.add_argument('--destination', type=str, default='Data/caption_vectors.txt',
                        help='caption file')

    args = parser.parse_args()

    #
    caps = pickle.load(open(args.location, "rb"))
    dico = pickle.load(open(args.dictionary))

    #
    image_captions2index(captions=caps, dictionary=dico)

    cprint(
        text="\nFinished encoding captions from {} to vectors and saved to {}".format(args.location, args.destination),
        color="green",
        attrs=["bold"])

    pickle.dump(caps, open(args.destination, "wb"), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()