import nltk
import pickle


def build_dictionary(text):
    if len(text) > 1:
        text = "".join(text)
    vocab = set(nltk.tokenize.word_tokenize(text.lower()))

    # add special tokens
    specials = dict()
    specials["_null_"] = -1
    specials["_eos_"] = 0
    specials["_unk_"] = 1

    # add words (tokens) from the vocabulary
    words = {w: i + len(specials) for i, w in enumerate(vocab)}

    # merging into dictionary
    worddict = {
        **specials,
        **words
    }

    return worddict


def save_dictionary(dictionary, location):
    with open(location, "wb") as file:
        pickle.dump(dictionary, file)


def load_dictionary(location):
    with open(location, "rb") as file:
        worddict = pickle.load(file)

    return worddict
