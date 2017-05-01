import os
import pickle
from collections import OrderedDict

import numpy as np
from scipy import misc


def do(emb_loc, train_loc, val_loc, dest):
    embs = pickle.load(open(emb_loc, "rb"))
    for img in embs.keys():
        dir_ = train_loc if "train" in img else val_loc
        image = misc.imread("{}/{}.jpg".format(os.path.dirname(dir_), img), mode="RGB")
        image = np.asarray(image)
        caps = embs.get(img)
        for k in range(caps.shape[0]):
            if not os.path.exists("{}/{}/{}/".format(dest, os.path.dirname(dir_), img)):
                os.makedirs("{}/{}/{}/".format(dest, os.path.dirname(dir_), img))
            f = open("{}/{}/{}/data{}.pkl".format(dest, os.path.dirname(dir_), img, k + 1), "wb")
            d = OrderedDict()
            d["image"] = image
            d["embedding"] = caps[k]
            pickle.dump(image, file=f, protocol=pickle.HIGHEST_PROTOCOL)


def do2(dic_loc, train_dest, val_dest):
    dico = pickle.load(open(dic_loc, "rb"))
    train = OrderedDict()
    val = OrderedDict()

    for img in dico.keys():
        if "train" in img:
            train[img] = len(dico.get(img))
        else:
            val[img] = len(dico.get(img))

    pickle.dump(train, open(os.path.join(train_dest, "train.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(val, open(os.path.join(val_dest, "val.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
