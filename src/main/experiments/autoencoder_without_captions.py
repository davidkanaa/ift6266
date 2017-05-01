from src.main.utils.datasets.mscoco import load_mscoco
from src.main.models.autoencoder.model import Autoencoder

# location of the pickled mscoco datas et
loc = "/home/davidkanaa/Documents/UdeM/ift6266_h17_deep-learning/project/MSCOCO_data/"

# load the data set
coco = load_mscoco(location=loc)

# batch size
b = 80

# maximum number of epochs
e = 10

#
model = Autoencoder(images_shape=[64, 64, 3])
model.train(datasets=coco, batch_size=b, max_epoch=e)

# @Todo: apply model on test set
