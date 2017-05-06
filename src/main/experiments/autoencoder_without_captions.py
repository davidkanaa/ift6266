from src.main.utils.datasets.mscoco import load_mscoco
from src.main.utils.datasets.dataloader import load_mscoco
from src.main.models.autoencoder.model import Autoencoder_Emb

# location of the pickled mscoco datas et
loc = "/home/davidkanaa/Documents/UdeM/ift6266_h17_deep-learning/data/inpainting/all2014/"  #"/home/davidkanaa/Documents/UdeM/ift6266_h17_deep-learning/project/MSCOCO_data/"

# load the data set
coco = load_mscoco(location=loc)

# batch size
b = 100

# maximum number of epochs
e = 10

#
model = Autoencoder_Emb(images_shape=[64, 64, 3])
model.train(datasets=coco, batch_size=b, max_epoch=e)

# @Todo: apply model on test set
