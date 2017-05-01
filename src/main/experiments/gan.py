from src.main.utils.datasets.mscoco import load_mscoco
from src.main.models.gan.model import DCGAN_Emb

# location of the pickled mscoco datas et
loc = "/home/davidkanaa/Documents/UdeM/ift6266_h17_deep-learning/project/MSCOCO_data/"

# load the data set
from src.main.utils.read_data import mscoco
coco = mscoco  # load_mscoco(location=loc)

# batch size
b = 80

# maximum number of epochs
e = 10

#
model = DCGAN_Emb(images_shape=[64, 64, 3])
model.train(datasets=coco, batch_size=b, max_epoch=e)
