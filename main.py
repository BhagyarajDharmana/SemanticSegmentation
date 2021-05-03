import HelperFunctions
import tempfile
from six.moves import urllib
import tensorflow as tf
from PIL import Image
import os

MODEL_NAME = 'xception_coco_voctrainval'
_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
_MODEL_URLS = {'xception_coco_voctrainval': 'deeplabv3_pascal_trainval_2018_01_04.tar.gz'}
_TARBALL_NAME = 'deeplab_model.tar.gz'

model_dir = tempfile.mkdtemp()
tf.io.gfile.makedirs(model_dir)

download_path = os.path.join(model_dir, _TARBALL_NAME)
print('downloading model, this might take a while...')
urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
                           download_path)
print('download completed! loading DeepLab model...')
MODEL = HelperFunctions.DeepLabModel(download_path)
print('model loaded successfully!')

raw_img = Image.open("img.png")
resized_im, seg_map = MODEL.run(raw_img)
HelperFunctions.vis_segmentation(resized_im, seg_map)
