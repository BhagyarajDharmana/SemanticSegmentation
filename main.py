import HelperFunctions
from six.moves import urllib
from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
import time
from pathlib import Path
from PIL import Image

app = Flask(__name__)

MODEL_NAME = 'xception_coco_voctrainval'
_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
_MODEL_URLS = {'xception_coco_voctrainval': 'deeplabv3_pascal_trainval_2018_01_04.tar.gz'}
_TARBALL_NAME = 'deeplab_model.tar.gz'
download_path = Path(__name__).parent / "xception_coco_voctrainval"


def download_model():
    if not download_path.is_file():
        print('downloading model, this might take a while...')

        urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
                                   download_path)
        print('download completed! loading DeepLab model...')
    else:
        print("Using the dowloaded model")

    model = HelperFunctions.DeepLabModel(download_path)
    print('model loaded successfully!')

    return model


@app.route('/')
def index():
    """
    Renders 'index.html'
    """
    return render_template('index.html')


@app.route('/upload/', methods=['GET', 'POST'])
def file_upload():
    if request.method == 'POST':
        if 'uploadfile' not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files['uploadfile']
        filename = secure_filename(file.filename)
        file_path = Path(__name__).parent / "user_uploads" / filename
        file.save(file_path)
        img_raw = Image.open(file_path)
        t1 = time.time()
        resized_im, seg_map = model.run(img_raw)
        HelperFunctions.vis_segmentation(resized_im, seg_map)
        t2 = time.time()
        print('time: {}'.format(t2 - t1))

        print('detections:')
        print('output saved to: {}'.format('detections.jpg'))

        return send_file(Path(__name__).parent / "detections.jpg")
    return render_template('upload.html')


model = download_model()


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)

