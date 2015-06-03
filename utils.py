import cPickle as pickle
from nolearn.lasagne import BatchIterator
from pandas.io.parsers import read_csv
from matplotlib import pyplot
from sklearn.utils import shuffle
import numpy as np
import os
from PIL import Image
from threading import Lock,Thread
from Queue import Queue
import logging
from time import gmtime, strftime

FTRAIN = 'data/training.csv'
FTEST = 'data/test.csv'
DATEFORMAT = "%Y-%m-%d_%H:%M:%S"

if not os.path.exists("logs"):
    os.makedirs("logs")
time_stamp = strftime(DATEFORMAT, gmtime())
logging.basicConfig(filename='logs/%s.log' % (time_stamp),
                    level=logging.DEBUG,
                    datefmt=DATEFORMAT,
                    )

def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

def load2d(test=False, cols=None):
    X, y = load(test=test)
    X = X.reshape(-1, 1, 96, 96)
    return X, y



def plot_result(net):
    def plot_sample(x, y, axis):
        img = x.reshape(96, 96)
        axis.imshow(img, cmap='gray')
        axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

    X, _ = load(test=True)
    y_pred = net.predict(X)

    fig = pyplot.figure(figsize=(6, 6))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(X[i], y_pred[i], ax)

    pyplot.show()

class FlipBatchIterator(BatchIterator):
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
        ]

    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        if yb is not None:
            # Horizontal flip of all x coordinates:
            yb[indices, ::2] = yb[indices, ::2] * -1

            # Swap places, e.g. left_eye_center_x -> right_eye_center_x
            for a, b in self.flip_indices:
                yb[indices, a], yb[indices, b] = (
                    yb[indices, b], yb[indices, a])

        return Xb, yb



def resize_dataset(source_path, target_path, resolution):
    resize_process = ResizeProcess(source_path, target_path, resolution)
    resize_process.get_files()
    resize_process.start()


class ResizeProcess:
    def __init__(self, source_path, target_path, resolution):
        self._lock = Lock()
        self._source_path = source_path
        self._target_path = target_path
        self._resolution = resolution
        self.image_files = []

    def get_files(self):
        VALID_IMAGE_FORMATS = ["jpeg", "jpg", "png"]
        self.image_files = [filename for filename
                           in os.listdir(self._source_path)]

    def start(self):
        Thread(target=self._resize_image_thread).start()
        Thread(target=self._resize_image_thread).start()
        Thread(target=self._resize_image_thread).start()
        Thread(target=self._resize_image_thread).start()
        Thread(target=self._resize_image_thread).start()
        Thread(target=self._resize_image_thread).start()
        Thread(target=self._resize_image_thread).start()
        Thread(target=self._resize_image_thread).start()

    def _resize_image_thread(self):
        while len(self.image_files) > 0:
            with self._lock:
                filename = self.image_files.pop()

            image = self._load_image(self._source_path, filename)
            resized_image = self._resize_image(image, self._resolution)
            self._write_image(resized_image, filename, self._target_path)

    def _load_image(self, path, filename):
        filepath = path + os.sep + filename
        return Image.open(open(filepath))

    def _resize_image(self, image, resolution):
        return image.resize(resolution)

    def _write_image(self, image, filename, path):
        filepath = path + os.sep + filename
        image.save(filepath)

def log(string):
    time_stamp = strftime(DATEFORMAT, gmtime())
    logging.info(string)
    print time_stamp, string
