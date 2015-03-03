# add to kfkd.py
import os
import numpy as np
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from matplotlib import pyplot
import lasagne.layers.cuda_convnet
import theano

try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer


FTRAIN = '~/projects/conv_nn_facial_keypoints/data/training.csv'
FTEST = '~/projects/conv_nn_facial_keypoints/data/test.csv'

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

def float32(k):
    return np.cast['float32'](k)

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

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

if __name__ == "__main__":
    # use the cuda-convnet implementations of conv and max-pool layer
    Conv2DLayer = layers.cuda_convnet.Conv2DCCLayer
    MaxPool2DLayer = layers.cuda_convnet.MaxPool2DCCLayer

    net7 = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', Conv2DLayer),
            ('pool1', MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),  # !
            ('conv2', Conv2DLayer),
            ('pool2', MaxPool2DLayer),
            ('dropout2', layers.DropoutLayer),  # !
            ('conv3', Conv2DLayer),
            ('pool3', MaxPool2DLayer),
            ('dropout3', layers.DropoutLayer),  # !
            ('hidden4', layers.DenseLayer),
            ('dropout4', layers.DropoutLayer),  # !
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],

        input_shape=(None, 1, 96, 96),
        conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_ds=(2, 2),
        dropout1_p=0.1,  # !
        conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_ds=(2, 2),
        dropout2_p=0.2,  # !
        conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_ds=(2, 2),
        dropout3_p=0.3,  # !
        hidden4_num_units=1000,
        dropout4_p=0.5,  # !
        hidden5_num_units=1000,
        output_num_units=30, output_nonlinearity=None,

        update_learning_rate=theano.shared(float32(0.03)),
        update_momentum=theano.shared(float32(0.9)),

        regression=True,
        batch_iterator_train=FlipBatchIterator(batch_size=128),
        on_epoch_finished=[
        	AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
	        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],
        max_epochs=10000,
        verbose=1,
        )

    X, y = load2d()  # load 2-d data
    net7.fit(X, y)

    # Training for 1000 epochs will take a while.  We'll pickle the
    # trained model so that we can load it back later:
    import cPickle as pickle
    import sys
    sys.setrecursionlimit(10000)
    with open('net7.pickle', 'wb') as f:
        pickle.dump(net7, f, -1)


    def plot_sample(x, y, axis):
        img = x.reshape(96, 96)
        axis.imshow(img, cmap='gray')
        axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

    X, _ = load2d(test=True)
    y_pred = net7.predict(X)

    fig = pyplot.figure(figsize=(6, 6))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(X[i], y_pred[i], ax)

    #pyplot.show()

    from sklearn.metrics import mean_squared_error
    print mean_squared_error(net7.predict(X), y)
