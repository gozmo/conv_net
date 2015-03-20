import cPickle as pickle
import numpy as np
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from sklearn.metrics import mean_squared_error
import sys
from nolearn.lasagne import BatchIterator

#to be able to store big networks
sys.setrecursionlimit(100000)

try:
	from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
	from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError:
	Conv2DLayer = layers.Conv2DLayer
	MaxPool2DLayer = layers.MaxPool2DLayer

class BasicNetwork:
    name = "basic_network"

    def train(self, X, y):
        self._input_size = len(X[0])
        self._output_size = len(y[0])

        self.setup_network()

        self._net.fit(X, y)

    def setup_network(X,y):
        print "#"*10 + "\nself._run needs to be implemented, running basic network\n" + "#"*10
        self._net = NeuralNet(
            layers=[   #three layers: one hidden layer
                ('input', layers.InputLayer),
                ('hidden', layers.DenseLayer),
                ('output', layers.DenseLayer),
                ],
             #layer parameters:
            input_shape=(128, self._input_size),  # 128 images per batch times 96x96 input pixels
            hidden_num_units=100,  # number of units in hidden layer
            output_nonlinearity=None,  # output layer uses identity function
            output_num_units=self._output_size,  # 30 target values

            ## optimization method:
            update=nesterov_momentum,
            update_learning_rate=0.01,
            update_momentum=0.9,

            regression=True,  # flag to indicate we're dealing with regression problem
            max_epochs=200,  # we want to train this many epochs
            verbose=1,
            )

    def save_net(self):
        with open('%s.pickle'% self.name, 'wb') as f:
            pickle.dump(self._net, f, -1)

    def predict(self, X):
        return self._net.predict(X)

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
