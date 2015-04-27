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
        self._shape = X[0].shape
        self._input_size = len(X[0])
        self._output_size = len(y[0])

        self.setup_network()

        print X.shape
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


    def float32(self, k):
        return np.cast['float32'](k)

class FlipBatchIterator(BatchIterator):

    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

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
        new_value = np.cast['float32'](self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)
