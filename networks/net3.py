from nolearn.lasagne import NeuralNet
from lasagne import layers
from sklearn.metrics import mean_squared_error
import lasagne.layers.cuda_convnet
from basic_network import BasicNetwork
from basic_network import FlipBatchIterator

try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError:
    Conv2DLayer =layers.Conv2DLayer
    MaxPool2DLayer =layers.MaxPool2DLayer

class Network(BasicNetwork):
    def __init__(self):
        self.name = "net3"

    def setup_network(self):
        Conv2DLayer =layers.cuda_convnet.Conv2DCCLayer
        MaxPool2DLayer =layers.cuda_convnet.MaxPool2DCCLayer

        self._net = NeuralNet(
            layers=[
                ('input', layers.InputLayer),
                ('conv1', Conv2DLayer),
                ('pool1', MaxPool2DLayer),
                ('conv2', Conv2DLayer),
                ('pool2', MaxPool2DLayer),
                ('conv3', Conv2DLayer),
                ('pool3', MaxPool2DLayer),
                ('hidden4', layers.DenseLayer),
                ('hidden5', layers.DenseLayer),
                ('output', layers.DenseLayer),
                ],
            input_shape=(None, 1, 96, 96),
            conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_ds=(2, 2),
            conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_ds=(2, 2),
            conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_ds=(2, 2),
            hidden4_num_units=500, hidden5_num_units=500,
            output_num_units=30, output_nonlinearity=None,

            update_learning_rate=0.01,
            update_momentum=0.9,

            regression=True,
        batch_iterator_train=FlipBatchIterator(batch_size=128),
            max_epochs=3000,
            verbose=1,
            )
