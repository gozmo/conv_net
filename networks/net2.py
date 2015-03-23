from lasagne import layers
from nolearn.lasagne import NeuralNet
from basic_network import BasicNetwork


try:
	from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
	from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError:
	Conv2DLayer = layers.Conv2DLayer
	MaxPool2DLayer = layers.MaxPool2DLayer

class Network(BasicNetwork):
    def __init__(self):
        self.name = "net2"

    def setup_network(self):
        # use the cuda-convnet implementations of conv and max-pool layer
        Conv2DLayer = layers.cuda_convnet.Conv2DCCLayer
        MaxPool2DLayer = layers.cuda_convnet.MaxPool2DCCLayer

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
            input_shape=(None,1, self._input_size),
            conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_ds=(2, 2),
            conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_ds=(2, 2),
            conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_ds=(2, 2),
            hidden4_num_units=500, hidden5_num_units=500,
            output_num_units=self._output_size, output_nonlinearity=None,

            update_learning_rate=0.01,
            update_momentum=0.9,

            regression=True,
            max_epochs=1000,
            verbose=1,
            )
