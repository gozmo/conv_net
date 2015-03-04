from lasagne import layers
from sklearn.metrics import mean_squared_error
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet
import lasagne.layers.cuda_convnet
import theano
import utils

try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer

class Network:
    def __init__(self):
        self.name = "net7"

    def run(self,X,y):
        # use the cuda-convnet implementations of conv and max-pool layer
        Conv2DLayer = layers.cuda_convnet.Conv2DCCLayer
        MaxPool2DLayer = layers.cuda_convnet.MaxPool2DCCLayer

        net = NeuralNet(
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

            update_learning_rate=theano.shared(utils.float32(0.03)),
            update_momentum=theano.shared(utils.float32(0.9)),

            regression=True,
            batch_iterator_train=utils.FlipBatchIterator(batch_size=128),
            on_epoch_finished=[
                utils.AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
                utils.AdjustVariable('update_momentum', start=0.9, stop=0.999),
            ],
            max_epochs=10000,
            verbose=1,
            )

        net.fit(X, y)

        utils.save_net(net, "net7")

        print mean_squared_error(net.predict(X), y)
