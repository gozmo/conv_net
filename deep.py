from conv_net.networks.net3 import Network as Net2
from conv_net.networks.net3 import Network as Net3
from conv_net.networks.net3 import Network as Net4
from conv_net.networks.net3 import Network as Net5
from conv_net.networks.net3 import Network as Net6
from conv_net.networks.net3 import Network as Net7

networks = [Net2, Net3, Net4, Net5, Net6, Net7]
#networks = [Net2, Net3]

def train_network(dataset):
    for network_class in networks:
         network = network_class()

         for X,y in dataset.read_training_set():
             network.train(X,y)
         yield network
