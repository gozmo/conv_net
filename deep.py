from conv_net.networks.net2 import Network as Net2
from conv_net.networks.net3 import Network as Net3
from conv_net.networks.net4 import Network as Net4
from conv_net.networks.net5 import Network as Net5
from conv_net.networks.net6 import Network as Net6
from conv_net.networks.net7 import Network as Net7
from utils import log

#networks = [Net2, Net3, Net4, Net5, Net6, Net7]
networks = [Net4, Net5, Net6, Net7]

def train_network(dataset):
    for network_class in networks:
        network = network_class()
        log("Training network: %s" % ( network.name))

        for X,y in dataset.read_training_set():
            log("training new batch")
            network.train(X,y)
        yield network
