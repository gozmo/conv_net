from conv_net.networks.net2 import Network as Net2
from conv_net.networks.net3 import Network as Net3
from conv_net.networks.net4 import Network as Net4
from conv_net.networks.net5 import Network as Net5
from conv_net.networks.net6 import Network as Net6
from conv_net.networks.net7 import Network as Net7
from utils import log
from utils import quadratic_weighted_kappa

#networks = [Net2, Net3, Net4, Net5, Net6, Net7]
networks_dict = {"net2": Net2, "net3":Net3, "net4":Net4, "net5":Net5, "net6":Net6, "net7":Net7}
networks = [Net4, Net5, Net6, Net7]

def train_network(dataset):
    for network_class in networks:
        network = network_class()
        log("Training network: %s" % ( network.name))

        for X,y in dataset.read_training_set():
            log("training new batch")
            network.train(X,y)
        yield network

def train_network_cross_validation(dataset, folds, network_name):
    network_class = networks_dict[network_name]
    scores = []
    kfold = 3
    log("Training network: %s" % ( network_name))

    for x in xrange(kfold):
        results_scalar = []
        target_scalar = []
        for X,y in dataset.read_training_set_cross_validation(x):
            network = network_class()
            network.train(X,y)

            validation_set = dataset.get_validation_set()
            results= network.predict(validation_set)
            results_scalar += [result.argmax() for result in results]
            target_scalar += [target.argmax() for target in y]
        kappa = quadratic_weighted_kappa(results_scalar, target_scalar)
        scores.append(kappa)
        log("\tfold: %s, kappa: %s" % (x, kappa))
    log("Cross validation finished: %s" % scores)
