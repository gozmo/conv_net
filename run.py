import argparse
import imp
import utils
import sys

sys.path.append("networks")

def run_networks(network_names):
    networks = []
    for network_name in network_names:
        f, filename, description = imp.find_module(network_name, ["networks"])
        network_module = imp.load_module(network_name, f, filename, description)
        network = network_module.Network()
        networks.append(network)

    X_2d, y_2d = utils.load2d()  # load 2-d data
    X_1d, y_1d = utils.load()
    for network in networks:
        if network.name == "net1":
            print "net1"
            network.train(X_1d, y_1d)
        else:
            print network.name
            network.train(X_2d, y_2d)

if '__main__' == __name__:
    parser = argparse.ArgumentParser(description='Train networks')
    network_names = [("net%s" % x) for x in xrange(1,8)]
    for network_name in network_names:
        parser.add_argument("--" + network_name, action="store_true")
    parser.add_argument("--all")
    args = parser.parse_args()

    if not args.all:
        network_names = [ network_name for network_name \
                             in network_names \
                             if getattr(args, network_name)]

    run_networks(network_names)
