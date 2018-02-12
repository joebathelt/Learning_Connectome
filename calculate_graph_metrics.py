#! /usr/bin/env python
import optparse
import os
import re
import sys


def main():
    import os
    import sys
    p = optparse.OptionParser()

    p.add_option('--network_file', '-a')
    p.add_option('--out_filename', '-o')

    options, arguments = p.parse_args()
    network_file =  options.network_file
    out_filename = options.out_filename

    def calculate_graph_metrics(network_file):
        import sys
        sys.path.append('/home/jb07/nipype_installation/')

        import bct
        import numpy as np

        network = np.loadtxt(network_file)
        clustering_coefficient = np.mean(bct.clustering_coef_wu(network))
        global_efficiency = bct.charpath(bct.distance_wei(bct.weight_conversion(network, 'lengths'))[0])[1]

        # Normalize with a random version
        random_clustering_coefficient = list()
        random_global_efficiency = list()

        for i in np.arange(0,20): # creating 20 scrambled versions of the original network
            random_network = network.copy()
            random_network = random_network[np.random.permutation(random_network.shape[0])][np.random.permutation(random_network.shape[0])]
            random_clustering_coefficient.append(np.mean(bct.clustering_coef_wu(random_network)[~np.isinf(bct.clustering_coef_wu(random_network))]))
            random_global_efficiency.append(np.mean(bct.charpath(bct.distance_wei(bct.weight_conversion(random_network, 'lengths'))[0])[1]))

        random_clustering_coefficient = np.mean(random_clustering_coefficient)
        random_global_efficiency = np.mean(random_global_efficiency)

        # Normalize the graph metrics with the measures from the scrambled version
        average_clustering = clustering_coefficient/random_clustering_coefficient
        global_efficiency = global_efficiency/random_global_efficiency

        np.savetxt(out_filename, [average_clustering, global_efficiency])

    calculate_graph_metrics(network_file)

if __name__ == '__main__':
    # main should return 0 for success, something else (usually 1) for error.
    sys.exit(main())
