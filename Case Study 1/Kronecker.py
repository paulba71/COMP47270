import numpy as np
import networkx as nx
from HelperMethods import draw_graph


def create_kronecker():
    initiator = np.matrix('1 1 1 1 1 1; 1 1 0 0 0 0; 1 0 1 0 0 0; 1 0 0 1 0 0; 1 0 0 0 1 0; 1 0 0 0 0 1')

    print(initiator)
    print(type(initiator))
    kron_result=initiator
    #build this up to the power of 5
    for i in range(5):
        print (i)
        kron_result = np.kron(kron_result, initiator)

    print (kron_result)
    G=nx.from_numpy_matrix(kron_result)
    draw_graph(G)
    print('')
    return G


G = create_kronecker()
#perform the analysis of this graph.

