import numpy as np
import networkx as nx
import HelperMethods as helpers


def create_kronecker():
    initiator = np.matrix('1 1 1 1 1 1; 1 1 0 0 0 0; 1 0 1 0 0 0; 1 0 0 1 0 0; 1 0 0 0 1 0; 1 0 0 0 0 1')
    alternative_initiator=np.matrix('1 1 0 0 0 1; 1 1 0 0 1 0; 0 0 1 1 0 0; 0 0 1 1 0 0; 0 1 0 0 1 1; 1 0 0 0 1 1')

    print ("Initiator matrix")
    print(initiator)
    kron_result=alternative_initiator
    #build this up to the power of 5
    for i in range(4):
        print(i)
        kron_result = np.kron(kron_result, alternative_initiator)
    print (kron_result)
    print("Building the graph from the matrix")
    G = nx.from_numpy_matrix(kron_result)
    return G


def analsys_graph(G):
    #print("Drawing the graph")
    #helpers.draw_graph(G)
    print("analysing the graph")
    print("Nodes: ", G.number_of_nodes())
    print("Edges: ", G.number_of_edges())
    # Draw loglog graph...
    #DG = G.to_directed()
    print("Drawing the loglog graph")
    helpers.plot_degseq(G, False)
    print('')


G = create_kronecker()
#perform the analysis of this graph.
analsys_graph(G)
