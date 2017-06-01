import random

import HelperMethods as helpers
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy as sp
import scipy.sparse
from scipy.cluster.vq import vq, kmeans
from scipy.spatial import Delaunay

facebook = "C:\\Users\\paulb\\PycharmProjects\\COMP47270\\Datasets\\facebook_combined.txt"
twitter = "C:\\Users\\paulb\\PycharmProjects\\COMP47270\\Datasets\\twitter_combined.txt"
google = "C:\\Users\\paulb\\PycharmProjects\\COMP47270\\Datasets\\web-Google1.txt"
roads_CA = "C:\\Users\\paulb\\PycharmProjects\\COMP47270\\Datasets\\roadNet-CA.txt"
amazon = "C:\\Users\\paulb\\PycharmProjects\\COMP47270\\Datasets\\com-amazon-ungraph.txt"
college = "C:\\Users\\paulb\\PycharmProjects\\COMP47270\\Datasets\\collegemsg.txt"


def get_default_graph():
    num_nodes = 100
    x = [random.random() for i in range(num_nodes)]
    y = [random.random() for i in range(num_nodes)]

    x = np.array(x)
    y = np.array(y)

    # Make a graph with num_nodes nodes and zero edges
    # Plot the nodes using x,y as the node positions

    G = nx.Graph()
    for i in range(num_nodes):
        node_name = str(i)
        G.add_node(node_name)

    # Now add some edges - use Delaunay tesselation
    # to produce a planar graph. Delaunay tesselation covers the
    # convex hull of a set of points with triangular simplices (in 2D)

    points = np.column_stack((x, y))
    dl = Delaunay(points)
    tri = dl.simplices

    edges = np.zeros((2, 6 * len(tri)), dtype=int)
    data = np.ones(6 * len(points))
    j = 0
    for i in range(len(tri)):
        edges[0][j] = tri[i][0]
        edges[1][j] = tri[i][1]
        j += 1
        edges[0][j] = tri[i][1]
        edges[1][j] = tri[i][0]
        j += 1
        edges[0][j] = tri[i][0]
        edges[1][j] = tri[i][2]
        j +=  1
        edges[0][j] = tri[i][2]
        edges[1][j] = tri[i][0]
        j += 1
        edges[0][j] = tri[i][1]
        edges[1][j] = tri[i][2]
        j += 1
        edges[0][j] = tri[i][2]
        edges[1][j] = tri[i][1]
        j += 1

    data = np.ones(6 * len(tri))
    A = sp.sparse.csc_matrix((data, (edges[0, :], edges[1, :])))

    for i in range(A.nnz):
        A.data[i] = 1.0

    G = nx.to_networkx_graph(A)

    return G


def load_graph(name, as_directed=False):
    graph = None
    if name == 'facebook':
        print('Loading facebook graph')
        if as_directed:
            graph = nx.read_edgelist(facebook,create_using=nx.DiGraph())
        else:
            graph = nx.read_edgelist(facebook)

    if name == 'twitter':
        print('Loading twitter graph')
        if as_directed:
            graph = nx.read_edgelist(twitter, create_using=nx.DiGraph())
        else:
            graph = nx.read_edgelist(twitter)

    if name == 'google':
        print('Loading google graph')
        if as_directed:
            graph = nx.read_edgelist(google, create_using=nx.DiGraph())
        else:
            graph = nx.read_edgelist(google)

    if name == 'amazon':
        print('Loading amazon graph')
        if as_directed:
            graph = nx.read_edgelist(amazon, create_using=nx.DiGraph())
        else:
            graph = nx.read_edgelist(amazon)

    if name == 'roads_CA':
        print('Loading roads_CA graph')
        if as_directed:
            graph = nx.read_edgelist(roads_CA, create_using=nx.DiGraph())
        else:
            graph = nx.read_edgelist(roads_CA)

    if name == 'college':
        print('Loading collegemsg')
        if as_directed:
            graph = nx.read_edgelist(college, create_using=nx.DiGraph())
        else:
            graph = nx.read_edgelist(college)

    if graph is None:
        graph = get_default_graph()

    return graph


def plot_graph(G, pos, fignum, draw_edges=True):
    fig = plt.figure(fignum, figsize=(8, 8))
    fig.clf()
    nx.draw_networkx_nodes(G,
                            pos,
                            node_size=20,
                            hold=False,
                        )

    if(draw_edges):
        nx.draw_networkx_edges(G,pos, hold=True)
    fig.show()


def repos_with_eigs(A, num_nodes):

    # Use the eigenvectors of the normalised Laplacian to calculate placement positions
    # for the nodes in the graph

    #   eigen_pos holds the positions
    print("Re-plotting using eigenvectors")

    eigen_pos = dict()
    deg = A.sum(0)
    diags = np.array([0])
    D = sp.sparse.spdiags(deg, diags, A.shape[0], A.shape[1])
    Dinv = sp.sparse.spdiags(1 / deg, diags, A.shape[0], A.shape[1])
    # Normalised laplacian
    L = Dinv * (D - A)
    E, V = sp.sparse.linalg.eigs(L, 3, None, 100.0, 'SM')
    V = V.real

    for i in range(num_nodes):
        eigen_pos[i] = V[i, 1].real, V[i, 2].real

    return eigen_pos, V


def cluster_nodes_from_original_example(G, feat, pos, eigen_pos):
    book,distortion = kmeans(feat,3)
    codes,distortion = vq(feat, book)

    nodes = np.array(range(G.number_of_nodes()))
    W0 = nodes[codes == 0].tolist()
    W1 = nodes[codes == 1].tolist()
    W2 = nodes[codes == 2].tolist()
    print("W0 ", W0)
    print("W1 ", W1)
    print("W2 ", W2)

    # Show the clusters in different colours... Red (already drawn), Blue and Magenta
    plt.figure(3)
    nx.draw_networkx_nodes(G, eigen_pos, node_size=40, hold=True, nodelist=W0, node_color='m')
    nx.draw_networkx_nodes(G, eigen_pos, node_size=40, hold=True, nodelist=W1, node_color='b')
    plt.figure(2)
    nx.draw_networkx_nodes(G, pos, node_size=40, hold=True, nodelist=W0, node_color='m')
    nx.draw_networkx_nodes(G, pos, node_size=40, hold=True, nodelist=W1, node_color='b')


def cluster_nodes_from_networkx(G):
    print ("Clustering using the networkx library")


def cluster_from_my_algoritm(G):
    print("Clustering from my own algorithm")


def measure_clustering_quality(G):
    print("Measure the quality of the clustering")


def analyse_strongly_connected_components(G):
    try:
        num_strongly_connected_components = nx.number_strongly_connected_components(G)
        print("Strongly Connected Component Count - ", num_strongly_connected_components)
        comps = helpers.get_strongly_connected_components(G)
        component_sizes = []
        for nodes in comps:
            component_sizes.extend([len(nodes)])
        component_sizes.sort(reverse=True)
        print("Biggest connected component size: ", component_sizes[0])
        # Get in degree
        indegrees = G.in_degree().values()
        incount = 0
        for degree in indegrees:
            if degree == 0:
                incount += 1
        print("In degree - ", incount)

        #Get out degree
        outdegrees = G.out_degree().values()
        outcount = 0
        for degree in outdegrees:
            if degree == 0:
                outcount += 1
        print("Outdegree = ", outcount)

    except Exception as e:
        print("Exception: - ", e)


def analyse_directed_graph():
    graph = load_graph('college', True)
    # helpers.draw_graph(graph)
    print("Graph nodes: ", graph.number_of_nodes())
    print("Graph edges: ", graph.number_of_edges())  # already has edges from dataset...

    analyse_strongly_connected_components(graph)

    # Draw loglog graph...
    helpers.plot_degseq(graph, False)

    # wait at end...
    input("Press Enter to Continue ...")


def main():

    graph = load_graph("facebook")
    helpers.plot_degseq(graph, False)
    num_nodes = graph.number_of_nodes()

    x = [random.random() for i in range(num_nodes)]
    y = [random.random() for i in range(num_nodes)]

    x = np.array(x)
    y = np.array(y)

    pos = dict()
    for i in range(num_nodes):
        pos[i] = x[i], y[i]

    print("Graph nodes: ", graph.number_of_nodes())
    print("Graph edges: ", graph.number_of_edges()) # already has edges from dataset...
    print("Pos nodes: ", len(pos))

    # plot the graph...
    A = nx.adjacency_matrix(graph)
    graph = nx.Graph(A)
    print("Graph nodes after rebuild: ", graph.number_of_nodes())
    print("Graph edges after rebuild: ", graph.number_of_edges())  # already has edges from dataset...
    # Plot the nodes only
    plot_graph(graph, pos, 1, False)
    # plot the nodes and edges
    plot_graph(graph, pos, 2)

    # reposition with the eigenvectors
    eigenv_pos, V = repos_with_eigs(A, num_nodes)
    plot_graph(graph, eigenv_pos, 3)

    print("Plotting spring layout")
    pos=nx.spring_layout(graph)
    plot_graph(graph, pos, 4)

    # Look at the clustering
    features = np.column_stack((V[:, 1], V[:, 2]))
    cluster_nodes_from_original_example(graph, features, pos, eigenv_pos)

    # Finally, use the columns of A directly for clustering
    # cluster_nodes_from_original_example(graph, A.todense(), pos, eigenv_pos)

    # wait at end...
    input("Press Enter to Continue ...")


#main()

analyse_directed_graph()



