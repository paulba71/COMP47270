# Only in the project as a reference...
# No code from here is run in the case studies
# All running code has been refactored into Main.py


# import the networkx network analysis package
import networkx as nx
from networkx.algorithms import community


# import the plotting functionality from matplotlib
import matplotlib.pyplot as plt

# import Delaunay tesselation
from scipy.spatial import Delaunay

# import kmeans
from scipy.cluster.vq import vq, kmeans, whiten

import itertools
import numpy as np
import scipy as sp
import random

import platform
import community


if platform.system() != "Darwin":
    facebook = "C:\\Users\\paulb\\PycharmProjects\\COMP47270\\Datasets\\facebook_combined.txt"
    twitter = "C:\\Users\\paulb\\PycharmProjects\\COMP47270\\Datasets\\twitter_combined.txt"
    google = "C:\\Users\\paulb\\PycharmProjects\\COMP47270\\Datasets\\web-Google1.txt"
    roads_CA = "C:\\Users\\paulb\\PycharmProjects\\COMP47270\\Datasets\\roadNet-CA.txt"
    amazon = "C:\\Users\\paulb\\PycharmProjects\\COMP47270\\Datasets\\com-amazon-ungraph.txt"
    college = "C:\\Users\\paulb\\PycharmProjects\\COMP47270\\Datasets\\collegemsg.txt"
else:
    facebook = "/Users/paulba/PycharmProjects/COMP47270/Datasets/facebook_combined.txt"
    twitter = "/Users/paulba/PycharmProjects/COMP47270/Datasets/twitter_combined.txt"
    google = "/Users/paulba/PycharmProjects/COMP47270/Datasets/web-Google1.txt"
    roads_CA = "/Users/paulba/PycharmProjects/COMP47270/Datasets/roadNet-CA.txt"
    amazon = "/Users/paulba/PycharmProjects/COMP47270/Datasets/com-amazon-ungraph.txt"
    college = "/Users/paulba/PycharmProjects/COMP47270/Datasets/collegemsg.txt"


def get_default_graph(get_edges):
    num_nodes = 100
    x = [random.random() for i in range(num_nodes)]
    y = [random.random() for i in range(num_nodes)]

    x = np.array(x)
    y = np.array(y)

    # Make a graph with num_nodes nodes and zero edges
    # Plot the nodes using x,y as the node positions

    graph = nx.Graph()
    for i in range(num_nodes):
        node_name = str(i)
        graph.add_node(node_name)

    # Now add some edges - use Delaunay tesselation
    # to produce a planar graph. Delaunay tesselation covers the
    # convex hull of a set of points with triangular simplices (in 2D)

    points = np.column_stack((x, y))
    dl = Delaunay(points)
    tri = dl.simplices

    if get_edges:
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
            j += 1
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
        adjacency_matrix = sp.sparse.csc_matrix((data, (edges[0, :], edges[1, :])))

        for i in range(adjacency_matrix.nnz):
            adjacency_matrix.data[i] = 1.0

        graph = nx.to_networkx_graph(adjacency_matrix)

    return graph


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
        graph = get_default_graph(True)

    return graph


def get_random_positions(graph_size):
    x = [random.random() for i in range(graph_size)]
    y = [random.random() for i in range(graph_size)]

    x = np.array(x)
    y = np.array(y)

    pos = dict()
    for i in range(graph_size):
        pos[i] = x[i], y[i]

    return pos


def get_default_edges(graph):
    num_nodes = graph.number_of_nodes()
    A = nx.adjacency_matrix(graph)
    x = [random.random() for i in range(num_nodes)]
    y = [random.random() for i in range(num_nodes)]

    x = np.array(x)
    y = np.array(y)

    # Now add some edges - use Delaunay tesselation
    # to produce a planar graph. Delaunay tesselation covers the
    # convex hull of a set of points with triangular simplices (in 2D)

    points = np.column_stack((x, y))
    dl = Delaunay(points)
    tri = dl.simplices

    edges = np.zeros((2, 6 * len(tri)), dtype=int)
    # data = np.ones(6 * len(points))
    j = 0
    for i in range(len(tri)):
        edges[0][j] = tri[i][0]
        edges[1][j] = tri[i][1]
        j = j + 1
        edges[0][j] = tri[i][1]
        edges[1][j] = tri[i][0]
        j = j + 1
        edges[0][j] = tri[i][0]
        edges[1][j] = tri[i][2]
        j = j + 1
        edges[0][j] = tri[i][2]
        edges[1][j] = tri[i][0]
        j = j + 1
        edges[0][j] = tri[i][1]
        edges[1][j] = tri[i][2]
        j = j + 1
        edges[0][j] = tri[i][2]
        edges[1][j] = tri[i][1]
        j = j + 1

    data = np.ones(6 * len(tri))
    adjacency_matrix = sp.sparse.csc_matrix((data, (edges[0, :], edges[1, :])))

    for i in range(adjacency_matrix.nnz):
        adjacency_matrix.data[i] = 1.0

    graph = nx.to_networkx_graph(adjacency_matrix)
    return graph


def plot_graph(graph, pos, fig_num):

    label = dict()
    label_pos = dict()
    for i in range(graph.number_of_nodes()):
        label[i] = i
        label_pos[i] = pos[i][0]+0.02, pos[i][1]+0.02

    fig = plt.figure(fig_num, figsize=(8, 8))
    fig.clf()
    nx.draw_networkx_nodes(graph, pos, node_size=40, hold=False)
    nx.draw_networkx_edges(graph, pos, hold=True)
    nx.draw_networkx_labels(graph, label_pos, label, font_size=10, hold=True)
    fig.show()


def count_edge_cuts(graph, w0, w1, w2):
    edge_cut_count = 0
    edge_uncut_count = 0
    for edge in graph.edges_iter():
        # This may be inefficient but I'll just check if both nodes are in 0, 1, or two
        if edge[0] in w0 and edge[1] in w0:
            edge_uncut_count += 1
        elif edge[0] in w1 and edge[1] in w1:
            edge_uncut_count += 1
        elif edge[0] in w2 and edge[1] in w2:
            edge_uncut_count += 1
        else:
            edge_cut_count += 1
    print('Edge cuts: ', edge_cut_count)
    print('Contained edges: ', edge_uncut_count)
    return edge_cut_count, edge_uncut_count


def random_partition(graph):
    w0 = []
    w1 = []
    w2 = []
    for node in graph.nodes_iter():
        set_choice = random.randint(0, 2)
        if set_choice == 0:
            w0.append(node)
        if set_choice == 1:
            w1.append(node)
        if set_choice == 2:
            w2.append(node)
    return w0, w1, w2


def cluster_nodes(graph, feat, pos, eigen_pos):
    book,distortion = kmeans(feat, 3)
    codes,distortion = vq(feat, book)

    nodes = np.array(range(graph.number_of_nodes()))
    W0 = nodes[codes == 0].tolist()
    W1 = nodes[codes == 1].tolist()
    W2 = nodes[codes == 2].tolist()
    print("W0 ", W0)
    print("W1 ", W1)
    print("W2 ", W2)
    count_edge_cuts(graph, W0, W1, W2)

    plt.figure(3)
    nx.draw_networkx_nodes(graph, eigen_pos, node_size=40, hold=True, nodelist=W0, node_color='m')
    nx.draw_networkx_nodes(graph, eigen_pos, node_size=40, hold=True, nodelist=W1, node_color='b')

    plt.figure(2)
    nx.draw_networkx_nodes(graph, pos, node_size=40, hold=True, nodelist=W0, node_color='m')
    nx.draw_networkx_nodes(graph, pos, node_size=40, hold=True, nodelist=W1, node_color='b')


def analyse_graph (graph):
    print("Graph dimensions:")
    print("=================")
    print(nx.info(graph))
    max_degree = 0
    min_degree = 999999
    ave_degree = 0
    counter = 0
    for node in graph.nodes():
        degree=graph.degree(node)
        if degree > max_degree:
            max_degree = degree
        if min_degree > degree:
            min_degree = degree
        ave_degree += degree
        counter += 1
    ave_degree = ave_degree / counter

    print ("====================")
    print ("Max node degree: ", max_degree)
    print ("Min node degree: ", min_degree)
    print ("Ave node degree: ", ave_degree)
    print ("====================")


def get_community_partitions(graph):
    partitions = community.best_partition(graph)
    communities = [partitions.get(node) for node in graph.nodes()]
    community_count = set(communities)
    print("====================")
    print("Community detected the following number of partitions: ", len(community_count))
    print("====================")



# def get_naive_partitions(graph):
    # nodes=graph.

def run_random_models():
    G = get_default_graph(False)
    pos = get_random_positions(G.number_of_nodes())
    # Need to rebuild the graph...
    A = nx.adjacency_matrix(G)  # will use the adjacency matrix later
    G = nx.Graph(A)

    # Draw the nodes
    plot_graph(G, pos, 1)

    num_nodes = G.number_of_nodes()

    # Add some edges
    G = get_default_edges(G)
    # Need to rebuild the graph...
    A = nx.adjacency_matrix(G)
    G = nx.Graph(A)

    # Print out the graph info...
    analyse_graph(G)

    # Draw the connected graph
    plot_graph(G,pos,2)

    # Get a random partition
    W0, W1, W2 = random_partition(G)
    count_edge_cuts(G, W0, W1, W2)

    partitions = get_community_partitions(G)

    # Try using girvan_newman implementation in networkx
    #k = 3
    #G = nx.Graph(A)
    #comp = community.girvan_newman(G)
    #limited = itertools.takewhile(lambda c: len(c) <= k, comp)
    #for communities in limited:
    #    print(tuple(sorted(c) for c in next(comp)))

    # Use the networkx implementation
    #communities = list(nx.k_clique_communities(G, num_nodes/3))
    #print("networkx W0: ", communities[0])
    #print("networkx W1: ", communities[1])
    #print("networkx W2: ", communities[2])

    # Use the eigenvectors of the normalised Laplacian to calculate placement positions
    # for the nodes in the graph
    # eigen_pos holds the positions
    eigen_pos = dict()
    deg = A.sum(0)
    diags = np.array([0])
    D = sp.sparse.spdiags(deg,diags,A.shape[0],A.shape[1])
    Dinv = sp.sparse.spdiags(1/deg,diags,A.shape[0],A.shape[1])
    # Normalised laplacian
    L = Dinv*(D - A)
    E, V= sp.sparse.linalg.eigs(L, 3, None, 100.0, 'SM')
    V = V.real

    for i in range(num_nodes):
        eigen_pos[i] = V[i, 1].real, V[i, 2].real


    # for n,nbrsdict in G.adjacency_iter():
    #     for nbr,eattr in nbrsdict.items():
    #         if 'weight' in eattr:
    #             print n,nbr,eattr['weight']

    plot_graph(G,eigen_pos,3)


    # Now let's see if the eigenvectors are good for clustering
    # Use kmeans to cluster the points in the vector V

    features = np.column_stack((V[:,1], V[:,2]))
    cluster_nodes(G,features,pos,eigen_pos)

    # Finally, use the columns of A directly for clustering
    cluster_nodes(G,A.todense(),pos,eigen_pos)

    input("Press Enter to Continue ...")


def run_facebook_models():
    G = load_graph("facebook")
    pos = get_random_positions(G.number_of_nodes())
    # Need to rebuild the graph...
    A = nx.adjacency_matrix(G)  # will use the adjacency matrix later
    G = nx.Graph(A)

    # Draw the nodes
    plot_graph(G, pos, 1)

    num_nodes = G.number_of_nodes()

    # Print out the graph info...
    analyse_graph(G)

    # Draw the connected graph
    plot_graph(G, pos, 2)

    # Get a random partition
    W0, W1, W2 = random_partition(G)
    count_edge_cuts(G, W0, W1, W2)

    partitions = get_community_partitions(G)

    # Try using girvan_newman implementation in networkx
    # k = 3
    # G = nx.Graph(A)
    # comp = community.girvan_newman(G)
    # limited = itertools.takewhile(lambda c: len(c) <= k, comp)
    # for communities in limited:
    #    print(tuple(sorted(c) for c in next(comp)))



    # Use the eigenvectors of the normalised Laplacian to calculate placement positions
    # for the nodes in the graph
    # eigen_pos holds the positions
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

    # for n,nbrsdict in G.adjacency_iter():
    #     for nbr,eattr in nbrsdict.items():
    #         if 'weight' in eattr:
    #             print n,nbr,eattr['weight']

    plot_graph(G, eigen_pos, 3)

    # Now let's see if the eigenvectors are good for clustering
    # Use kmeans to cluster the points in the vector V

    features = np.column_stack((V[:, 1], V[:, 2]))
    cluster_nodes(G, features, pos, eigen_pos)

    # Finally, use the columns of A directly for clustering
    cluster_nodes(G, A.todense(), pos, eigen_pos)

    input("Press Enter to Continue ...")


# Run an example.
# run_random_models()
run_facebook_models()