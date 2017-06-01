import networkx as nx
import numpy as np
import scipy as sp
import random
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


def main():
    graph = load_graph("facebook")