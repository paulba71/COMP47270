# Only in the project as a reference...
# No code from here is run in the case studies
# All running code has been refactored into Main.py


# import the networkx network analysis package
import networkx as nx
from itertools import product


# import the plotting functionality from matplotlib
import matplotlib.pyplot as plt

# import Delaunay tesselation
from scipy.spatial import Delaunay

# import kmeans
from scipy.cluster.vq import vq, kmeans, whiten

import numpy as np
import scipy as sp
import random

import platform
import community
import operator

from contextlib import suppress


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


# Modularity taken from networkx source code...
def modularity(G, communities, weight='weight'):
    r"""Returns the modularity of the given partition of the graph.

    Modularity is defined in [1]_ as

    .. math::

        Q = \frac{1}{2m} \sum_{ij} \left( A_{ij} - \frac{k_ik_j}{2m}\right)
            \delta(c_i,c_j)

    where *m* is the number of edges, *A* is the adjacency matrix of
    `G`, :math:`k_i` is the degree of *i* and :math:`\delta(c_i, c_j)`
    is 1 if *i* and *j* are in the same community and 0 otherwise.

    Parameters
    ----------
    G : NetworkX Graph

    communities : list
        List of sets of nodes of `G` representing a partition of the
        nodes.

    Returns
    -------
    Q : float
        The modularity of the paritition.

    Raises
    ------
    NotAPartition
        If `communities` is not a partition of the nodes of `G`.

    Examples
    --------
    >>> G = nx.barbell_graph(3, 0)
    >>> nx.algorithms.community.modularity(G, [{0, 1, 2}, {3, 4, 5}])
    0.35714285714285704

    References
    ----------
    .. [1] M. E. J. Newman *Networks: An Introduction*, page 224.
       Oxford University Press, 2011.

    """
    # if not is_partition(G, communities):
    #    raise NotAPartition(G, communities)

    multigraph = G.is_multigraph()
    directed = G.is_directed()
    m = G.size(weight=weight)
    if directed:
        out_degree = dict(G.out_degree(weight=weight))
        in_degree = dict(G.in_degree(weight=weight))
        norm = 1 / m
    else:
        out_degree = dict(G.degree(weight=weight))
        in_degree = out_degree
        norm = 1 / (2 * m)

    def val(u, v):
        try:
            if multigraph:
                w = sum(d.get(weight, 1) for k, d in G[u][v].items())
            else:
                w = G[u][v].get(weight, 1)
        except KeyError:
            w = 0
        # Double count self-loops if the graph is undirected.
        if u == v and not directed:
            w *= 2
        return w - in_degree[u] * out_degree[v] * norm

    Q = sum(val(u, v) for c in communities for u, v in product(c, repeat=2))
    return Q * norm

# Performance and associated helper functions taken from networkx source code...

def intra_community_edges(G, partition):
    """Returns the number of intra-community edges according to the given
    partition of the nodes of `G`.

    `G` must be a NetworkX graph.

    `partition` must be a partition of the nodes of `G`.

    The "intra-community edges" are those edges joining a pair of nodes
    in the same block of the partition.

    """
    return sum(G.subgraph(block).size() for block in partition)


def inter_community_edges(G, partition):
    """Returns the number of inter-community edges according to the given
    partition of the nodes of `G`.

    `G` must be a NetworkX graph.

    `partition` must be a partition of the nodes of `G`.

    The *inter-community edges* are those edges joining a pair of nodes
    in different blocks of the partition.

    Implementation note: this function creates an intermediate graph
    that may require the same amount of memory as required to store
    `G`.

    """
    # Alternate implementation that does not require constructing a new
    # graph object (but does require constructing an affiliation
    # dictionary):
    #
    #     aff = dict(chain.from_iterable(((v, block) for v in block)
    #                                    for block in partition))
    #     return sum(1 for u, v in G.edges() if aff[u] != aff[v])
    #
    return nx.quotient_graph(G, partition, create_using=nx.MultiGraph()).size()


def inter_community_non_edges(G, partition):
    """Returns the number of inter-community non-edges according to the
    given partition of the nodes of `G`.

    `G` must be a NetworkX graph.

    `partition` must be a partition of the nodes of `G`.

    A *non-edge* is a pair of nodes (undirected if `G` is undirected)
    that are not adjacent in `G`. The *inter-community non-edges* are
    those non-edges on a pair of nodes in different blocks of the
    partition.

    Implementation note: this function creates two intermediate graphs,
    which may require up to twice the amount of memory as required to
    store `G`.

    """
    # Alternate implementation that does not require constructing two
    # new graph objects (but does require constructing an affiliation
    # dictionary):
    #
    #     aff = dict(chain.from_iterable(((v, block) for v in block)
    #                                    for block in partition))
    #     return sum(1 for u, v in nx.non_edges(G) if aff[u] != aff[v])
    #
    #return inter_community_edges(nx.complement(G), partition)


def performance(G, partition):
    """Returns the performance of a partition.

    The *performance* of a partition is the ratio of the number of
    intra-community edges plus inter-community non-edges with the total
    number of potential edges.

    Parameters
    ----------
    G : NetworkX graph
        A simple graph (directed or undirected).

    partition : sequence

        Partition of the nodes of `G`, represented as a sequence of
        sets of nodes. Each block of the partition represents a
        community.

    Returns
    -------
    float
        The performance of the partition, as defined above.

    Raises
    ------
    NetworkXError
        If `partition` is not a valid partition of the nodes of `G`.

    References
    ----------
    .. [1] Santo Fortunato.
           "Community Detection in Graphs".
           *Physical Reports*, Volume 486, Issue 3--5 pp. 75--174
           <http://arxiv.org/abs/0906.0612>

    """
    # Compute the number of intra-community edges and inter-community
    # edges.
    intra_edges = intra_community_edges(G, partition)
    inter_edges = inter_community_non_edges(G, partition)
    # Compute the number of edges in the complete graph (directed or
    # undirected, as it depends on `G`) on `n` nodes.
    #
    # (If `G` is an undirected graph, we divide by two since we have
    # double-counted each potential edge. We use integer division since
    # `total_pairs` is guaranteed to be even.)
    n = len(G)
    total_pairs = n * (n - 1)
    if not G.is_directed():
        total_pairs //= 2
    return (intra_edges + inter_edges) / total_pairs


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


def count_edge_cuts(graph, w0, w1, w2, method):
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
    print('Community detection method is: ', method)
    print('Edge cuts: ', edge_cut_count)
    print('Contained edges: ', edge_uncut_count)
    return edge_cut_count, edge_uncut_count


def count_edge_cuts_from_list(graph, list_of_partitions, method):
    edge_cut_count = 0
    edge_uncut_count = 0
    for edge in graph.edges_iter():
        found = False
        for lst in list_of_partitions:
            # This may be inefficient but I'll just check if both nodes are in 0, 1, or two
            if edge[0] in lst and edge[1] in lst and not found:
                edge_uncut_count += 1
                found = True
        if not found:
            edge_cut_count += 1
    print('Community detection method is: ', method)
    print('Edge cuts: ', edge_cut_count)
    print('Contained edges: ', edge_uncut_count)
    return edge_cut_count, edge_uncut_count


def get_modularity(graph, list_of_partitions):
    print("Calculating modularity")
    mod = modularity(graph, list_of_partitions)
    return mod

def get_performance(graph, list_of_partitions):
    print("Calculating performance")
    # Commented out as it not working as expected...
    # perf = performance(graph, list_of_partitions)
    # return perf


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


def cluster_nodes(graph, feat, pos, eigen_pos, type):
    book,distortion = kmeans(feat, 3)
    codes,distortion = vq(feat, book)

    nodes = np.array(range(graph.number_of_nodes()))
    W0 = nodes[codes == 0].tolist()
    W1 = nodes[codes == 1].tolist()
    W2 = nodes[codes == 2].tolist()
    print("W0 ", W0)
    print("W1 ", W1)
    print("W2 ", W2)
    count_edge_cuts(graph, W0, W1, W2, type)
    communities = list()
    communities.append(W0)
    communities.append(W1)
    communities.append(W2)
    mod = get_modularity(graph, communities)
    print("Modularity: ", mod)



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
        degree = graph.degree(node)
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
    #components = sorted(communities, key=len, reverse=True)
    print("====================")
    print("Community detected the following number of partitions: ", len(community_count))
    print("====================")
    for i in community_count:
        print("Count for community {} is {}.".format(i, communities.count(i)))
    return communities


def girvan_newman (G):

    if len(G.nodes()) == 1:
        return [G.nodes()]

    def find_best_edge(G0):
        """
        Networkx implementation of edge_betweenness
        returns a dictionary. Make this into a list,
        sort it and return the edge with highest betweenness.
        """
        eb = nx.edge_betweenness_centrality(G0)
        eb_il = eb.items()
        #eb_il.sort(key=lambda x: x[1], reverse=True)
        eb_il_sorted=sorted(eb_il, key=lambda x: x[1], reverse=True)
        return eb_il_sorted[0][0]

    components = list(nx.connected_component_subgraphs(G))

    while len(components) == 1:
        G.remove_edge(*find_best_edge(G))
        components = list(nx.connected_component_subgraphs(G))

    result = [c.nodes() for c in components]

    looper = 0
    for c in components:
        print("Call number: ", looper)
        looper += 1
        result.extend(girvan_newman(c))

    return result


def get_girvan_newman_communities(G):
    comp = girvan_newman(G)
    print("====================")
    print("girvan_newman detected the following number of partitions: ", len(comp))
    print("====================")
    return comp


def remove_neighbours(graph, node, neighbours):
    with suppress(Exception):   # Needed if the edge was already removed.
        first = True
        for neighbour in neighbours:
            if first == False:
                graph.remove_edge(node, neighbour)
            first = False
    return graph


def get_naive_partitions(graph):
    # nodes = graph.nodes
    # Calculate betweenness centrality
    print("Number of connected components before: ", nx.number_connected_components(graph))
    bt = nx.betweenness_centrality(graph)
    sorted_bt = sorted(bt.items(), key=operator.itemgetter(1))
    sorted_bt.reverse()
    sorted_list = list(sorted_bt)
    # Remove the edges until we have 3 connected components
    node_index = 0
    while nx.number_connected_components(graph) < 3:
        print("Removing neighbours from node: ",node_index)
        top_node = sorted_list[node_index][0]
        top_neighbours=nx.neighbors(graph,top_node)
        graph = remove_neighbours(graph, top_node, top_neighbours)
        node_index += 1
    print("Number of connected components after: ", nx.number_connected_components(graph))
    print("Number of nodes whose neighbours had to be removed: ", node_index)
    components = sorted(nx.connected_components(graph), key = len, reverse=True)
    return_components = list()
    for i in range(nx.number_connected_components(graph)):
        print(components[i])
        return_components.append(components[i])
    return return_components


def get_community_from_list(community_list, index):
    return_list = list()
    node = 0
    for i in community_list:
        if community_list[node] == index:
            return_list.append(node)
        node += 1
    return return_list


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
    list_of_partitions = list()
    list_of_partitions.append(W0)
    list_of_partitions.append(W1)
    list_of_partitions.append(W2)
    count_edge_cuts_from_list(G, list_of_partitions, "Random")
    mod = get_modularity(G, list_of_partitions)
    print("Modularity: ", mod)
    perf = get_performance(G, list_of_partitions)
    print("Performance: ", perf)

    # Networkx community partitioning
    partitions = get_community_partitions(G)
    partitions_count = set(partitions)
    list_of_partitions = list()
    length = len(partitions_count)
    for i in range(length):
        comm = get_community_from_list(partitions, i)
        print(comm)
        list_of_partitions.append(comm)
    count_edge_cuts_from_list(G, list_of_partitions, "Communities extension")
    mod = get_modularity(G,list_of_partitions)
    print("Modularity: ", mod)
    perf = get_performance(G, list_of_partitions)
    print("Performance: ", perf)

    # NetworkX Girvan Newman partitioning
    G = nx.Graph(A)
    gncomps = get_girvan_newman_communities(G)
    count_edge_cuts_from_list(G, gncomps, "Girvan Newman")
    mod = get_modularity(G, gncomps)
    print("Modularity: ", mod)

    # My naive partitioning algorithm.
    G = nx.Graph(A)
    communities = get_naive_partitions(G)
    # Rebuild the graph as the last operation was destructive
    G = nx.Graph(A)
    count_edge_cuts_from_list(G, communities, "My naive algorithm")
    mod = get_modularity(G, communities)
    print("Modularity: ", mod)

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
    cluster_nodes(G,features,pos,eigen_pos, "eigenvectors")

    # Finally, use the columns of A directly for clustering
    cluster_nodes(G,A.todense(),pos,eigen_pos, "Adjacency matrix")

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
    list_of_partitions = list()
    list_of_partitions.append(W0)
    list_of_partitions.append(W1)
    list_of_partitions.append(W2)
    count_edge_cuts_from_list(G, list_of_partitions, "Random")
    mod = get_modularity(G, list_of_partitions)
    print("Modularity: ", mod)

    # Networkx community partitioning
    partitions = get_community_partitions(G)
    partitions_count = set(partitions)
    list_of_partitions = list()
    length = len(partitions_count)
    for i in range(length):
        comm = get_community_from_list(partitions, i)
        print(comm)
        list_of_partitions.append(comm)
    count_edge_cuts_from_list(G, list_of_partitions, "Communities extension")
    mod = get_modularity(G, list_of_partitions)
    print("Modularity: ", mod)

    # My naive partitioning algorithm.
    G = nx.Graph(A)
    communities = get_naive_partitions(G)
    # Rebuild the graph as the last operation was destructive
    G = nx.Graph(A)
    count_edge_cuts_from_list(G, communities, "My naive algorithm")
    mod = get_modularity(G, communities)
    print("Modularity: ", mod)

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

    plot_graph(G, eigen_pos, 3)

    # Now let's see if the eigenvectors are good for clustering
    # Use kmeans to cluster the points in the vector V

    features = np.column_stack((V[:, 1], V[:, 2]))
    cluster_nodes(G, features, pos, eigen_pos, "eigenvectors")

    # Finally, use the columns of A directly for clustering
    cluster_nodes(G, A.todense(), pos, eigen_pos, "adjacency matrix")

    # NetworkX Girvan Newman partitioning - moved to the bottom for perf reasons...
    G = nx.Graph(A)
    gncomps = get_girvan_newman_communities(G)
    count_edge_cuts_from_list(G, gncomps, "Girvan Newman")
    mod = get_modularity(G, gncomps)
    print("Modularity: ", mod)

    input("Press Enter to Continue ...")


# Run an example, comment out random model or facebook model to prevent them running...
# run_random_models()
run_facebook_models()