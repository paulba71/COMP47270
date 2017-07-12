# Only in the project as a reference...
# No code from here is run in the case studies
# All running code has been refactored into Main.py


# import the networkx network analysis package
import networkx as nx

# import the plotting functionality from matplotlib
import matplotlib.pyplot as plt

#import Delaunay tesselation
from scipy.spatial import Delaunay

# import kmeans
from scipy.cluster.vq import vq, kmeans, whiten

import numpy as np
import scipy as sp
import random

import platform


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


def get_random_positions(graph_size):
    x = [random.random() for i in range(graph_size)]
    y = [random.random() for i in range(graph_size)]

    x = np.array(x)
    y = np.array(y)

    pos = dict()
    for i in range(graph_size):
        pos[i] = x[i], y[i]

    return pos


def placement():
    #Start of commented out code

    G = get_default_graph(False)
    pos = get_random_positions(G.number_of_nodes())
    # Need to rebuild the graph...
    A = nx.adjacency_matrix(G)  # will use the adjacency matrix later
    G = nx.Graph(A)

    plot_graph(G, pos, 1)

    num_nodes = G.number_of_nodes()
    x = [random.random() for i in range(num_nodes)]
    y = [random.random() for i in range(num_nodes)]

    x = np.array(x)
    y = np.array(y)

    # Now add some edges - use Delaunay tesselation
    # to produce a planar graph. Delaunay tesselation covers the
    # convex hull of a set of points with triangular simplices (in 2D)

    points = np.column_stack((x,y))
    dl=Delaunay(points)
    tri = dl.simplices

    edges = np.zeros((2, 6*len(tri)),dtype=int)
    data=np.ones(6*len(points))
    j=0
    for i in range(len(tri)):
        edges[0][j]=tri[i][0]
        edges[1][j]=tri[i][1]
        j = j+1
        edges[0][j]=tri[i][1]
        edges[1][j]=tri[i][0]
        j = j+1
        edges[0][j]=tri[i][0]
        edges[1][j]=tri[i][2]
        j = j+1
        edges[0][j]=tri[i][2]
        edges[1][j]=tri[i][0]
        j = j+1
        edges[0][j]=tri[i][1]
        edges[1][j]=tri[i][2]
        j=j+1
        edges[0][j]=tri[i][2]
        edges[1][j]=tri[i][1]
        j=j+1

    data=np.ones(6*len(tri))
    A = sp.sparse.csc_matrix((data,(edges[0,:],edges[1,:])))

    for i in range(A.nnz):
        A.data[i] = 1.0

    G = nx.to_networkx_graph(A)

    # End of commented out code...
    plot_graph(G,pos,2)

    # Get a random partition
    W0, W1, W2 = random_partition(G)
    count_edge_cuts(G, W0, W1, W2)

    # Use the eigenvectors of the normalised Laplacian to calculate placement positions
    # for the nodes in the graph



    #   eigen_pos holds the positions
    eigen_pos = dict()
    deg = A.sum(0)
    diags = np.array([0])
    D = sp.sparse.spdiags(deg,diags,A.shape[0],A.shape[1])
    Dinv = sp.sparse.spdiags(1/deg,diags,A.shape[0],A.shape[1])
    # Normalised laplacian
    L = Dinv*(D - A)
    E, V= sp.sparse.linalg.eigs(L,3,None,100.0,'SM')
    V = V.real

    for i in range(num_nodes):
        eigen_pos[i] = V[i,1].real,V[i,2].real


    # for n,nbrsdict in G.adjacency_iter():
    #     for nbr,eattr in nbrsdict.items():
    #         if 'weight' in eattr:
    #             print n,nbr,eattr['weight']

    plot_graph(G,eigen_pos,3)


    # Now let's see if the eigenvectors are good for clustering
    # Use kemans to cluster the points in the vector V

    features = np.column_stack((V[:,1], V[:,2]))
    cluster_nodes(G,features,pos,eigen_pos)

    # Finally, use the columns of A directly for clustering
    cluster_nodes(G,A.todense(),pos,eigen_pos)

    input("Press Enter to Continue ...")


def plot_graph(G,pos,fignum):

    label = dict()
    labelpos=dict()
    for i in range(G.number_of_nodes()):
        label[i] = i
        labelpos[i] = pos[i][0]+0.02, pos[i][1]+0.02


    fig=plt.figure(fignum,figsize=(8,8))
    fig.clf()
    nx.draw_networkx_nodes(G,
                            pos,
                            node_size=40,
                            hold=False,
                        )

    nx.draw_networkx_edges(G,pos, hold=True)
    nx.draw_networkx_labels(G,
                            labelpos,
                            label,
                            font_size=10,
                            hold=True,
                        )
    fig.show()


def count_edge_cuts(G, W0, W1, W2):
    edge_cut_count=0
    edge_uncut_count=0
    for edge in G.edges_iter():
        # This may be inefficient but I'll just check if both nodes are in 0, 1, or two
        if edge[0] in W0 and edge[1] in W0:
            edge_uncut_count += 1
        elif edge[0] in W1 and edge[1] in W1:
            edge_uncut_count += 1
        elif edge[0] in W0 and edge[1] in W0:
            edge_uncut_count += 1
        else:
            edge_cut_count += 1
    print('Edge cuts: ', edge_cut_count)
    print('Community edges: ', edge_uncut_count)
    return edge_cut_count, edge_uncut_count


def random_partition(G):
    W0=[]
    W1=[]
    W2=[]
    for node in G.nodes_iter():
        set = random.randint(0,2)
        if set == 0:
            W0.append(node)
        if set == 1:
            W1.append(node)
        if set == 2:
            W2.append(node)
    return W0, W1, W2


def cluster_nodes(G, feat, pos, eigen_pos):
    book,distortion = kmeans(feat,3)
    codes,distortion = vq(feat, book)

    nodes = np.array(range(G.number_of_nodes()))
    W0 = nodes[codes==0].tolist()
    W1 = nodes[codes==1].tolist()
    W2 = nodes[codes==2].tolist()
    print("W0 ", W0)
    print("W1 ", W1)
    print("W2 ", W2)
    count_edge_cuts(G, W0, W1, W2)

    plt.figure(3)
    nx.draw_networkx_nodes(G,
                           eigen_pos,
                           node_size=40,
                           hold=True,
                           nodelist=W0,
                           node_color='m'
                        )
    nx.draw_networkx_nodes(G,
                           eigen_pos,
                           node_size=40,
                           hold=True,
                           nodelist=W1,
                           node_color='b'
                        )
    plt.figure(2)
    nx.draw_networkx_nodes(G,
                           pos,
                           node_size=40,
                           hold=True,
                           nodelist=W0,
                           node_color='m'
                        )
    nx.draw_networkx_nodes(G,
                           pos,
                           node_size=40,
                           hold=True,
                           nodelist=W1,
                           node_color='b'
                        )

# Run an example.
placement()
