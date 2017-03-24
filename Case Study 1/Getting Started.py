import networkx as nx
import matplotlib.pyplot as plt
#import display_degseq

facebook = "/Users/paulbarnes/PycharmProjects/SNAP Datasets/facebook_combined.txt"
twitter = "/Users/paulbarnes/PycharmProjects/SNAP Datasets/twitter_combined.txt"
google = "/Users/paulbarnes/PycharmProjects/SNAP Datasets/web-Google.txt"
roads_CA = "/Users/paulbarnes/PycharmProjects/SNAP Datasets/roadNet-CA.txt"


def draw_graph(gr):
    A = nx.to_scipy_sparse_matrix(gr)
    fig = plt.figure()
    plt.spy(A)
    fig.show()

print("---------------------")
print("Facebook - undirected")
print("---------------------")

G = nx.read_edgelist(facebook)
print("Nodes: ", len(G))

# Draw the graph
draw_graph(G)

r = nx.degree_assortativity_coefficient(G)
print("Degree Assortativity Coefficient = ", r)

#n = nx.average_neighbor_degree(G)
#k = nx.k_nearest_neighbors(G)

c = nx.clustering(G)
print("Clustering co-efficient = ", c)
a = nx.average_clustering(G)
print("Average Clustering co-efficient = ", a)
print("")

print("------------------")
print("Twitter - directed")
print("------------------")

#G2 = nx.read_edgelist(twitter,create_using=nx.DiGraph())
G2 = nx.read_edgelist(twitter)
print("Nodes: ",len(G2))

# Draw the graph
draw_graph(G2)

r = nx.degree_assortativity_coefficient(G2)
print("Degree Assortativity Coefficient = ", r)

n = nx.average_neighbor_degree(G2)
k = nx.k_nearest_neighbors(G2)
c = nx.clustering(G2)
print("Clustering co-efficient = ", c)
a = nx.average_clustering(G2)
print("Average Clustering co-efficient = ", a)
print("")



#print("Displaying the chart")
#display_degseq.plot_degseq(G2, False)
#print("Chart done")