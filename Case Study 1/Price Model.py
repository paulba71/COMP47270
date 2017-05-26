import networkx as nx
import random
import HelperMethods as helpers
import cProfile

# average citation count
c=50
# number of nodes to grow to
# n = 500
# n = 5000
n = 1000
# n = 50000
# factor to use the pref attachment out of 10
a = 2  # 20%
# a = 5  # 50%
# a = 10  # 100%


def map_random_number_to_node(n, degrees):
    offset = 0
    index = 0
    random_chance = random.randint(1, 10)
    if random_chance <= a:
        return random.randint(0, len(degrees)-1)

    while n > offset:
        offset += degrees[index]
        index += 1
    return index


def add_citations(graph, i, degrees):
    if i <= c:
        # initial nodes just cite each other
        for j in range(i):
            graph.add_edge(i, j)
            degrees[j] += 1
            # print("Edge added from {} to {}", i, j)
    else:
        for j in range(c):
            # Pick a random number up to the number of citations or graph edges
            total=graph.number_of_edges()
            random_no = random.randint(0,total)
            node_to_cite=map_random_number_to_node(random_no, degrees)
            graph.add_edge(i,node_to_cite)
            degrees[node_to_cite] += 1
            # print("Edge added from {} to {}", i, node_to_cite)


def create_price_model_graph():
    # local count of degrees
    degrees = {}
    graph = nx.DiGraph()
    # Add the first node
    graph.add_node(0)
    # iterate until n adding nodes
    for i in range(n):
        print(i)
        degrees[i] = 0
        graph.add_node(i)
        add_citations(graph, i, degrees)
    return graph


def print_price_model_graph_as_text(G):
    for node in G.edges_iter():
        print(node)


def run_price_model_sim():
    full_graph = create_price_model_graph()

    print_price_model_graph_as_text(full_graph)
    helpers.plot_degseq(full_graph, False)
    input("Press enter to finish.")

run_price_model_sim()
