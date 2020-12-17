import matplotlib.pyplot as plt
import networkx as nx


def plot_graph(G, name, colors):
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos=pos, node_color=colors,
                           cmap=plt.get_cmap("Paired"))
    nx.draw_networkx_edges(G, pos=pos)
    plt.tight_layout()
    plt.axis("off")
    plt.draw()