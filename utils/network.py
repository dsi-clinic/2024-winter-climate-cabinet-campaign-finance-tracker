import networkx as nx


def network_metrics(net_graph: nx.Graph) -> None:
    """Given a network graph, return a text files with list of nodes
    with greatest calculated centrality

    Args:
        net_graph: network graph as defined by networkx

    Returns:
        a text file with list of nodes with greatest calculated
        centrality for each metric: in degree, out degree,
        eigenvector, and betweenness

    """
    in_degree = nx.in_degree_centrality(net_graph)
    out_degree = nx.out_degree_centrality(net_graph)
    eigenvector = nx.eigenvector_centrality_numpy(net_graph, weight="amount")
    betweenness = nx.betweenness_centrality(net_graph, weight="amount")

    with open("network_metrics.txt", "w") as file:
        file.write(f"in degree centrality: {in_degree}\n")
        file.write(f"out degree centrality: {out_degree}\n")
        file.write(f"eigenvector centrality: {eigenvector}\n")
        file.write(f"betweenness centrality: {betweenness}")
