import itertools

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
    in_degree = nx.in_degree_centrality(
        net_graph
    )  # calculates in degree centrality of nodes
    out_degree = nx.out_degree_centrality(
        net_graph
    )  # calculated out degree centrality of nodes
    eigenvector = nx.eigenvector_centrality_numpy(
        net_graph, weight="amount"
    )  # calculates eigenvector centrality of nodes
    betweenness = nx.betweenness_centrality(
        net_graph, weight="amount"
    )  # calculates betweenness centrality of nodes

    assortativity = nx.attribute_assortativity_coefficient(
        net_graph, "classification"
    )  # calculates assortativity of graph

    num_nodes = len(net_graph.nodes())
    num_edges = len(net_graph.edges())
    density = num_edges / (
        num_nodes * (num_nodes - 1)
    )  # calculates density of graph

    k = 5
    comp = nx.community.girvan_newman(net_graph)
    for communities in itertools.islice(comp, k):
        communities = tuple(
            sorted(c) for c in communities
        )  # creates clusters of nodes with high interactions where granularity = 5

    with open("network_metrics.txt", "w") as file:
        file.write(f"in degree centrality: {in_degree}\n")
        file.write(f"out degree centrality: {out_degree}\n")
        file.write(f"eigenvector centrality: {eigenvector}\n")
        file.write(f"betweenness centrality: {betweenness}\n\n")

        file.write(f"assortativity based on 'classification': {assortativity}")

        file.write(f"density': {density}")

        file.write(f"communities': {communities}")
