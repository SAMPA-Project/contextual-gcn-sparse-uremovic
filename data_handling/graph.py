import networkx as nx
import numpy as np

def load_graph_traffic(path, counters):
    import json
    with open(counters, 'r') as f:
        counters = json.load(f)
    el = nx.read_weighted_edgelist(path)

    # add isolated nodes missing from edgelist
    for node in counters:
        if node not in el:
            el.add_node(node)

    # relabel to 0..N-1 in lexicographic order
    nodes   = sorted(el.nodes())
    mapping = {node_id: i for i, node_id in enumerate(nodes)}
    G       = nx.relabel_nodes(el, mapping)

    edge_weight = np.array([d['weight'] for u, v, d in G.edges(data=True)])
    edge_index  = np.array([[u, v] for u, v in G.edges()]).T

    inv_mapping = {i: node_id for node_id, i in mapping.items()}

    return edge_weight, edge_index#, G, mapping, inv_mapping

# meteo
def load_graph(graph):
    el = nx.read_weighted_edgelist('data/meteo/graph/{}/edgelist.nx'.format(graph))
    return lg(el)

# base logic
def lg(el):
    edge_weight = []
    edge_index = [[], []]
    nodes = []
    for u, edges in el.adjacency():
        nodes.append(u)
        for v in edges.keys():
            if (u < v):
                edge_weight.append(edges[v]['weight'])
                edge_index[0].append(u)
                edge_index[1].append(v)
    edge_weight = np.array(edge_weight)
    edge_index = np.array(edge_index)

    nodes = sorted(nodes)
    mapping = {}
    for i, node_id in enumerate(nodes):
        mapping[node_id] = i

    edge_index = np.vectorize(mapping.get)(edge_index)

    return (edge_weight, edge_index)