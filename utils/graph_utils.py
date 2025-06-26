# load the topology G=(V,E,W,\phi) from graph_topology.json
# construct the graph in mathematical form

import json
from typing import Dict, List, Tuple

class Graph:
    def __init__(self, path:str):
        self.nodes, self.adj_list, self.phi = load_graph(path)
    
    # 后续还可以再扩展一些处理图的函数

def load_graph(path: str):
    """
    Load a graph from JSON file.

    Args:
        path (str): path to the JSON file

    Returns:
        nodes (List[int]): list of node ids
        adj_list (Dict[int, List[Tuple[int, int]]]): adjacency list with (neighbor, edge weight)
        phi (Dict[int, int]): importance weight for each node
    """
    with open(path, "r") as f:
        data = json.load(f)

    nodes: List[int] = data["nodes"]
    edges: List[Dict] = data["edges"]
    phi: Dict[int, int] = {int(k): int(v) for k, v in data["phi"].items()}

    # Initialize adjacency list
    adj_list: Dict[int, List[Tuple[int, int]]] = {node: [] for node in nodes}

    # Populate edges
    for edge in edges:
        src = edge["from"]
        dst = edge["to"]
        weight = edge["weight"]
        adj_list[src].append((dst, weight))

    return nodes, adj_list, phi