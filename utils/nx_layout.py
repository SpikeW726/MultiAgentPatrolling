import networkx as nx
import numpy as np
from utils.graph_utils import Graph

def create_nx_layout(map_graph: Graph):
    """
    使用 networkx 的 kamada_kawai_layout 算法创建高质量的节点布局。
    该实现确保了以下两点：
    1. 真实性: 明确使用边的'weight'作为距离度量，使得边的视觉长度与其权重成正比。
    2. 清晰性: 根据节点数量动态缩放整个布局，从根本上解决节点拥挤问题。

    Args:
        map_graph: 自定义的 Graph 对象

    Returns:
        dict: 节点ID到(x, y)坐标的映射
    """
    # 1. 创建一个 networkx 图对象，并正确地添加带权重的边
    G = nx.Graph()
    for node in map_graph.nodes:
        G.add_node(node)
    
    for node, neighbors in map_graph.adj_list.items():
        for neighbor, weight in neighbors:
            if node < neighbor: # 避免重复添加
                G.add_edge(node, neighbor, weight=weight)
            
    # 2. 计算一个更温和、非线性的动态缩放因子
    #    这可以确保布局随节点数适度增长，而不会增长得过快
    #    从而让visualize_utils.py中的视觉缩放能够生效。
    num_nodes = len(map_graph.nodes)
    scale_factor = 5.0 + np.sqrt(num_nodes)
    
    # 3. 使用 kamada_kawai_layout 计算最终布局
    #    - weight='weight': 关键参数，强制算法将边的权重作为它们的目标长度。
    #    - scale=scale_factor: 将最终生成的坐标进行等比例放大，解决拥挤问题。
    pos = nx.kamada_kawai_layout(G, weight='weight', scale=scale_factor)
    
    return pos 