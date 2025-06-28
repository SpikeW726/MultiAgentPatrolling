import networkx as nx
from utils.graph_utils import Graph

def create_nx_layout(map_graph: Graph):
    """
    使用 networkx 的 kamada_kawai_layout 算法创建节点布局。
    该算法会考虑边的权重，尝试使节点间的几何距离与图论距离成正比。

    Args:
        map_graph: 自定义的 Graph 对象

    Returns:
        dict: 节点ID到(x, y)坐标的映射
    """
    # 1. 创建一个 networkx 图对象
    G = nx.Graph()

    # 2. 从 map_graph 添加节点和带权重的边
    for node in map_graph.nodes:
        G.add_node(node)
    
    for node, neighbors in map_graph.adj_list.items():
        for neighbor, weight in neighbors:
            # networkx的spring_layout等将'weight'解释为更强的吸引力（更短的边）
            # kamada_kawai_layout将'weight'解释为距离，所以权重越大距离越远
            # 这符合我们的直觉
            G.add_edge(node, neighbor, weight=weight)
            
    # 3. 使用 kamada_kawai_layout 计算布局
    # dist参数指定了节点间距离的计算方式，我们直接用边的权重
    # scale参数可以放大整个布局，以获得更好的视觉间距
    pos = nx.kamada_kawai_layout(G, dist=None, weight='weight', scale=2.0)
    
    return pos 