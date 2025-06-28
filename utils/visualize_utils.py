import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from utils.graph_utils import Graph
from typing import Dict, List
from matplotlib.colors import to_rgb
from utils.nx_layout import create_nx_layout # 导入新的布局函数


def plot_idleness_charts(avg_idleness_history, worst_idleness_history, algorithm_name, map_name):
    """
    根据传入的idleness历史数据绘制折线图
    Args:
        avg_idleness_history (list): 每一步的平均空闲度历史
        worst_idleness_history (list): 每一步的最差空闲度历史
        algorithm_name (str): 算法名称
        map_name (str): 地图名称
    """
    evaluation_steps = len(avg_idleness_history)
    # 绘制Average Idleness图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(evaluation_steps), avg_idleness_history, 'b-', linewidth=1)
    plt.xlabel('Time Step')
    plt.ylabel('Average Idleness')
    plt.title(f'{algorithm_name} on {map_name} - Average Idleness')
    plt.grid(True, alpha=0.3)
    
    # 绘制Worst Idleness图
    plt.subplot(1, 2, 2)
    plt.plot(range(evaluation_steps), worst_idleness_history, 'r-', linewidth=1)
    plt.xlabel('Time Step')
    plt.ylabel('Worst Idleness')
    plt.title(f'{algorithm_name} on {map_name} - Worst Idleness')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    folder_name = f"{algorithm_name}_results"
    os.makedirs(folder_name, exist_ok=True)
    idleness_filename = os.path.join(folder_name, f"{algorithm_name}_idleness_eval_{map_name}.png")
    plt.savefig(idleness_filename, dpi=300, bbox_inches='tight')
    plt.close() # 关闭图像，防止与动画窗口重叠
    
    print(f"Idleness plots saved as '{idleness_filename}'")


def create_animation(map_graph, agent_positions_history, total_frames, algorithm_name, map_name):
    """
    创建agent移动的动画视频
    Args:
        map_graph: 地图图结构
        agent_positions_history: agent位置历史记录
        total_frames: 总步数
        algorithm_name: 算法名称，用于文件名和标题
        map_name: 地图名称，用于文件名和标题
    """
    print("Starting animation...")
    
    num_nodes = len(map_graph.nodes)
    # Scale figure size based on number of nodes to prevent crowding
    figure_scale_factor = 1.0 + num_nodes / 30.0

    # 创建图形
    fig, ax = plt.subplots(figsize=(10 * figure_scale_factor, 8 * figure_scale_factor))
    
    # 绘制地图 - 使用 networkx 的专业布局算法
    node_positions = create_nx_layout(map_graph)
    # node_positions = create_balanced_layout(map_graph) # 注释掉旧的布局函数
    # node_positions = create_circular_layout(map_graph)
    # node_positions = create_improved_layout(map_graph)
    
    # 绘制边（先绘制，在底层）
    for node in map_graph.nodes:
        for neighbor, weight in map_graph.adj_list[node]:
            if node < neighbor:  # 避免重复绘制
                pos1 = node_positions[node]
                pos2 = node_positions[neighbor]
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'k-', alpha=0.6, linewidth=2)  # type: ignore
    
    # 绘制节点（中间层）
    for node, pos in node_positions.items():
        ax.plot(pos[0], pos[1], 'o', markersize=35, color='skyblue', markeredgecolor='black', linewidth=2)  # type: ignore
        ax.text(pos[0], pos[1], str(node), ha='center', va='center', fontsize=16, fontweight='bold')  # type: ignore
    
    # 绘制边权标签（中间层）- 智能位置调整
    label_positions = calculate_label_positions(map_graph, node_positions)
    for (node, neighbor), (label_x, label_y) in label_positions.items():
        weight = map_graph.get_edge_length(node, neighbor)
        if weight is not None:
            ax.text(label_x, label_y, str(weight), ha='center', va='center',   # type: ignore
                   fontsize=10, fontweight='normal', bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=0.5))
    
    # 设置图形范围 - 根据实际节点位置动态调整
    x_coords = [pos[0] for pos in node_positions.values()]
    y_coords = [pos[1] for pos in node_positions.values()]
    x_margin = (max(x_coords) - min(x_coords)) * 0.1
    y_margin = (max(y_coords) - min(y_coords)) * 0.1
    
    ax.set_xlim(min(x_coords) - x_margin, max(x_coords) + x_margin)  # type: ignore
    ax.set_ylim(min(y_coords) - y_margin, max(y_coords) + y_margin)  # type: ignore
    ax.set_aspect('equal')  # type: ignore
    ax.set_title(f'{algorithm_name} on {map_name} - Agent Movement', fontsize=16, fontweight='bold')  # type: ignore
    
    # 移除坐标轴
    ax.set_xticks([])  # type: ignore
    ax.set_yticks([])  # type: ignore
    ax.spines['top'].set_visible(False)  # type: ignore
    ax.spines['right'].set_visible(False)  # type: ignore
    ax.spines['bottom'].set_visible(False)  # type: ignore
    ax.spines['left'].set_visible(False)  # type: ignore
    
    # 创建agent标记（最上层）- 使用zorder确保在最上层
    agent_markers = []
    agent_labels = []
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'yellow']
    
    # 转换颜色名称为RGB元组，以便计算亮度
    rgb_colors = [to_rgb(c) for c in colors]

    for i in range(len(agent_positions_history[0])):
        bg_color = rgb_colors[i % len(rgb_colors)]
        text_color = get_text_color_for_bg(bg_color)

        marker, = ax.plot([], [], '^', markersize=28, color=bg_color,
                         markeredgecolor='black', linewidth=2, label=f'Agent {i}', zorder=10)
        label = ax.text(0, 0, str(i), ha='center', va='center', color=text_color, fontsize=10, fontweight='bold', zorder=11)
        agent_markers.append(marker)
        agent_labels.append(label)
    
    ax.legend(fontsize=12, labelspacing=1.5)  # type: ignore
    
    def animate(frame):
        if frame >= len(agent_positions_history):
            return (*agent_markers, *agent_labels)
        
        current_positions = agent_positions_history[frame]
        
        for i, (agent_id, (start_node, end_node, progress)) in enumerate(current_positions.items()):
            if i < len(agent_markers):
                if start_node == end_node:
                    # 在节点上
                    pos = node_positions[start_node]
                    agent_markers[i].set_data([pos[0]], [pos[1]])
                    agent_labels[i].set_position((pos[0], pos[1]))
                else:
                    # 在边上，插值位置
                    start_pos = node_positions[start_node]
                    end_pos = node_positions[end_node]
                    current_x = start_pos[0] + progress * (end_pos[0] - start_pos[0])
                    current_y = start_pos[1] + progress * (end_pos[1] - start_pos[1])
                    agent_markers[i].set_data([current_x], [current_y])
                    agent_labels[i].set_position((current_x, current_y))
        
        return (*agent_markers, *agent_labels)
    
    # 创建动画 - 移除blit=True以支持文本动画，调整interval让动画变慢
    anim = animation.FuncAnimation(fig, animate, frames=min(total_frames, len(agent_positions_history)), 
                                 interval=90, blit=False, repeat=True)
    
    # 保存动画
    folder_name = f"{algorithm_name}_results"
    os.makedirs(folder_name, exist_ok=True)
    animation_filename = os.path.join(folder_name, f"{algorithm_name}_animation_{map_name}.gif")
    anim.save(animation_filename, writer='pillow', fps=12, dpi=80) # 添加dpi参数以解决quantization error
    plt.close() # 关闭动画窗口
    
    print(f"Animation saved as '{animation_filename}'")


def get_text_color_for_bg(bg_color_rgb):
    """
    根据背景色的亮度决定使用黑色或白色文本以获得最佳对比度。
    Args:
        bg_color_rgb (tuple): 背景色的RGB元组, e.g., (1, 0, 0) for red.
    Returns:
        str: 'white' or 'black'.
    """
    # 计算颜色的感知亮度 (perceived luminance)
    # 公式: Y = 0.299*R + 0.587*G + 0.114*B
    luminance = 0.299 * bg_color_rgb[0] + 0.587 * bg_color_rgb[1] + 0.114 * bg_color_rgb[2]
    
    # 如果亮度大于0.5，背景是亮的，使用黑色文字；否则使用白色文字。
    return 'black' if luminance > 0.5 else 'white'


def calculate_label_positions(map_graph, node_positions):
    """
    计算边权标签的位置，使其保持在边上但避免重叠
    - 标签交替地放置在靠近边的一个端点的位置
      而不是都放在中点，以避免重叠
    
    Args:
        map_graph: 地图图结构
        node_positions: 节点位置字典
    
    Returns:
        dict: (node, neighbor) -> (x, y) 标签位置映射
    """
    label_positions = {}
    
    # 收集所有边
    edges = []
    for node in map_graph.nodes:
        for neighbor, weight in map_graph.adj_list[node]:
            if node < neighbor:  # 避免重复
                edges.append((node, neighbor, weight))
    
    # 按节点对排序，以获得一致的交替效果
    edges.sort() 
    
    for i, (node, neighbor, weight) in enumerate(edges):
        pos1 = np.array(node_positions[node])
        pos2 = np.array(node_positions[neighbor])
        
        edge_vector = pos2 - pos1
        
        # 交替标签位置，一个靠近起点，一个靠近终点，避免全部挤在中间
        if i % 2 == 0:
            # 放在离起点35%的位置
            position_ratio = 0.35
        else:
            # 放在离起点65%的位置
            position_ratio = 0.65
            
        final_label_pos = pos1 + edge_vector * position_ratio
        
        label_positions[(node, neighbor)] = (final_label_pos[0], final_label_pos[1])
        
    return label_positions

def create_circular_layout(map_graph):
    """
    创建圆形布局 - 节点排在外围一圈，减少边穿过节点
    
    Args:
        map_graph: 地图图结构
    
    Returns:
        dict: 节点ID到(x, y)坐标的映射
    """
    # 初始化节点位置（圆周分布）
    node_positions = {}
    num_nodes = len(map_graph.nodes)
    radius = 5.0  # 增大半径，让节点分布更分散
    
    # 计算每个节点的度数，度数大的节点优先分配更好的位置
    node_degrees = {}
    for node in map_graph.nodes:
        node_degrees[node] = len(map_graph.adj_list[node])
    
    # 按度数排序，度数大的节点优先选择位置
    sorted_nodes = sorted(map_graph.nodes, key=lambda x: node_degrees[x], reverse=True)
    
    # 分配位置
    for i, node in enumerate(sorted_nodes):
        angle = 2 * np.pi * i / num_nodes
        # 添加一些随机偏移，避免完全对称
        angle_offset = (np.random.random() - 0.5) * 0.1
        radius_offset = (np.random.random() - 0.5) * 0.3
        node_positions[node] = (
            (radius + radius_offset) * np.cos(angle + angle_offset),
            (radius + radius_offset) * np.sin(angle + angle_offset)
        )
    
    # 简单的力导向优化，主要优化边不穿过节点
    for iteration in range(50):
        forces = {node: np.array([0.0, 0.0]) for node in map_graph.nodes}
        
        # 排斥力（防止节点重叠）
        for i, node1 in enumerate(map_graph.nodes):
            for node2 in list(map_graph.nodes)[i+1:]:
                pos1 = np.array(node_positions[node1])
                pos2 = np.array(node_positions[node2])
                
                distance = np.linalg.norm(pos1 - pos2)
                if distance > 0 and distance < 2.0:  # 如果太近
                    force_magnitude = 2.0 / (distance + 0.1)
                    force_direction = (pos1 - pos2) / distance
                    force = force_magnitude * force_direction
                    
                    forces[node1] += force
                    forces[node2] -= force
        
        # 向心力（保持节点在合理范围内）
        center = np.array([0.0, 0.0])
        for node in map_graph.nodes:
            pos = np.array(node_positions[node])
            distance_to_center = np.linalg.norm(pos - center)
            if distance_to_center > 7.0:  # 如果太远
                force_magnitude = (distance_to_center - 7.0) * 0.1
                force_direction = (center - pos) / distance_to_center
                force = force_magnitude * force_direction
                forces[node] += force
        
        # 更新节点位置
        for node in map_graph.nodes:
            force = forces[node]
            # 限制力的大小
            force_magnitude = np.linalg.norm(force)
            if force_magnitude > 0.2:
                force = force * 0.2 / force_magnitude
            
            node_positions[node] = (
                node_positions[node][0] + force[0],
                node_positions[node][1] + force[1]
            )
    
    return node_positions

def create_improved_layout(map_graph, max_iterations=200):
    """
    创建改进的节点布局 - 基于边权的力导向布局
    
    Args:
        map_graph: 地图图结构
        max_iterations: 最大迭代次数
    
    Returns:
        dict: 节点ID到(x, y)坐标的映射
    """
    # 初始化节点位置（随机分布）
    node_positions = {}
    for i, node in enumerate(map_graph.nodes):
        angle = 2 * np.pi * i / len(map_graph.nodes)
        radius = 3 + np.random.random() * 2  # 添加一些随机性
        node_positions[node] = (radius * np.cos(angle), radius * np.sin(angle))
    
    # 计算所有边的权重范围
    all_weights = []
    for node in map_graph.nodes:
        for neighbor, weight in map_graph.adj_list[node]:
            if node < neighbor:  # 避免重复
                all_weights.append(weight)
    
    if not all_weights:
        return node_positions
    
    min_weight = min(all_weights)
    max_weight = max(all_weights)
    weight_range = max_weight - min_weight
    
    # 力导向布局迭代
    for iteration in range(max_iterations):
        # 计算每个节点受到的力
        forces = {node: np.array([0.0, 0.0]) for node in map_graph.nodes}
        
        # 弹簧力（基于边权）
        for node in map_graph.nodes:
            for neighbor, weight in map_graph.adj_list[node]:
                if node < neighbor:  # 避免重复计算
                    pos1 = np.array(node_positions[node])
                    pos2 = np.array(node_positions[neighbor])
                    
                    # 计算当前距离
                    current_distance = np.linalg.norm(pos1 - pos2)
                    
                    # 目标距离基于权重（权重越大，目标距离越远）
                    if weight_range > 0:
                        normalized_weight = (weight - min_weight) / weight_range
                        target_distance = 1.0 + normalized_weight * 3.0  # 1.0到4.0之间
                    else:
                        target_distance = 2.0
                    
                    # 计算弹簧力
                    if current_distance > 0:
                        force_magnitude = (current_distance - target_distance) * 0.1
                        force_direction = (pos2 - pos1) / current_distance
                        force = force_magnitude * force_direction
                        
                        forces[node] += force
                        forces[neighbor] -= force
        
        # 排斥力（防止节点重叠）
        for i, node1 in enumerate(map_graph.nodes):
            for node2 in list(map_graph.nodes)[i+1:]:
                pos1 = np.array(node_positions[node1])
                pos2 = np.array(node_positions[node2])
                
                distance = np.linalg.norm(pos1 - pos2)
                if distance > 0 and distance < 1.0:  # 如果太近
                    force_magnitude = 0.5 / (distance + 0.1)
                    force_direction = (pos1 - pos2) / distance
                    force = force_magnitude * force_direction
                    
                    forces[node1] += force
                    forces[node2] -= force
        
        # 更新节点位置
        for node in map_graph.nodes:
            force = forces[node]
            # 限制力的大小
            force_magnitude = np.linalg.norm(force)
            if force_magnitude > 0.5:
                force = force * 0.5 / force_magnitude
            
            node_positions[node] = (
                node_positions[node][0] + force[0],
                node_positions[node][1] + force[1]
            )
    
    return node_positions

def create_balanced_layout(map_graph, max_iterations=150):
    """
    创建平衡布局 - 兼顾美观和边权真实性
    此版本会根据节点数量自动缩放，以避免拥挤
    
    Args:
        map_graph: 地图图结构
        max_iterations: 最大迭代次数
    
    Returns:
        dict: 节点ID到(x, y)坐标的映射
    """
    num_nodes = len(map_graph.nodes)
    
    # 根据节点数定义缩放因子，节点越多，布局越扩展
    scale_factor = 1.0 + num_nodes / 25.0

    # 初始化节点位置（圆周分布）
    node_positions = {}
    radius = 3.0 * scale_factor
    
    # 计算每个节点的度数，度数大的节点优先分配更好的位置
    node_degrees = {}
    for node in map_graph.nodes:
        node_degrees[node] = len(map_graph.adj_list[node])
    
    # 按度数排序，度数大的节点优先选择位置
    sorted_nodes = sorted(map_graph.nodes, key=lambda x: node_degrees[x], reverse=True)
    
    # 分配位置
    for i, node in enumerate(sorted_nodes):
        angle = 2 * np.pi * i / num_nodes
        # 添加一些随机偏移，避免完全对称
        angle_offset = (np.random.random() - 0.5) * 0.15
        radius_offset = (np.random.random() - 0.5) * 0.4
        node_positions[node] = (
            (radius + radius_offset) * np.cos(angle + angle_offset),
            (radius + radius_offset) * np.sin(angle + angle_offset)
        )
    
    # 计算所有边的权重范围
    all_weights = []
    for node in map_graph.nodes:
        for neighbor, weight in map_graph.adj_list[node]:
            if node < neighbor:  # 避免重复
                all_weights.append(weight)
    
    if not all_weights:
        return node_positions
    
    min_weight = min(all_weights)
    max_weight = max(all_weights)
    weight_range = max_weight - min_weight
    
    # 力导向布局迭代 - 平衡美观和真实性
    for iteration in range(max_iterations):
        forces: Dict[int, List[float]] = {node: [0.0, 0.0] for node in map_graph.nodes}
        
        # 弹簧力（基于边权）- 权重越大，目标距离越远，但范围适中
        for node in map_graph.nodes:
            for neighbor, weight in map_graph.adj_list[node]:
                if node < neighbor:  # 避免重复计算
                    pos1 = np.array(node_positions[node])
                    pos2 = np.array(node_positions[neighbor])
                    
                    # 计算当前距离
                    current_distance = np.linalg.norm(pos1 - pos2)
                    
                    # 目标距离基于权重，但范围更合理
                    if weight_range > 0:
                        normalized_weight = (weight - min_weight) / weight_range
                        target_distance = (1.5 * scale_factor) + normalized_weight * (1.5 * scale_factor)
                    else:
                        target_distance = 2.5 * scale_factor
                    
                    # 计算弹簧力
                    if current_distance > 0:
                        force_magnitude = (current_distance - target_distance) * 0.08
                        force_direction = (pos2 - pos1) / current_distance
                        force = force_magnitude * force_direction
                        
                        forces[node][0] += force[0]
                        forces[node][1] += force[1]
                        forces[neighbor][0] -= force[0]
                        forces[neighbor][1] -= force[1]
        
        # 排斥力（防止节点重叠）
        for i, node1 in enumerate(map_graph.nodes):
            for node2 in list(map_graph.nodes)[i+1:]:
                pos1 = np.array(node_positions[node1])
                pos2 = np.array(node_positions[node2])
                
                distance = np.linalg.norm(pos1 - pos2)
                # 调整排斥力距离阈值
                if distance > 0 and distance < (1.8 * scale_factor):
                    force_magnitude = 1.5 / (distance + 0.1)
                    force_direction = (pos1 - pos2) / distance
                    force = force_magnitude * force_direction
                    
                    forces[node1][0] += force[0]
                    forces[node1][1] += force[1]
                    forces[node2][0] -= force[0]
                    forces[node2][1] -= force[1]
        
        # 向心力（保持节点在合理范围内）
        center = np.array([0.0, 0.0])
        for node in map_graph.nodes:
            pos = np.array(node_positions[node])
            distance_to_center = np.linalg.norm(pos - center)
            # 调整向心力距离阈值
            if distance_to_center > (4.0 * scale_factor):
                force_magnitude = (distance_to_center - (4.0 * scale_factor)) * 0.1
                force_direction = (center - pos) / distance_to_center
                force = force_magnitude * force_direction
                forces[node][0] += force[0]
                forces[node][1] += force[1]
        
        # 更新节点位置
        for node in map_graph.nodes:
            force = np.array(forces[node])
            # 限制力的大小
            force_magnitude = np.linalg.norm(force)
            if force_magnitude > 0.25:
                force = force * 0.25 / force_magnitude
            
            node_positions[node] = (
                node_positions[node][0] + force[0],
                node_positions[node][1] + force[1]
            )
    
    return node_positions