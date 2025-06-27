import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from utils.graph_utils import Graph

def evaluate_and_visualize(agents, map_graph:Graph, algorithm_name, evaluation_steps=500):
    """
    运行训练好的agents并生成可视化
    
    Args:
        agents: 训练好的agent列表
        map_graph: 地图图结构
        algorithm_name: 算法名称，用于文件名和标题
        evaluation_steps: 评估步数
    """
    # 初始化
    node_idleness = {n: 0 for n in map_graph.nodes}
    normalization = len(agents) / len(map_graph.nodes)
    
    # 记录数据
    graph_idleness_history = [] # record the Graph-Idleness(node average Node-Idleness) of each time-step
    avg_idleness_history = [] # record the Average-Idleness(time average Graph-Idleness) of each time-step
    worst_idleness_history = [] # record the max Node-Idleness of each time-step
    
    # 为动画准备数据 - 增加子步数实现连续运动
    agent_positions_history = []
    sub_steps_per_step = 3  # 减少到3个子步，提高速度
    
    print(f"Starting {algorithm_name} evaluation...")
    
    for step in range(evaluation_steps):
        # 更新所有节点idleness
        for n in node_idleness:
            node_idleness[n] += 1
            
        # 记录当前step的idleness
        current_graph_idleness = sum(node_idleness.values()) * normalization / len(node_idleness)
        graph_idleness_history.append(current_graph_idleness)
        current_avg_idleness = sum(graph_idleness_history) / (step+1)
        avg_idleness_history.append(current_avg_idleness)
        current_worst_idleness = max(node_idleness.values()) * normalization
        worst_idleness_history.append(current_worst_idleness)
        
        # 记录当前agent位置（用于动画）- 连续运动
        for sub_step in range(sub_steps_per_step):
            current_positions = {}
            for agent in agents:
                if agent.on_edge:
                    # 在边上，计算插值位置 - 连续运动
                    # 获取边的总时间（边权）
                    edge_weight = None
                    for n, w in agent.map.adj_list[agent.position]:
                        if n == agent.target_node:
                            edge_weight = w
                            break
                    
                    if edge_weight is not None and edge_weight > 0:
                        # 计算当前时间步的进度
                        step_progress = (edge_weight - agent.edge_time_left) / edge_weight
                        # 添加子步的进度
                        sub_progress = sub_step / sub_steps_per_step
                        # 总进度
                        total_progress = step_progress + sub_progress / edge_weight
                        total_progress = min(total_progress, 1.0)  # 确保不超过1
                    else:
                        total_progress = 1.0
                    current_positions[agent.agent_id] = (agent.position, agent.target_node, total_progress)
                else:
                    # 在节点上
                    current_positions[agent.agent_id] = (agent.position, agent.position, 1.0)
            agent_positions_history.append(current_positions)
        
        # 每个agent行动 - 正确调用step方法
        for agent in agents:
            result = agent.step(node_idleness)
            if result is not None:
                # 到达节点，重置该节点idleness
                node_idleness[result] = 0

                # 添加与训练循环一致的逻辑以防止停顿
                # 1. 获取在新节点上的状态
                current_state = agent.get_state(node_idleness)
                
                # 2. 根据学习到的策略选择下一个动作
                next_action = agent.select_action(current_state)
                
                # 3. 准备立即移动
                edge_weight = agent.map.get_edge_length(agent.position, next_action)
                agent.on_edge = True
                agent.edge_time_left = int(edge_weight) if edge_weight is not None else 0
                agent.target_node = next_action
                
                # 4. 保持智能体状态更新的一致性
                agent.last_state = current_state
                agent.last_action = next_action
    
    # 绘制Average Idleness图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(evaluation_steps), avg_idleness_history, 'b-', linewidth=1)
    plt.xlabel('Time Step')
    plt.ylabel('Average Idleness')
    plt.title(f'{algorithm_name} - Average Idleness over Time')
    plt.grid(True, alpha=0.3)
    
    # 绘制Worst Idleness图
    plt.subplot(1, 2, 2)
    plt.plot(range(evaluation_steps), worst_idleness_history, 'r-', linewidth=1)
    plt.xlabel('Time Step')
    plt.ylabel('Worst Idleness')
    plt.title(f'{algorithm_name} - Worst Idleness over Time')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    idleness_filename = f'{algorithm_name.lower()}_idleness_evaluation.png'
    plt.savefig(idleness_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"{algorithm_name} evaluation completed! Final average idleness: {avg_idleness_history[-1]:.2f}")
    print(f"Final worst idleness: {max(worst_idleness_history):.2f}")
    print(f"Idleness plots saved as '{idleness_filename}'")
    
    create_animation(map_graph, agent_positions_history, evaluation_steps * sub_steps_per_step, algorithm_name)

def create_animation(map_graph, agent_positions_history, total_steps, algorithm_name):
    """
    创建agent移动的动画视频
    
    Args:
        map_graph: 地图图结构
        agent_positions_history: agent位置历史记录
        total_steps: 总步数
        algorithm_name: 算法名称，用于文件名和标题
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制地图 - 使用混合布局，平衡美观和真实性
    node_positions = create_balanced_layout(map_graph)
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
        ax.plot(pos[0], pos[1], 'o', markersize=25, color='lightblue', markeredgecolor='black', linewidth=2)  # type: ignore
        ax.text(pos[0], pos[1], str(node), ha='center', va='center', fontsize=14, fontweight='bold')  # type: ignore
    
    # 绘制边权标签（中间层）- 智能位置调整
    label_positions = calculate_label_positions(map_graph, node_positions)
    for (node, neighbor), (label_x, label_y) in label_positions.items():
        weight = None
        for n, w in map_graph.adj_list[node]:
            if n == neighbor:
                weight = w
                break
        if weight is not None:
            ax.text(label_x, label_y, str(weight), ha='center', va='center',   # type: ignore
                   fontsize=11, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", 
                   facecolor='yellow', alpha=0.9, edgecolor='black', linewidth=1))
    
    # 设置图形范围 - 根据实际节点位置动态调整
    x_coords = [pos[0] for pos in node_positions.values()]
    y_coords = [pos[1] for pos in node_positions.values()]
    x_margin = (max(x_coords) - min(x_coords)) * 0.1
    y_margin = (max(y_coords) - min(y_coords)) * 0.1
    
    ax.set_xlim(min(x_coords) - x_margin, max(x_coords) + x_margin)  # type: ignore
    ax.set_ylim(min(y_coords) - y_margin, max(y_coords) + y_margin)  # type: ignore
    ax.set_aspect('equal')  # type: ignore
    ax.set_title(f'{algorithm_name} Agent Movement Animation', fontsize=16, fontweight='bold')  # type: ignore
    
    # 移除坐标轴
    ax.set_xticks([])  # type: ignore
    ax.set_yticks([])  # type: ignore
    ax.spines['top'].set_visible(False)  # type: ignore
    ax.spines['right'].set_visible(False)  # type: ignore
    ax.spines['bottom'].set_visible(False)  # type: ignore
    ax.spines['left'].set_visible(False)  # type: ignore
    
    # 创建agent标记（最上层）- 使用zorder确保在最上层
    agent_markers = []
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    for i in range(len(agent_positions_history[0])):
        marker, = ax.plot([], [], '^', markersize=20, color=colors[i % len(colors)],   # type: ignore
                         markeredgecolor='black', linewidth=3, label=f'Agent {i}', zorder=10)
        agent_markers.append(marker)
    
    ax.legend(fontsize=12)  # type: ignore
    
    def animate(frame):
        if frame >= len(agent_positions_history):
            return agent_markers
        
        current_positions = agent_positions_history[frame]
        
        for i, (agent_id, (start_node, end_node, progress)) in enumerate(current_positions.items()):
            if i < len(agent_markers):
                if start_node == end_node:
                    # 在节点上
                    pos = node_positions[start_node]
                    agent_markers[i].set_data([pos[0]], [pos[1]])
                else:
                    # 在边上，插值位置
                    start_pos = node_positions[start_node]
                    end_pos = node_positions[end_node]
                    current_x = start_pos[0] + progress * (end_pos[0] - start_pos[0])
                    current_y = start_pos[1] + progress * (end_pos[1] - start_pos[1])
                    agent_markers[i].set_data([current_x], [current_y])
        
        return agent_markers
    
    # 创建动画 - 调整更新频率，让移动更快
    anim = animation.FuncAnimation(fig, animate, frames=min(total_steps, len(agent_positions_history)), 
                                 interval=300, blit=True, repeat=True)  # 减少interval到300ms
    
    # 保存动画
    animation_filename = f'{algorithm_name.lower()}_animation.gif'
    anim.save(animation_filename, writer='pillow', fps=3)  # 增加fps到3
    plt.show()
    
    print(f"Animation saved as '{animation_filename}'")

def calculate_label_positions(map_graph, node_positions):
    """
    计算边权标签的位置，保持在边上
    
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
    
    # 按权重排序，权重大的边优先放置标签
    edges.sort(key=lambda x: x[2], reverse=True)
    
    for node, neighbor, weight in edges:
        pos1 = node_positions[node]
        pos2 = node_positions[neighbor]
        
        # 简单使用中点位置
        mid_x = (pos1[0] + pos2[0]) / 2
        mid_y = (pos1[1] + pos2[1]) / 2
        
        label_positions[(node, neighbor)] = (mid_x, mid_y)
    
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
    
    Args:
        map_graph: 地图图结构
        max_iterations: 最大迭代次数
    
    Returns:
        dict: 节点ID到(x, y)坐标的映射
    """
    # 初始化节点位置（圆周分布）
    node_positions = {}
    num_nodes = len(map_graph.nodes)
    radius = 4.5  # 适中的半径
    
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
        forces = {node: np.array([0.0, 0.0]) for node in map_graph.nodes}
        
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
                        target_distance = 2.5 + normalized_weight * 2.5  # 2.5到5.0之间
                    else:
                        target_distance = 3.5
                    
                    # 计算弹簧力
                    if current_distance > 0:
                        force_magnitude = (current_distance - target_distance) * 0.08
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
                if distance > 0 and distance < 2.2:  # 如果太近
                    force_magnitude = 1.5 / (distance + 0.1)
                    force_direction = (pos1 - pos2) / distance
                    force = force_magnitude * force_direction
                    
                    forces[node1] += force
                    forces[node2] -= force
        
        # 向心力（保持节点在合理范围内）
        center = np.array([0.0, 0.0])
        for node in map_graph.nodes:
            pos = np.array(node_positions[node])
            distance_to_center = np.linalg.norm(pos - center)
            if distance_to_center > 6.5:  # 如果太远
                force_magnitude = (distance_to_center - 6.5) * 0.1
                force_direction = (center - pos) / distance_to_center
                force = force_magnitude * force_direction
                forces[node] += force
        
        # 更新节点位置
        for node in map_graph.nodes:
            force = forces[node]
            # 限制力的大小
            force_magnitude = np.linalg.norm(force)
            if force_magnitude > 0.25:
                force = force * 0.25 / force_magnitude
            
            node_positions[node] = (
                node_positions[node][0] + force[0],
                node_positions[node][1] + force[1]
            )
    
    return node_positions