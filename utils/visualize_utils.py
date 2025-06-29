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
    # 移除动态画布缩放，使用固定的、合理的尺寸来避免内存溢出
    figure_scale_factor = 1.0 + num_nodes / 30.0

    # --- 新增: 动态视觉元素缩放 ---
    # 使用12个节点的图作为视觉基准
    baseline_nodes = 12.0
    # 为标记点和字体计算缩放因子
    # 缩放是非线性的(sqrt)，以避免在大型图中元素过小
    # 0.6的下限可防止元素变得难以辨认
    visual_scale_factor = max(0.6, np.sqrt(baseline_nodes / num_nodes)) if num_nodes > 0 else 1.0

    # 计算动态大小
    node_markersize = 35 * visual_scale_factor
    node_fontsize = 16 * visual_scale_factor
    agent_markersize = 28 * visual_scale_factor
    agent_fontsize = 10 * visual_scale_factor
    edge_label_fontsize = 10 * visual_scale_factor

    # 创建图形
    fig, ax = plt.subplots(figsize=(12 * figure_scale_factor, 10 * figure_scale_factor))
    # fig, ax = plt.subplots(figsize=(12, 10)) # 使用固定的画布尺寸
    
    # 绘制地图 - 使用 networkx 的专业布局算法
    node_positions = create_nx_layout(map_graph)
    
    # 绘制边（先绘制，在底层）
    for node in map_graph.nodes:
        for neighbor, weight in map_graph.adj_list[node]:
            if node < neighbor:  # 避免重复绘制
                pos1 = node_positions[node]
                pos2 = node_positions[neighbor]
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'k-', alpha=0.6, linewidth=2)  # type: ignore
    
    # 绘制节点（中间层）
    for node, pos in node_positions.items():
        ax.plot(pos[0], pos[1], 'o', markersize=node_markersize, color='skyblue', markeredgecolor='black', linewidth=2)  # type: ignore
        ax.text(pos[0], pos[1], str(node), ha='center', va='center', fontsize=node_fontsize, fontweight='bold')  # type: ignore
    
    # 绘制边权标签（中间层）- 智能位置调整
    label_positions = calculate_label_positions(map_graph, node_positions)
    for (node, neighbor), (label_x, label_y) in label_positions.items():
        weight = map_graph.get_edge_length(node, neighbor)
        if weight is not None:
            ax.text(label_x, label_y, str(weight), ha='center', va='center',   # type: ignore
                   fontsize=edge_label_fontsize, fontweight='normal', bbox=dict(boxstyle="round,pad=0.3", 
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

        marker, = ax.plot([], [], '^', markersize=agent_markersize, color=bg_color,
                         markeredgecolor='black', linewidth=2, label=f'Agent {i}', zorder=10)
        label = ax.text(0, 0, str(i), ha='center', va='center', color=text_color, fontsize=agent_fontsize, fontweight='bold', zorder=11)
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
    anim.save(animation_filename, writer='pillow', fps=12, dpi=120) 
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

