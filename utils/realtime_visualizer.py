import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

# Import helpers from the original visualizer to reuse logic
from utils.graph_utils import Graph
from utils.nx_layout import create_nx_layout
from utils.visualize_utils import get_text_color_for_bg, calculate_label_positions

class RealtimeVisualizer:
    """
    Handles real-time, interactive visualization of agent movements on the graph.
    Instead of creating a GIF after the simulation, it opens a matplotlib window
    and updates it at each step of the simulation.
    """

    def __init__(self, map_graph: Graph, num_agents: int, algorithm_name: str, map_name: str):
        """
        Initializes the visualization window, draws the static graph, and prepares
        agent markers.
        """
        plt.ion()  # Turn on interactive mode for real-time updates

        self.map_graph = map_graph
        num_nodes = len(self.map_graph.nodes)

        # --- Dynamic visual element scaling (copied from visualize_utils.py) ---
        figure_scale_factor = 1.0 + num_nodes / 30.0
        baseline_nodes = 12.0
        visual_scale_factor = max(0.6, np.sqrt(baseline_nodes / num_nodes)) if num_nodes > 0 else 1.0

        node_markersize = 35 * visual_scale_factor
        node_fontsize = 16 * visual_scale_factor
        self.agent_markersize = 28 * visual_scale_factor
        self.agent_fontsize = 10 * visual_scale_factor
        edge_label_fontsize = 10 * visual_scale_factor

        # --- Figure and Axis Setup (copied from visualize_utils.py) ---
        self.fig, self.ax = plt.subplots(figsize=(12 * figure_scale_factor, 10 * figure_scale_factor))
        self.node_positions = create_nx_layout(self.map_graph)

        # Draw static graph elements
        self._draw_graph_background(node_fontsize, node_markersize, edge_label_fontsize)
        self.ax.set_title(f'{algorithm_name} on {map_name} - Real-time Movement', fontsize=16, fontweight='bold')

        # --- Agent Markers Setup (copied from visualize_utils.py) ---
        self.agent_markers = []
        self.agent_labels = []
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'yellow']
        rgb_colors = [to_rgb(c) for c in colors]

        for i in range(num_agents):
            bg_color = rgb_colors[i % len(rgb_colors)]
            text_color = get_text_color_for_bg(bg_color)
            marker, = self.ax.plot([], [], '^', markersize=self.agent_markersize, color=bg_color,
                                   markeredgecolor='black', linewidth=2, label=f'Agent {i}', zorder=10)
            label = self.ax.text(0, 0, str(i), ha='center', va='center', color=text_color, 
                                 fontsize=self.agent_fontsize, fontweight='bold', zorder=11)
            self.agent_markers.append(marker)
            self.agent_labels.append(label)
        
        self.ax.legend(fontsize=12, labelspacing=1.5)
        self.fig.canvas.manager.set_window_title("Real-time Agent Patrolling")
        self.fig.show()
        plt.pause(0.1) # Pause to ensure the window is drawn

    def _draw_graph_background(self, node_fontsize, node_markersize, edge_label_fontsize):
        """Helper to draw the static parts of the graph."""
        # Draw edges
        for node in self.map_graph.nodes:
            for neighbor, weight in self.map_graph.adj_list[node]:
                if node < neighbor:
                    pos1 = self.node_positions[node]
                    pos2 = self.node_positions[neighbor]
                    self.ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'k-', alpha=0.6, linewidth=2)

        # Draw nodes
        for node, pos in self.node_positions.items():
            self.ax.plot(pos[0], pos[1], 'o', markersize=node_markersize, color='skyblue', markeredgecolor='black', linewidth=2)
            self.ax.text(pos[0], pos[1], str(node), ha='center', va='center', fontsize=node_fontsize, fontweight='bold')

        # Draw edge weight labels
        label_positions = calculate_label_positions(self.map_graph, self.node_positions)
        for (node, neighbor), (label_x, label_y) in label_positions.items():
            weight = self.map_graph.get_edge_length(node, neighbor)
            if weight is not None:
                self.ax.text(label_x, label_y, str(weight), ha='center', va='center',
                             fontsize=edge_label_fontsize, fontweight='normal', bbox=dict(boxstyle="round,pad=0.3",
                             facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=0.5))

        # Set plot limits and aesthetics
        x_coords = [pos[0] for pos in self.node_positions.values()]
        y_coords = [pos[1] for pos in self.node_positions.values()]
        x_margin = (max(x_coords) - min(x_coords)) * 0.1 if x_coords else 0.1
        y_margin = (max(y_coords) - min(y_coords)) * 0.1 if y_coords else 0.1
        self.ax.set_xlim(min(x_coords) - x_margin, max(x_coords) + x_margin)
        self.ax.set_ylim(min(y_coords) - y_margin, max(y_coords) + y_margin)
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)

    def update(self, agent_positions_this_frame: dict):
        """
        Updates the positions of all agents on the plot for a single frame.

        Args:
            agent_positions_this_frame (dict): A dictionary mapping agent_id to
                                               (start_node, end_node, progress).
        """
        if not plt.fignum_exists(self.fig.number):
            print("Visualization window was closed. Stopping updates.")
            return False # Indicate that the loop should stop

        for i, (agent_id, (start_node, end_node, progress)) in enumerate(agent_positions_this_frame.items()):
            if i < len(self.agent_markers):
                if start_node == end_node:
                    pos = self.node_positions[start_node]
                    self.agent_markers[i].set_data([pos[0]], [pos[1]])
                    self.agent_labels[i].set_position((pos[0], pos[1]))
                else:
                    # Ensure nodes exist before getting positions
                    if start_node not in self.node_positions or end_node not in self.node_positions:
                        continue
                    start_pos = self.node_positions[start_node]
                    end_pos = self.node_positions[end_node]
                    current_x = start_pos[0] + progress * (end_pos[0] - start_pos[0])
                    current_y = start_pos[1] + progress * (end_pos[1] - start_pos[1])
                    self.agent_markers[i].set_data([current_x], [current_y])
                    self.agent_labels[i].set_position((current_x, current_y))
        
        # Redraw the canvas
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01) # A small pause is needed for redraw
        return True

    def close(self):
        """
        Keeps the window open until it's manually closed by the user.
        """
        if plt.fignum_exists(self.fig.number):
            print("\nVisualization finished. Close the plot window to end the program.")
            self.ax.set_title("Simulation Finished. Close window to exit.", fontsize=16, fontweight='bold')
            self.fig.canvas.draw()
            plt.show(block=True) # Keep window open until user closes it
        plt.ioff()
        plt.close(self.fig) 