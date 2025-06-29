from typing import Dict, List
import os
import random
import matplotlib.pyplot as plt
import pickle
from utils.graph_utils import Graph
from utils.visualize_utils import plot_idleness_charts, create_animation
from utils.realtime_visualizer import RealtimeVisualizer

class GBLA_agent:
    def __init__(self, agent_id, init_node, map:Graph, gamma=0.9, epsilon=0.1):
        self.agent_id = agent_id
        self.position:int = init_node
        self.map = map
        self.Q_table = dict()
        self.visit_count = dict() # is used to achieve decreasing alpha
        self.last_state: tuple = (0,0,0,0,()) # last_node=0 implies the agent is in initial state  
        self.last_action = int()
        self.gamma = gamma
        self.epsilon = epsilon
        self.on_edge = False
        self.edge_time_left = 0
        self.target_node = int()

    def get_state(self, node_Idleness, agent_intentions:Dict):
        '''
        return the state tuple of agent
        which is (current_node, last_node, neighbor_maxIdle_node, neighbor_minIdle_node, neighbor_is_target_dict)
        '''
        neighbors = [n for n, _ in self.map.adj_list[self.position]]
        neighbor_Idleness = [node_Idleness[n] for n in neighbors]

        max_idle = max(neighbor_Idleness) if neighbor_Idleness else 0
        max_nodes = [n for n, q in zip(neighbors, neighbor_Idleness) if q == max_idle]
        max_node = random.choice(max_nodes)
        min_idle = min(neighbor_Idleness) if neighbor_Idleness else 0
        min_nodes = [n for n, q in zip(neighbors, neighbor_Idleness) if q == min_idle]
        min_node = random.choice(min_nodes)

        neighbor_is_target = {n:agent_intentions.get(n, False) for n in neighbors}
        
        # To be used as a dictionary key, the state must be hashable.
        # We convert the intentions dict to a sorted tuple of items.
        hashable_intentions = tuple(sorted(neighbor_is_target.items()))

        return (self.position, self.last_state[0], max_node, min_node, hashable_intentions)

    def update_Q(self, state, action:int, reward, next_state):
        self.visit_count[(state,action)] = self.visit_count.get((state,action), 0) + 1
        alpha = 1 / (2 + self.visit_count[(state,action)]/15) 
        last_Q = self.Q_table.get((state, action), 0)
        
        # adjust the discounted factor according to edge length
        edge_weight = self.map.get_edge_length(state[0], action)
        
        if not edge_weight == 0:
            discount_factor = self.gamma ** edge_weight
        else:
            discount_factor = self.gamma
            
        # get the max Q value of next state
        next_neighbors = [n for n, _ in self.map.adj_list[action]]
        next_Qs = [self.Q_table.get((next_state, n), 0) for n in next_neighbors]
        max_next_Q = max(next_Qs) if next_Qs else 0
        # update Q-table
        new_Q = last_Q + alpha * (reward + discount_factor * max_next_Q - last_Q)
        self.Q_table[(state, action)] = new_Q
    
    def select_action(self, state, evaluation_mode=False):
        neighbors = [n for n, _ in self.map.adj_list[self.position]]
        # in evaluation mode, always be greedy. Otherwise, use ε-greedy.
        if not evaluation_mode and random.random() < self.epsilon:
            return random.choice(neighbors)
        else:
            q_values = [self.Q_table.get((state, n), 0) for n in neighbors]
            max_q = max(q_values)
            # if there are multiple max Q value actions, select one randomly
            best_actions = [n for n, q in zip(neighbors, q_values) if q == max_q]
            return random.choice(best_actions)

    def step(self, node_Idleness, agent_intentions: Dict, evaluation_mode=False):
        '''
        if reach a node after this time-step, return the node it reach
        otherwise, return None
        '''
        if self.on_edge:
            if isinstance(self.edge_time_left, int):
                self.edge_time_left -= 1
            else:
                # todo: consider different velocity, which may result in decimal term
                raise ValueError(f"edge_time_left should be int, got {type(self.edge_time_left)}: {self.edge_time_left}") 
            if self.edge_time_left == 0:
                self.position = self.target_node
                self.on_edge = False
                return self.position
            else:
                return None
            
        else:
            # special case: the agent is just initailized
            state = self.get_state(node_Idleness, agent_intentions)
            action = self.select_action(state, evaluation_mode)

            edge_weight = self.map.get_edge_length(self.position, action)

            self.on_edge = True
            self.edge_time_left = int(edge_weight) if edge_weight is not None else 0
            self.target_node = action
            agent_intentions[action] = True
            self.last_state = state
            self.last_action = action
            return None

def main():
    # Set to True to open an interactive window for visualization during evaluation.
    # Set to False to run evaluation without the real-time window.
    ENABLE_REALTIME_VISUALIZATION = True 

    map_path1 = 'graphs/simple_8nodes.json'
    map_path2 = 'graphs/medium_12nodes.json'
    map_path3 = 'graphs/essay_MapB.json'
    map_path = map_path3
    map_name = os.path.splitext(os.path.basename(map_path))[0]
    folder_name = f"GBLA_results"
    os.makedirs(folder_name, exist_ok=True)
    model_filename = os.path.join(folder_name, f'GBLA_agents_{map_name}.pkl')
    
    train_map = Graph(map_path)
    if map_name == 'essay_MapB':
        agent_num = 10
    elif map_name == 'medium_12nodes' or 'simple_8nodes':
        agent_num = 3
    else:
        agent_num = 2

    episode_num = 10000  
    episode_len = 600 # quantity of time-step in one episode

    GBLAagents = None
    # try to load pre-trained model
    if os.path.exists(model_filename):
        print(f"Loading pre-trained model from {model_filename}...")
        with open(model_filename, 'rb') as f:
            GBLAagents = pickle.load(f)

        if len(GBLAagents) != agent_num:
            print(f"Warning: Model agent count ({len(GBLAagents)}) mismatches current setting ({agent_num}). Retraining...")
            GBLAagents = None
            
    if GBLAagents is None:
        print("No pre-trained model found or agent count mismatch. Starting new training...")
        # initialize positions of the agents randomly
        init_positions = random.sample(train_map.nodes, agent_num)
        GBLAagents = [GBLA_agent(i, pos, train_map) for i, pos in enumerate(init_positions)]
        # initialize node idleness
        node_Idleness = {n: 0 for n in train_map.nodes}
        node_is_target = {n: False for n in train_map.nodes}
        
        # record the normalized average idleness of each episode
        episode_idleness = []
        
        # train
        for j in range(episode_num):
            # re-initialize positions of the agents in every episode
            init_positions = random.sample(train_map.nodes, agent_num)
            for i, agent in enumerate(GBLAagents):
                agent.position = init_positions[i]
                agent.last_state = (0,0,0,0,())
                agent.on_edge = False
                agent.edge_time_left = 0
                agent.target_node = int()
            node_Idleness = {n: 0 for n in train_map.nodes}
            node_is_target = {n: False for n in train_map.nodes}
            total_Idleness = 0
            
            for step in range(episode_len):
                for n in node_Idleness:
                    node_Idleness[n] += 1
                for agent in GBLAagents:
                    result = agent.step(node_Idleness, node_is_target)
                    if result is not None:
                        # reach a node
                        reward = node_Idleness[result]
                        node_Idleness[result] = 0
                        node_is_target[result] = False
                        current_state = agent.get_state(node_Idleness, node_is_target)
                        agent.update_Q(agent.last_state, agent.last_action, reward, current_state)
                        
                        # choose target-node and move onto the edge immediately
                        next_action = agent.select_action(current_state)
                        edge_weight = agent.map.get_edge_length(agent.position, next_action)
                        
                        agent.on_edge = True
                        agent.edge_time_left = int(edge_weight) if edge_weight is not None else 0
                        agent.target_node = next_action
                        node_is_target[next_action] = True
                        
                        agent.last_state = current_state
                        agent.last_action = next_action

                # calculate the Instantaneous Graph Idleness of current time-step
                current_graph_idleness = sum(node_Idleness.values()) / len(node_Idleness)
                total_Idleness += current_graph_idleness
            
            # calculate the Average Idleness of current episode
            episode_avg_idleness = total_Idleness / episode_len
            
            # normalize
            normalized_avg_idleness = episode_avg_idleness * (agent_num / len(train_map.nodes))
            episode_idleness.append(normalized_avg_idleness)
            
            if (j + 1) % 100 == 0:
                print(f"Episode {j+1}/{episode_num}, Normalized Avg Idleness: {normalized_avg_idleness:.2f}")

        # traning plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, episode_num + 1), episode_idleness, 'b-', linewidth=1)
        plt.xlabel('Episode')
        plt.ylabel('Normalized Average Idleness')
        plt.title(f'GBLA Training Progress on {map_name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        file_name = os.path.join(folder_name,f'GBLA_training_curve_{map_name}.png')
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training completed! Final normalized average idleness: {episode_idleness[-1]:.2f}")
        print(f"Training curve saved as '{file_name}'")

        with open(model_filename, 'wb') as f:
            pickle.dump(GBLAagents, f)
        print(f"Trained agents saved to {model_filename}")
    
    # evaluation and visualization
    print("\nStarting evaluation...")
    evaluation_steps = 600
    
    # reset agents and environment for evaluation
    init_positions = random.sample(train_map.nodes, agent_num)
    for i, agent in enumerate(GBLAagents):
        agent.position = init_positions[i]
        agent.last_state = (0,0,0,0,())
        agent.on_edge = False
        agent.edge_time_left = 0
        agent.target_node = int()
        
    node_Idleness = {n: 0 for n in train_map.nodes}
    node_is_target = {n: False for n in train_map.nodes}

    # data recorders for plotting idleness charts
    graph_idleness_history = []
    avg_idleness_history = []
    worst_idleness_history = []
    
    # --- Real-time Visualizer Setup ---
    visualizer = None
    if ENABLE_REALTIME_VISUALIZATION:
        print("Real-time visualization is enabled.")
        visualizer = RealtimeVisualizer(train_map, agent_num, "GBLA", map_name)
    
    for step in range(evaluation_steps):
        for n in node_Idleness:
            node_Idleness[n] += 1
        # record idleness for plotting
        normalization = agent_num / len(train_map.nodes)
        current_graph_idleness = sum(node_Idleness.values()) * normalization / len(node_Idleness)
        graph_idleness_history.append(current_graph_idleness)
        current_avg_idleness = sum(graph_idleness_history) / (step + 1)
        avg_idleness_history.append(current_avg_idleness)
        current_worst_idleness = max(node_Idleness.values()) * normalization
        worst_idleness_history.append(current_worst_idleness)
        
        # --- Real-time Visualization Update ---
        if visualizer:
            # We use sub-steps here to make the agent movement appear smoother in the real-time window.
            sub_steps_per_step = 3 
            visualization_active = True
            for sub_step in range(sub_steps_per_step):
                current_positions = {}
                for agent in GBLAagents:
                    if agent.on_edge:
                        edge_weight = agent.map.get_edge_length(agent.last_state[0], agent.target_node)
                        if edge_weight is not None and edge_weight > 0:
                            step_progress = (edge_weight - agent.edge_time_left) / edge_weight
                            sub_progress = sub_step / sub_steps_per_step
                            total_progress = min(1.0, step_progress + sub_progress / edge_weight)
                        else:
                            total_progress = 1.0
                        current_positions[agent.agent_id] = (agent.last_state[0], agent.target_node, total_progress)
                    else:
                        current_positions[agent.agent_id] = (agent.position, agent.position, 1.0)
                
                if not visualizer.update(current_positions):
                    # Window was closed by user, stop trying to update
                    visualizer = None 
                    visualization_active = False
                    break
            if not visualization_active:
                break # Exit evaluation loop if window was closed

        # update agent states
        for agent in GBLAagents:
            result = agent.step(node_Idleness, node_is_target, evaluation_mode=True)
            if result is not None:
                reward = node_Idleness[result]
                node_Idleness[result] = 0
                if result in node_is_target:
                    node_is_target[result] = False
                
                current_state = agent.get_state(node_Idleness, node_is_target)
                next_action = agent.select_action(current_state, evaluation_mode=True)
                edge_weight = agent.map.get_edge_length(agent.position, next_action)
                
                agent.on_edge = True
                agent.edge_time_left = int(edge_weight) if edge_weight is not None else 0
                agent.target_node = next_action
                node_is_target[next_action] = True
                agent.last_state = current_state
                agent.last_action = next_action

    if len(avg_idleness_history) > 0:
        print(f"Evaluation completed! Final average idleness: {avg_idleness_history[-1]:.2f}")
    else:
        print("Evaluation stopped early.")
    
    print("\nGenerating idleness charts...")
    plot_idleness_charts(avg_idleness_history, worst_idleness_history, "GBLA", map_name)
    
    # --- Close Real-time Visualizer Window ---
    if visualizer:
        visualizer.close()

if __name__ == "__main__":
    main()