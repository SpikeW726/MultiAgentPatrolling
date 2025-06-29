from typing import Dict
import os
import random
import matplotlib.pyplot as plt
import pickle
from GBLA_agent import GBLA_agent
from utils.graph_utils import Graph
from utils.visualize_utils import plot_idleness_charts, create_animation

class Extended_GBLA_agent(GBLA_agent):
    def __init__(self, agent_id, init_node, map:Graph, alpha=0.9, gamma=0.9, epsilon=0.1):
        super().__init__(agent_id, init_node, map, gamma, epsilon)
        self.alpha = alpha
        self.Q_table = dict() # the key is (index, action)
        self.ordered_state_list = list()
        self.last_state: tuple = (0,0,(),()) # last_node=0 implies the agent is in initial state 
    
    def get_state(self, node_Idleness, agent_intentions:Dict):
        '''
        return the state tuple of agent
        which is (current_node, last_node, Idle_sorted_neighbor_node_list, neighbor_is_target_dict)
        '''
        neighbors = [n for n, _ in self.map.adj_list[self.position]]
        neighbor_Idleness = [node_Idleness[n] for n in neighbors]
        neighbor_idleness_pairs = list(zip(neighbors, neighbor_Idleness))

        # sort the pairs based on idleness
        # 'reverse=True' means sorting from highest idleness to lowest
        sorted_neighbors_by_idleness = sorted(neighbor_idleness_pairs, key=lambda item: item[1], reverse=True)
        sorted_neighbor_nodes = [node for node, _ in sorted_neighbors_by_idleness]
        # Convert to a tuple to make it hashable for use in a dictionary key
        idle_sorted_neighbor_tuple = tuple(sorted_neighbor_nodes)

        # neighbor_is_target = {n:agent_intentions.get(n, 0) for n in neighbors}
        neighbor_is_target_bool = {n:(agent_intentions.get(n, 0) > 0) for n in neighbors}
        hashable_intentions = tuple(sorted(neighbor_is_target_bool.items()))

        return(self.position, self.last_state[0], idle_sorted_neighbor_tuple, hashable_intentions)

    def get_state_index(self, state):
        try:
            # try to find the index of the state
            state_idx = self.ordered_state_list.index(state)
        except ValueError:
            # if the state is not in the list, add it and get its new index
            self.ordered_state_list.append(state)
            state_idx = len(self.ordered_state_list) - 1
        return state_idx
    
    def update_Q(self, state, action: int, reward, next_state):
        state_idx = self.get_state_index(state)
        last_Q = self.Q_table.get((state_idx, action), 0)

        # adjust the discounted factor according to edge length
        edge_weight = self.map.get_edge_length(state[0], action)
        if not edge_weight == 0:
            discount_factor = self.gamma ** edge_weight
        else:
            discount_factor = self.gamma
        
        # get the max Q value of next state
        next_neighbors = [n for n, _ in self.map.adj_list[action]]
        next_state_idx = self.get_state_index(next_state)
        next_Qs = [self.Q_table.get((next_state_idx, n), 0) for n in next_neighbors]
        max_next_Q = max(next_Qs) if next_Qs else 0

        # update Q-table        
        new_Q = last_Q + self.alpha * (reward + discount_factor * max_next_Q - last_Q)
        self.Q_table[(state_idx, action)] = new_Q

    def select_action(self, state, evaluation_mode=False):
        state_idx = self.get_state_index(state)
        neighbors = [n for n, _ in self.map.adj_list[self.position]]
        if not evaluation_mode and random.random() < self.epsilon:
            return random.choice(neighbors)
        else:
            q_values = [self.Q_table.get((state_idx, n), 0) for n in neighbors]
            max_q = max(q_values)
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
            agent_intentions[action] += 1
            self.last_state = state
            self.last_action = action
            return None

def main():
    # set QUICK_PREVIEW as True for a quick visualization
    QUICK_PREVIEW = True

    map_path1 = 'graphs/simple_8nodes.json'
    map_path2 = 'graphs/medium_12nodes.json'
    map_path3 = 'graphs/essay_MapB.json'
    map_path = map_path3
    map_name = os.path.splitext(os.path.basename(map_path))[0]
    folder_name = f"Extended-GBLA_results"
    os.makedirs(folder_name, exist_ok=True)
    model_filename = os.path.join(folder_name, f'Extended-GBLA_agents_{map_name}.pkl')
    
    train_map = Graph(map_path)
    if map_name == 'essay_MapB':
        agent_num = 10
    elif map_name == 'medium_12nodes' or 'simple_8nodes':
        agent_num = 3
    else:
        agent_num = 2

    episode_num = 10000  
    episode_len = 600 # quantity of time-step in one episode

    Extended_GBLAagents = None
    # try to load pre-trained model
    if os.path.exists(model_filename):
        print(f"Loading pre-trained model from {model_filename}...")
        with open(model_filename, 'rb') as f:
            Extended_GBLAagents = pickle.load(f)

        if len(Extended_GBLAagents) != agent_num:
            print(f"Warning: Model agent count ({len(Extended_GBLAagents)}) mismatches current setting ({agent_num}). Retraining...")
            Extended_GBLAagents = None
            
    if Extended_GBLAagents is None:
        print("No pre-trained model found or agent count mismatch. Starting new training...")
        # randomly initialize the starting node shared by all the agents
        init_position = random.sample(train_map.nodes, 1)
        Extended_GBLAagents = [Extended_GBLA_agent(i, init_position[0], train_map) for i in range(agent_num)]
        # initialize node idleness
        node_Idleness = {n: 0 for n in train_map.nodes}
        node_is_target = {n: 0 for n in train_map.nodes}
        
        # record the normalized average idleness of each episode
        episode_idleness = []

        # train
        for i in range(episode_num):
            # re-initialize the starting position of all agents in every episode
            init_position = random.sample(train_map.nodes, 1)
            for agent in Extended_GBLAagents:
                agent.position = init_position[0]
                agent.last_state = (0,0,(),())
                agent.on_edge = False
                agent.edge_time_left = 0
                agent.target_node = int()
            node_Idleness = {n: 0 for n in train_map.nodes}
            # value indicates the quantity of agent who set key as its target node
            node_is_target = {n: 0 for n in train_map.nodes}  
            total_Idleness = 0   
        
            for step in range(episode_len):
                for n in node_Idleness:
                    node_Idleness[n] += 1
                for agent in Extended_GBLAagents:
                    result = agent.step(node_Idleness, node_is_target)
                    if result is not None:
                        # reach a node
                        node_is_target[result] -= 1
                        if node_is_target[result] == 0: # only one agent select this node as target
                            reward = node_Idleness[result]
                        else: reward = 0
                        node_Idleness[result] = 0
                        current_state = agent.get_state(node_Idleness, node_is_target)
                        agent.update_Q(agent.last_state, agent.last_action, reward, current_state)

                        # choose target-node and move onto the edge immediately
                        next_action = agent.select_action(current_state)
                        edge_weight = agent.map.get_edge_length(agent.position, next_action)
                        
                        agent.on_edge = True
                        agent.edge_time_left = int(edge_weight) if edge_weight is not None else 0
                        agent.target_node = next_action
                        node_is_target[next_action] += 1

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
            
            if (i + 1) % 100 == 0:
                print(f"Episode {i+1}/{episode_num}, Normalized Avg Idleness: {normalized_avg_idleness:.2f}")
        
        # traning plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, episode_num + 1), episode_idleness, 'b-', linewidth=1)
        plt.xlabel('Episode')
        plt.ylabel('Normalized Average Idleness')
        plt.title(f'Extended-GBLA Training Progress on {map_name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        file_name = os.path.join(folder_name,f'Extended-GBLA_training_curve_{map_name}.png')
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training completed! Final normalized average idleness: {episode_idleness[-1]:.2f}")
        print(f"Training curve saved as '{file_name}'")

        with open(model_filename, 'wb') as f:
            pickle.dump(Extended_GBLAagents, f)
        print(f"Trained agents saved to {model_filename}")
    
    # evaluation and visualization
    print("\nStarting evaluation...")
    evaluation_steps = 600

    # reset agents and environment for evaluation
    init_position = random.sample(train_map.nodes, 1)
    for agent in Extended_GBLAagents:
        agent.position = init_position[0]
        agent.last_state = (0,0,(),())
        agent.on_edge = False
        agent.edge_time_left = 0
        agent.target_node = int()

    node_Idleness = {n: 0 for n in train_map.nodes}
    node_is_target = {n: 0 for n in train_map.nodes}

    # data recorders for visualization
    graph_idleness_history = []
    avg_idleness_history = []
    worst_idleness_history = []
    agent_positions_history = []
    sub_steps_per_step = 3
    
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

        # record agent positions for animation
        for sub_step in range(sub_steps_per_step):
            current_positions = {}
            for agent in Extended_GBLAagents:
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
            agent_positions_history.append(current_positions)

        # update agent states
        for agent in Extended_GBLAagents:
            result = agent.step(node_Idleness, node_is_target, evaluation_mode=True)
            if result is not None:
                node_Idleness[result] = 0
                node_is_target[result] -= 1

                current_state = agent.get_state(node_Idleness, node_is_target)
                next_action = agent.select_action(current_state, evaluation_mode=True)
                edge_weight = agent.map.get_edge_length(agent.position, next_action)
                
                agent.on_edge = True
                agent.edge_time_left = int(edge_weight) if edge_weight is not None else 0
                agent.target_node = next_action
                node_is_target[next_action] += 1
                agent.last_state = current_state
                agent.last_action = next_action

    print(f"Evaluation completed! Final average idleness: {avg_idleness_history[-1]:.2f}")
    
    print("\nGenerating visualizations...")
    plot_idleness_charts(avg_idleness_history, worst_idleness_history, "Extended-GBLA", map_name)
    
    total_animation_frames = len(agent_positions_history)
    frames_to_render = total_animation_frames
    if QUICK_PREVIEW:
        frames_to_render = min(450, total_animation_frames) # only render the first 150 time-steps(450 frames)
        print(f"\nQUICK PREVIEW mode is ON. Rendering only {frames_to_render} frames for the animation.")
        
    create_animation(train_map, agent_positions_history, frames_to_render, "Extended-GBLA", map_name)

if __name__ == "__main__":
    main()      
                







