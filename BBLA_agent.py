import random
import matplotlib.pyplot as plt
from utils.graph_utils import Graph

class BBLA_agent:
    def __init__(self, agent_id, init_node, map:Graph, gamma=0.9, epsilon=0.1):
        self.agent_id = agent_id
        self.position:int = init_node
        self.map = map
        self.Q_table = dict()
        self.visit_count = dict() # is used to achieve decreasing alpha
        self.last_state = (0,0,0,0) # last_node=0 implies the agent is in initial state  
        self.last_action = int()
        self.gamma = gamma
        self.epsilon = epsilon
        self.on_edge = False
        self.edge_time_left = 0
        self.target_node = int()

    def get_state(self, node_Idleness):
        '''
        return the state tuple of agent
        which is (current_node, last_node, neighbor_Idle_max, neighbor_Idle_min)
        '''
        neighbors = [n for n, _ in self.map.adj_list[self.position]]
        neighbor_Idleness = [node_Idleness[n] for n in neighbors]
        max_idle = max(neighbor_Idleness) if neighbor_Idleness else 0
        min_idle = min(neighbor_Idleness) if neighbor_Idleness else 0
        return (self.position, self.last_state[0], max_idle, min_idle)

    def update_Q(self, state, action:int, reward, next_state):
        self.visit_count[(state,action)] = self.visit_count.get((state,action), 0) + 1
        alpha = 1 / (2 + self.visit_count[(state,action)]/15) 
        last_Q = self.Q_table.get((state, action), 0)
        # get the max Q value of next state
        next_neighbors = [n for n, _ in self.map.adj_list[action]]
        next_Qs = [self.Q_table.get((next_state, n), 0) for n in next_neighbors]
        max_next_Q = max(next_Qs) if next_Qs else 0
        # update Q value
        new_Q = last_Q + alpha * (reward + self.gamma * max_next_Q - last_Q)
        self.Q_table[(state, action)] = new_Q
    
    def select_action(self, state):
        neighbors = [n for n, _ in self.map.adj_list[self.position]]
        # ε-greedy
        if random.random() < self.epsilon:
            return random.choice(neighbors)
        else:
            q_values = [self.Q_table.get((state, n), 0) for n in neighbors]
            max_q = max(q_values)
            # if there are multiple max Q value actions, select one randomly
            best_actions = [n for n, q in zip(neighbors, q_values) if q == max_q]
            return random.choice(best_actions)

    def step(self, node_Idleness):
        '''
        if reach a node after this time-step, return the node it reach
        otherwise, return None
        '''
        if not self.on_edge:
            # exactly on the node, choose next target node
            state = self.get_state(node_Idleness)
            action = self.select_action(state)

            edge_weight = None
            for n, w in self.map.adj_list[self.position]:
                if n == action:
                    edge_weight = w
                    break
            self.on_edge = True
            self.edge_time_left = int(edge_weight) if edge_weight is not None else 0
            self.target_node = action
            self.last_state = state
            self.last_action = action
            return None
        else:
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

def main():
    Map_1 = Graph('graphs/simple_8nodes.json')
    agent_num = 3
    episode_num = 10000  
    episode_len = 600 # quantity of time-step in one episode
    # initialize positions of the agents randomly
    init_positions = random.sample(Map_1.nodes, agent_num)
    BBLAagents = [BBLA_agent(i, pos, Map_1) for i, pos in enumerate(init_positions)]
    # initialize node idleness
    node_Idleness = {n: 0 for n in Map_1.nodes}
    
    # record the normalized average idleness of each episode
    episode_idleness = []
    
    # train
    for i in range(episode_num):
        # re-initialize positions of the agents in every episode
        init_positions = random.sample(Map_1.nodes, agent_num)
        for i, agent in enumerate(BBLAagents):
            agent.position = init_positions[i]
            agent.last_state = (0,0,0,0)
            agent.on_edge = False
            agent.edge_time_left = 0
            agent.target_node = int()
        node_Idleness = {n: 0 for n in Map_1.nodes}
        total_Idleness = 0
        
        for step in range(episode_len):
            for n in node_Idleness:
                node_Idleness[n] += 1
            for agent in BBLAagents:
                result = agent.step(node_Idleness)
                if result is not None:
                    reward = node_Idleness[result]
                    node_Idleness[result] = 0
                    next_state = agent.get_state(node_Idleness)
                    agent.update_Q(agent.last_state, agent.last_action, reward, next_state)
            
            # calculate the Instantaneous Graph Idleness of current time-step
            current_graph_idleness = sum(node_Idleness.values()) / len(node_Idleness)
            total_Idleness += current_graph_idleness
        
        # calculate the Average Idleness of current episode
        episode_avg_idleness = total_Idleness / episode_len
        
        # normalize
        normalized_avg_idleness = episode_avg_idleness * (agent_num / len(Map_1.nodes))
        episode_idleness.append(normalized_avg_idleness)
        
        if (i + 1) % 100 == 0:
            print(f"Episode {i+1}/{episode_num}, Normalized Avg Idleness: {normalized_avg_idleness:.2f}")
    
    # traning plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, episode_num + 1), episode_idleness, 'b-', linewidth=1)
    plt.xlabel('Episode')
    plt.ylabel('Normalized Average Idleness')
    plt.title('BBLA Training Progress')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('BBLA_training_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training completed! Final normalized average idleness: {episode_idleness[-1]:.2f}")
    print("Training curve saved as 'BBLA_training_curve.png'")
    
    # evaluation and visualization
    print("\nStarting evaluation and visualization...")
    from visualization import evaluate_and_visualize
    evaluate_and_visualize(BBLAagents, Map_1, algorithm_name="BBLA", evaluation_steps=600)

if __name__ == "__main__":
    main()