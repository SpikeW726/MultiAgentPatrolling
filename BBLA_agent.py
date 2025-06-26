import random
from utils.graph_utils import Graph

class BBLA_agent:
    def __init__(self, agent_id, init_node, map:Graph, alpha, gamma, epsilon, vel=1):
        self.agent_id = agent_id
        self.position = init_node
        self.edge_from = 0 # last node this agent visited, 0 means the initial state
        self.vel = vel
        self.map = map
        self.Q_table = dict()
        self.last_state = None
        self.last_action = None
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_state(self, neighbor_Idleness):
        '''
        return the state tuple of agent
        which is (current_node, last_node, Idle_max, Idle_min)
        '''
        Idle_max = max(neighbor_Idleness)
        Idle_min = min(neighbor_Idleness)
        return (self.position, self.edge_from, Idle_max, Idle_min)

    def update_Q(self, state, action:int, reward, next_state):
        last_Q = self.Q_table.get((state, action), 0)
        # get the max Q value of next state
        next_neighbors = [n for n, _ in self.map.adj_list[action]]
        next_Qs = [self.Q_table.get((next_state, n), 0) for n in next_neighbors]
        max_next_Q = max(next_Qs) if next_Qs else 0
        # update Q value
        new_Q = last_Q + self.alpha * (reward + self.gamma * max_next_Q - last_Q)
        self.Q_table[(state, action)] = new_Q
    
    def select_action(self, state):
        neighbors = [n for n, _ in self.map.adj_list[self.position]]
        # ¦Å-greedy
        if random.random() < self.epsilon:
            return random.choice(neighbors)
        else:
            q_values = [self.Q_table.get((state, n), 0) for n in neighbors]
            max_q = max(q_values)
            # multiple max Q value actions, select one randomly
            best_actions = [n for n, q in zip(neighbors, q_values) if q == max_q]
            return random.choice(best_actions)

    def step(self, neighbor_Idleness):
        state = self.get_state(neighbor_Idleness)
        action = self.select_action(state)
        self.last_state = state
        self.last_action = action
        return action
    
def main():
    Map_1 = Graph('graphs/graph_topology.json')
    



        