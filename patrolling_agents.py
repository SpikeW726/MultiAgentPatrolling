from utils.graph_utils import load_graph
import random

class BBLAAgent:
    def __init__(self, agent_id, start_node, nodes, adj_list, epsilon=0.1, gamma=0.9):
        self.agent_id = agent_id
        self.position = start_node
        self.nodes = nodes
        self.adj_list = adj_list
        self.epsilon = epsilon
        self.gamma = gamma
        self.q_table = dict()
        self.visit_count = dict() # [MODIFIED] 用于动态alpha
        self.last_state = (0, 0, 0, 0)
        self.last_action = None
        # 新增移动状态
        self.on_edge = False # [MODIFIED]
        self.edge_time_left = 0 # [MODIFIED]
        self.target_node = None # [MODIFIED]

    def get_state(self, node_idleness):
        neighbors = [n for n, _ in self.adj_list[self.position]]
        neighbor_idleness = [node_idleness[n] for n in neighbors]
        max_idle = max(neighbor_idleness) if neighbor_idleness else 0
        min_idle = min(neighbor_idleness) if neighbor_idleness else 0
        return (self.position, self.last_state[0], max_idle, min_idle)

    def select_action(self, state, node_idleness):
        neighbors = [n for n, _ in self.adj_list[self.position]]
        if random.random() < self.epsilon:
            return random.choice(neighbors)
        q_values = [self.q_table.get((state, n), 0) for n in neighbors]
        max_q = max(q_values)
        best_actions = [n for n, q in zip(neighbors, q_values) if q == max_q]
        return random.choice(best_actions)

    def update_q(self, state, action, reward, next_state):
        key = (state, action)
        self.visit_count[key] = self.visit_count.get(key, 0) + 1 # [MODIFIED]
        alpha = 1 / (2 + self.visit_count[key]/15) # [MODIFIED]
        old_q = self.q_table.get(key, 0)
        next_neighbors = [n for n, _ in self.adj_list[action]]
        next_qs = [self.q_table.get((next_state, n), 0) for n in next_neighbors]
        max_next_q = max(next_qs) if next_qs else 0
        new_q = old_q + alpha * (reward + self.gamma * max_next_q - old_q)
        self.q_table[key] = new_q

    def step(self, node_idleness):
        if not self.on_edge: # [MODIFIED]
            # 在节点上，选择下一个目标节点
            state = self.get_state(node_idleness)
            action = self.select_action(state, node_idleness)
            # 查找边权
            edge_weight = None
            for n, w in self.adj_list[self.position]:
                if n == action:
                    edge_weight = w
                    break
            self.on_edge = True
            self.edge_time_left = int(edge_weight) if edge_weight is not None else 0 # [MODIFIED]
            self.target_node = action
            self.last_state = state
            self.last_action = action
            return None, 0 # [MODIFIED] 在边上，reward=0
        else:
            if isinstance(self.edge_time_left, int): # [MODIFIED]
                self.edge_time_left -= 1
            else:
                raise ValueError(f"edge_time_left should be int, got {type(self.edge_time_left)}: {self.edge_time_left}") # todo: consider different velocity, which may result in decimal term
            
            if self.edge_time_left == 0:
                # 到达目标节点
                self.position = self.target_node
                self.on_edge = False
                return self.position, 1 # [MODIFIED] 到达节点
            else:
                return None, 0 # [MODIFIED] 还在边上

    def learn(self, reward, next_state):
        self.update_q(self.last_state, self.last_action, reward, next_state)


def main():
    nodes, adj_list, phi = load_graph("graphs/graph_topology.json")
    num_agents = 3
    agent_positions = random.sample(nodes, num_agents)
    agents = [BBLAAgent(i, pos, nodes, adj_list) for i, pos in enumerate(agent_positions)]
    node_idleness = {n: 0 for n in nodes}
    num_episodes = 1000
    steps_per_episode = 200

    for episode in range(num_episodes):
        agent_positions = random.sample(nodes, num_agents)
        for i, agent in enumerate(agents):
            agent.position = agent_positions[i]
            agent.on_edge = False # [MODIFIED]
            agent.edge_time_left = 0 # [MODIFIED]
            agent.target_node = None # [MODIFIED]
        node_idleness = {n: 0 for n in nodes}
        for step in range(steps_per_episode):
            # 所有节点空闲度+1
            for n in node_idleness:
                node_idleness[n] += 1
            for agent in agents:
                result, flag = agent.step(node_idleness)
                if flag == 1 and result is not None: # [MODIFIED] 只有到达节点时才奖励
                    reward = node_idleness[result]
                    node_idleness[result] = 0
                    next_state = agent.get_state(node_idleness)
                    agent.learn(reward, next_state)

if __name__ == "__main__":
    main() 