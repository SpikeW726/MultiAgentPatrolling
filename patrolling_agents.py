from utils.graph_utils import load_graph
import random

class BBLAAgent:
    def __init__(self, agent_id, start_node, nodes, adj_list, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.agent_id = agent_id
        self.position = start_node
        self.nodes = nodes
        self.adj_list = adj_list
        self.epsilon = epsilon  # ̽����
        self.alpha = alpha      # ѧϰ��
        self.gamma = gamma      # �ۿ�����
        # Q��key: (state, action)��value: Qֵ
        self.q_table = dict()
        self.last_state = None
        self.last_action = None

    def get_state(self, node_idleness):
        # ״̬���Զ��壬�����õ�ǰλ�ú��ھӽڵ�Ŀ��ж����/��Сֵ
        neighbors = [n for n, _ in self.adj_list[self.position]]
        neighbor_idleness = [node_idleness[n] for n in neighbors]
        max_idle = max(neighbor_idleness) if neighbor_idleness else 0
        min_idle = min(neighbor_idleness) if neighbor_idleness else 0
        # ״̬Ԫ��: (��ǰλ��, ��һ����Դ, �ھ������ж�, �ھ���С���ж�)
        return (self.position, max_idle, min_idle)

    def select_action(self, state, node_idleness):
        # ��-greedy����
        neighbors = [n for n, _ in self.adj_list[self.position]]
        if random.random() < self.epsilon:
            return random.choice(neighbors)
        # ѡ��Qֵ���Ķ���
        q_values = [self.q_table.get((state, n), 0) for n in neighbors]
        max_q = max(q_values)
        # ������Qֵ�������ѡһ��
        best_actions = [n for n, q in zip(neighbors, q_values) if q == max_q]
        return random.choice(best_actions)

    def update_q(self, state, action, reward, next_state):
        old_q = self.q_table.get((state, action), 0)
        # ��һ��״̬���п��ܶ��������Qֵ
        next_neighbors = [n for n, _ in self.adj_list[action]]
        next_qs = [self.q_table.get((next_state, n), 0) for n in next_neighbors]
        max_next_q = max(next_qs) if next_qs else 0
        # Q-Learning����
        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
        self.q_table[(state, action)] = new_q

    def step(self, node_idleness):
        state = self.get_state(node_idleness)
        action = self.select_action(state, node_idleness)
        # ��¼��һ��
        self.last_state = state
        self.last_action = action
        return action

    def learn(self, reward, next_state):
        self.update_q(self.last_state, self.last_action, reward, next_state)


def main():
    # ��ȡͼ�ṹ
    nodes, adj_list, phi = load_graph("graphs/graph_topology.json")
    num_agents = 3  # ���Զ���
    # agent��ʼλ�����
    agent_positions = random.sample(nodes, num_agents)
    agents = [BBLAAgent(i, pos, nodes, adj_list) for i, pos in enumerate(agent_positions)]
    # ��ʼ���ڵ���ж�
    node_idleness = {n: 0 for n in nodes}
    # ѵ������
    num_episodes = 1000
    steps_per_episode = 200

    for episode in range(num_episodes):
        # ÿ��agentλ������
        agent_positions = random.sample(nodes, num_agents)
        for i, agent in enumerate(agents):
            agent.position = agent_positions[i]
        node_idleness = {n: 0 for n in nodes}
        for step in range(steps_per_episode):
            # ÿ��agent�����ж�
            for agent in agents:
                # ѡ����һ���ڵ�
                next_node = agent.step(node_idleness)
                # ���㽱�������ʽڵ�Ŀ��ж�
                reward = node_idleness[next_node]
                # ���½ڵ���ж�
                for n in node_idleness:
                    node_idleness[n] += 1
                node_idleness[next_node] = 0  # �����ʹ���
                # agent�ƶ�
                prev_pos = agent.position
                agent.position = next_node
                # ѧϰQ��
                next_state = agent.get_state(node_idleness)
                agent.learn(reward, next_state)
            # ��ѡ����ӡ/��¼ͳ����Ϣ

if __name__ == "__main__":
    main() 