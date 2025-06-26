from utils.graph_utils import load_graph
import random

class BBLAAgent:
    def __init__(self, agent_id, start_node, nodes, adj_list, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.agent_id = agent_id
        self.position = start_node
        self.nodes = nodes
        self.adj_list = adj_list
        self.epsilon = epsilon  # 探索率
        self.alpha = alpha      # 学习率
        self.gamma = gamma      # 折扣因子
        # Q表，key: (state, action)，value: Q值
        self.q_table = dict()
        self.last_state = None
        self.last_action = None

    def get_state(self, node_idleness):
        # 状态可自定义，这里用当前位置和邻居节点的空闲度最大/最小值
        neighbors = [n for n, _ in self.adj_list[self.position]]
        neighbor_idleness = [node_idleness[n] for n in neighbors]
        max_idle = max(neighbor_idleness) if neighbor_idleness else 0
        min_idle = min(neighbor_idleness) if neighbor_idleness else 0
        # 状态元组: (当前位置, 上一步来源, 邻居最大空闲度, 邻居最小空闲度)
        return (self.position, max_idle, min_idle)

    def select_action(self, state, node_idleness):
        # ε-greedy策略
        neighbors = [n for n, _ in self.adj_list[self.position]]
        if random.random() < self.epsilon:
            return random.choice(neighbors)
        # 选择Q值最大的动作
        q_values = [self.q_table.get((state, n), 0) for n in neighbors]
        max_q = max(q_values)
        # 多个最大Q值动作随机选一个
        best_actions = [n for n, q in zip(neighbors, q_values) if q == max_q]
        return random.choice(best_actions)

    def update_q(self, state, action, reward, next_state):
        old_q = self.q_table.get((state, action), 0)
        # 下一个状态所有可能动作的最大Q值
        next_neighbors = [n for n, _ in self.adj_list[action]]
        next_qs = [self.q_table.get((next_state, n), 0) for n in next_neighbors]
        max_next_q = max(next_qs) if next_qs else 0
        # Q-Learning更新
        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
        self.q_table[(state, action)] = new_q

    def step(self, node_idleness):
        state = self.get_state(node_idleness)
        action = self.select_action(state, node_idleness)
        # 记录上一步
        self.last_state = state
        self.last_action = action
        return action

    def learn(self, reward, next_state):
        self.update_q(self.last_state, self.last_action, reward, next_state)


def main():
    # 读取图结构
    nodes, adj_list, phi = load_graph("graphs/graph_topology.json")
    num_agents = 3  # 可自定义
    # agent初始位置随机
    agent_positions = random.sample(nodes, num_agents)
    agents = [BBLAAgent(i, pos, nodes, adj_list) for i, pos in enumerate(agent_positions)]
    # 初始化节点空闲度
    node_idleness = {n: 0 for n in nodes}
    # 训练轮数
    num_episodes = 1000
    steps_per_episode = 200

    for episode in range(num_episodes):
        # 每轮agent位置重置
        agent_positions = random.sample(nodes, num_agents)
        for i, agent in enumerate(agents):
            agent.position = agent_positions[i]
        node_idleness = {n: 0 for n in nodes}
        for step in range(steps_per_episode):
            # 每个agent依次行动
            for agent in agents:
                # 选择下一个节点
                next_node = agent.step(node_idleness)
                # 计算奖励：访问节点的空闲度
                reward = node_idleness[next_node]
                # 更新节点空闲度
                for n in node_idleness:
                    node_idleness[n] += 1
                node_idleness[next_node] = 0  # 被访问归零
                # agent移动
                prev_pos = agent.position
                agent.position = next_node
                # 学习Q表
                next_state = agent.get_state(node_idleness)
                agent.learn(reward, next_state)
            # 可选：打印/记录统计信息

if __name__ == "__main__":
    main() 