import numpy as np
import pickle
import torch
import json
from collections import defaultdict

#path = 'bi'
#path = 'mul'
# path = 'unswGan'
path = '../datasets/tor/mul'
GTO = True

def compute_rwr_neighbors(adj_matrix_torch, restart_prob=0.15):
    """
    对每个节点执行RWR，并记录下那些键值为邻居边的节点。
    :param adj_matrix: 邻接矩阵
    :param restart_prob: 重启概率
    :return: 每个节点的邻居节点集合的字典
    """
    num_nodes = adj_matrix_torch.shape[0]
    rwr_neighbors_probs = {}

    # 转移到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adj_matrix_torch = adj_matrix_torch.to(device)
    P = adj_matrix_torch / adj_matrix_torch.sum(dim=1, keepdim=True)

    k = 0
    for start_node in range(num_nodes):
        print(start_node)
        v_start = torch.zeros(num_nodes).to(device)
        v_start[start_node] = 1
        v = v_start.clone()

        for _ in range(20):
            #v_next = (1 - restart_prob) * np.dot(P, v) + restart_prob * v_start
            v_next = (1 - restart_prob) * torch.matmul(P, v) + restart_prob * v_start
            if np.allclose(v.cpu(), v_next.cpu(), atol=1e-6):
                break
            v = v_next
        #print(v_next)
  
        # 从RWR结果中筛选邻居
        neighbors_probs = {}
        for node, prob in enumerate(v_next):
            if adj_matrix_torch[start_node][node] > 0:  # 如果节点是起始节点的邻居
                neighbors_probs[node] = prob.item()
        rwr_neighbors_probs[start_node] = neighbors_probs
        #print(rwr_neighbors_probs[start_node])

    return rwr_neighbors_probs

# nodes
nodes = np.load(path+"/nodes.npy", allow_pickle=True)
num_nodes = len(nodes)
print('num_nodes', num_nodes)

# mapping function from node ip to node id 即ip映射到索引
node_map = {}
for i, node in enumerate(nodes):
    node_map[node] = i

# adjacency adj: edge -> (node1, node2); adj_lists: {node: edge1, ..., edgen}以节点ID为键，值为与该节点相邻的边的id集合
if GTO:
    adj = np.load(path+"/adj_updated.npy", allow_pickle=True)
else:
    adj = np.load(path+"/adj.npy", allow_pickle=True)
#adj = np.load(path+"/adj.npy", allow_pickle=True)
adj_lists = defaultdict(set)
for i, line in enumerate(adj):
    node1 = node_map[line[0]]
    node2 = node_map[line[1]]
    adj_lists[node1].add(i)  # mutual neighbor
    adj_lists[node2].add(i)

# 初始化邻接矩阵
adj_matrix = np.zeros((num_nodes, num_nodes))

# 填充邻接矩阵
for edge in adj:
    node1 = node_map[edge[0]]
    node2 = node_map[edge[1]]
    adj_matrix[node1][node2] = 1
    adj_matrix[node2][node1] = 1  # 如果是无向图

adj_matrix_torch = torch.tensor(adj_matrix, dtype=torch.float32)

# 计算RWR邻居节点及其概率，根据两个节点可以确实边并进行边采样
print('开始计算')
rwr_neighbors_probs = compute_rwr_neighbors(adj_matrix_torch)
print('迭代结束')
# print(rwr_neighbors_probs)

if GTO:
    with open(path + '/gto_rwr_neighbors_probs.json', 'w') as f:
        json.dump(rwr_neighbors_probs, f)
else:
    with open(path + '/rwr_neighbors_probs.json', 'w') as f:
        json.dump(rwr_neighbors_probs, f)