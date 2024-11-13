import numpy as np
import pickle
import torch
import json
from collections import defaultdict

#path = 'bi'
#path = 'mul'
path = 'darknet'


def identify_subgraph_nodes(full_adj_matrix, start_node, depth=3):
    """
    识别从指定起始节点开始，深度为3的子图中的所有节点。
    """
    #num_nodes = full_adj_matrix.shape[0]
    subgraph_nodes = set([start_node])
    current_level_nodes = set([start_node])

    for _ in range(depth):
        next_level_nodes = set()
        for node in current_level_nodes:
            neighbors = set(np.where(full_adj_matrix[node, :] > 0)[0])
            next_level_nodes.update(neighbors)

        subgraph_nodes.update(next_level_nodes)
        current_level_nodes = next_level_nodes

    return list(subgraph_nodes)

def build_subgraph_matrix(full_adj_matrix, start_node, depth=3):
    """
    构建深度为3的子图的邻接矩阵，并返回子图中节点在原图中的索引。
    """
    subgraph_nodes = identify_subgraph_nodes(full_adj_matrix, start_node, depth)
    subgraph_matrix = full_adj_matrix[np.ix_(subgraph_nodes, subgraph_nodes)]
    #print('subgraph_matrix', subgraph_matrix)
    #print('subgraph_nodes', subgraph_nodes)
    return subgraph_matrix, subgraph_nodes

def rwr(subgraph_matrix, start_node_index_in_subgraph, device, restart_prob=0.15, threshold=0.001):
    """ 
    在子图上应用Random Walk with Restart (RWR) 算法。
    """
    n = subgraph_matrix.shape[0]
    P = subgraph_matrix / subgraph_matrix.sum(dim=1, keepdim=True)
    #print('subgraph_matrix', subgraph_matrix)
    #print('P', P)
    v_start = torch.zeros(n).to(device)
    v_start[start_node_index_in_subgraph] = 1
    v = v_start.clone()

    for _ in range(10):
            #v_next = (1 - restart_prob) * np.dot(P, v) + restart_prob * v_start
            v_next = (1 - restart_prob) * torch.matmul(P, v) + restart_prob * v_start
            if np.allclose(v, v_next, atol=1e-6):
                break
            v = v_next
    #print('v_next', v_next)
    return v_next

# def get_neighbors_with_probs(rwr_result, start_node, full_adj_matrix, subgraph_nodes):
#     """
#     从RWR算法的结果中提取邻居节点及其概率，并使用原始图中的节点ID。
#     """
#     neighbors_probs = {}
#     for idx, prob in enumerate(rwr_result):
#         original_node_id = subgraph_nodes[idx]
#         if full_adj_matrix[start_node, original_node_id] > 0:
#             neighbors_probs[original_node_id] = prob
#     return neighbors_probs

def compute_rwr_neighbors(adj_matrix_torch):
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
    #adj_matrix_torch = adj_matrix_torch.to(device)
    #P = adj_matrix_torch / adj_matrix_torch.sum(dim=1, keepdim=True)

    k = 0
    for start_node in range(num_nodes):
        print(start_node)
        subgraph_matrix, subgraph_nodes = build_subgraph_matrix(adj_matrix_torch, start_node)
        start_node_index_in_subgraph = subgraph_nodes.index(start_node)  # 在子图中找到起始节点的新索引
        #print('start_node_index_in_subgraph', start_node_index_in_subgraph)
        rwr_result = rwr(subgraph_matrix, start_node_index_in_subgraph, device)
  
        # 从RWR结果中筛选邻居
        neighbors_probs = {}
        for node, prob in enumerate(rwr_result):
            node = int(subgraph_nodes[node])
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
adj = np.load(path+"/adj_updated.npy", allow_pickle=True)
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


# 转移到GPU
adj_matrix_torch = torch.tensor(adj_matrix, dtype=torch.float32)

# 计算RWR邻居节点及其概率，根据两个节点可以确实边并进行边采样
print('开始计算')
rwr_neighbors_probs = compute_rwr_neighbors(adj_matrix_torch)
print('迭代结束')
#print(rwr_neighbors_probs)
 
with open(path + '/rwr_neighbors_probs.json', 'w') as f:
    json.dump(rwr_neighbors_probs, f)