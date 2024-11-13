import numpy as np
import pickle
import torch
import json
from collections import defaultdict
#from scipy.sparse import csr_matrix

#path = 'bi'
#path = 'mul'
path = 'unsw'

def compute_rwr_neighbors(adj_matrix_torch, restart_prob=0.15):
    num_nodes = adj_matrix_sparse.shape[0]
    rwr_neighbors_probs = {}

    # 转移到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adj_matrix_torch = adj_matrix_torch.to(device)

    # 计算概率转移矩阵P
    deg = adj_matrix_torch.sum(dim=1).to_dense()
    P = adj_matrix_torch / deg

    for start_node in range(num_nodes):
        v_start = torch.zeros(num_nodes, device=device)
        v_start[start_node] = 1
        v = v_start.clone()

        for _ in range(10):
            v_next = (1 - restart_prob) * P @ v + restart_prob * v_start
            if torch.allclose(v, v_next, atol=1e-6):
                break
            v = v_next

        neighbors_probs = {}
        for neighbor_node, prob in zip(adj_matrix_torch[start_node].indices(), v_next[adj_matrix_torch[start_node].indices()]):
            neighbors_probs[neighbor_node.item()] = prob.item()
        rwr_neighbors_probs[start_node] = neighbors_probs
        print(start_node + 1)
        print(rwr_neighbors_probs[start_node])

    return rwr_neighbors_probs

# nodes
nodes = np.load(path+"/nodes.npy", allow_pickle=True)
num_nodes = len(nodes)

# mapping function from node ip to node id 即ip映射到索引
node_map = {}
for i, node in enumerate(nodes):
    node_map[node] = i

# adjacency adj: edge -> (node1, node2); adj_lists: {node: edge1, ..., edgen}以节点ID为键，值为与该节点相邻的边的id集合
#adj = np.load(path+"/adj_updated.npy", allow_pickle=True)
adj = np.load(path+"/adj.npy", allow_pickle=True)
adj_lists = defaultdict(set)
for i, line in enumerate(adj):
    node1 = node_map[line[0]]
    node2 = node_map[line[1]]
    adj_lists[node1].add(i)  # mutual neighbor
    adj_lists[node2].add(i)

# 使用 (data, (row_ind, col_ind)) 格式创建稀疏矩阵的CSR格式
data = []
row_ind = []
col_ind = []
ind = []

# 填充data, row_ind 和 col_ind
for edge in adj:
    node1 = node_map[edge[0]]
    node2 = node_map[edge[1]]
    data.append(1)  # 边的权重，这里假设为1，表示存在连接
    data.append(1) 
    row_ind.append(node1)  # 行索引，表示边的起点
    row_ind.append(node2) 
    col_ind.append(node2)  # 列索引，表示边的终点
    col_ind.append(node1) 
ind.append(row_ind)
ind.append(col_ind)
ind = torch.LongTensor(ind)
data= torch.FloatTensor(data)

# 创建稀疏矩阵
adj_matrix_sparse = torch.sparse.FloatTensor(ind, data, torch.Size([num_nodes, num_nodes]))
#print(adj_matrix_sparse)
# values = torch.from_numpy(adj_matrix_sparse.data)
# indices = torch.from_numpy(np.vstack((adj_matrix_sparse.indices, adj_matrix_sparse.indptr)))

# # 创建PyTorch的稀疏张量
# adj_matrix_sparse = torch.sparse.FloatTensor(indices, values, torch.Size(adj_matrix_sparse.shape))

# # 转移到GPU
# adj_matrix_torch = torch.tensor(adj_matrix_sparse, dtype=torch.float32)

# 计算RWR邻居节点及其概率，根据两个节点可以确实边并进行边采样
print('开始计算')
rwr_neighbors_probs = compute_rwr_neighbors(adj_matrix_sparse)
print('迭代结束')
#print(rwr_neighbors_probs)
 
with open('rwr_neighbors_probs.json', 'w') as f:
    json.dump(rwr_neighbors_probs, f)