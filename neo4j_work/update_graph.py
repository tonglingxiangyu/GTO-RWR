import numpy as np
import networkx as nx
import random
import re

# 加载数据
path = '../datasets/iot/data'
# path = 'mul'
# path = 'darknet'
# path = 'unswGan'

# 判断是否是 tor 数据集
is_tor_dataset = 'tor' in path.lower()
tor_multiclass = True  # 需要根据实际情况修改此条件

nodes = np.load(path + '/nodes.npy', allow_pickle=True)
adj = np.load(path + '/adj.npy', allow_pickle=True)
print("数组维度:", adj.shape)
edge_features = np.load(path + '/edge_feat_scaled.npy', allow_pickle=True)

# 对于 tor 数据集，根据条件加载标签文件
if is_tor_dataset:
    edge_labels_mul = np.load(path + '/label_mul.npy', allow_pickle=True)
    edge_labels_bi = np.load(path + '/label_bi.npy', allow_pickle=True)
    # 根据某个条件判断使用哪种标签，这里假设使用多类标签
    edge_labels = edge_labels_mul if tor_multiclass else edge_labels_bi
    data_class = 'mul' if tor_multiclass else 'bi'
else:
    # 对于其他数据集，加载并更新两种标签文件
    edge_labels_mul = np.load(path + '/label_mul.npy', allow_pickle=True)
    edge_labels_bi = np.load(path + '/label_bi.npy', allow_pickle=True)

unique_dips = np.load(path + '/unique_dips.npy', allow_pickle=True)

# 创建 NetworkX 图
G = nx.DiGraph()
for node in nodes:
    G.add_node(node)

edge_feature_label_map = {node: [] for node in nodes}  # 为每个节点初始化一个特征列表
for i, edge in enumerate(adj):
    u, v = edge[0], edge[1]
    G.add_edge(u, v)
    if is_tor_dataset:
        edge_feature_label_map[u].append((edge_features[i], edge_labels[i]))  # 将边特征和标签作为元组添加到起点的列表中
    else:
        edge_feature_label_map[u].append((edge_features[i], edge_labels_bi[i], edge_labels_mul[i]))

num_connected_components = nx.number_weakly_connected_components(G)
print('初始的连通量：', num_connected_components)

# 更新图和 adj.npy
new_edges = []
new_edge_features = []
new_edge_labels_mul = []
new_edge_labels_bi = []
feature_dim = edge_features.shape[1]
default_feature = np.ones(feature_dim)

for ip in unique_dips:
    ip_regex = re.compile(r'^' + re.escape(ip) + r':\d+$')
    ip_nodes = [node for node in G.nodes if ip_regex.match(node)]
    ip_nodes = sorted(ip_nodes, key=lambda x: int(x.split(':')[1]), reverse=True)
    if len(ip_nodes) > 1:
        for i in range(len(ip_nodes) - 1):
            new_edge = [ip_nodes[i], ip_nodes[i + 1]]
            new_edges.append(new_edge)
            G.add_edge(ip_nodes[i], ip_nodes[i + 1])

            if edge_feature_label_map[ip_nodes[i]]:
                if is_tor_dataset:
                    selected_feature, selected_label = random.choice(edge_feature_label_map[ip_nodes[i]])
                else:
                    selected_feature, selected_label_bi, selected_label_mul = random.choice(edge_feature_label_map[ip_nodes[i]])
            else:
                selected_feature = default_feature
                selected_label = 0

            new_edge_features.append(selected_feature)
            if is_tor_dataset:
                if tor_multiclass:
                    new_edge_labels_mul.append(selected_label)
                else:
                    new_edge_labels_bi.append(selected_label)
            else:
                new_edge_labels_bi.append(selected_label_bi)
                new_edge_labels_mul.append(selected_label_mul)
            
        new_edge = [ip_nodes[len(ip_nodes) - 1], ip_nodes[0]]
        new_edges.append(new_edge)
        G.add_edge(ip_nodes[len(ip_nodes) - 1], ip_nodes[0])
        
        if edge_feature_label_map[ip_nodes[len(ip_nodes) - 1]]:
            if is_tor_dataset:
                selected_feature, selected_label = random.choice(edge_feature_label_map[ip_nodes[len(ip_nodes) - 1]])
            else:
                selected_feature, selected_label_bi, selected_label_mul = random.choice(edge_feature_label_map[ip_nodes[len(ip_nodes) - 1]])
        else:
            selected_feature = default_feature
            selected_label = 0

        new_edge_features.append(selected_feature)
        if is_tor_dataset:
            if tor_multiclass:
                new_edge_labels_mul.append(selected_label)
            else:
                new_edge_labels_bi.append(selected_label)
        else:
            new_edge_labels_bi.append(selected_label_bi)
            new_edge_labels_mul.append(selected_label_mul)

num_connected_components = nx.number_weakly_connected_components(G)
print('更改后的连通量：', num_connected_components)

# 更新 edge_feat.npy 和 adj.npy
new_edge_features = np.array(new_edge_features)
edge_features = np.vstack((edge_features, new_edge_features))

if is_tor_dataset:
    if tor_multiclass:
        new_edge_labels_mul = np.array(new_edge_labels_mul)
        edge_labels_mul = np.concatenate((edge_labels_mul, new_edge_labels_mul))
        np.save(path + '/label_mul_updated.npy', edge_labels_mul)
    else:
        new_edge_labels_bi = np.array(new_edge_labels_bi)
        edge_labels_bi = np.concatenate((edge_labels_bi, new_edge_labels_bi))
        np.save(path + '/label_bi_updated.npy', edge_labels_bi)
else:
    new_edge_labels_mul = np.array(new_edge_labels_mul)
    edge_labels_mul = np.concatenate((edge_labels_mul, new_edge_labels_mul))
    np.save(path + '/label_mul_updated.npy', edge_labels_mul)
    
    new_edge_labels_bi = np.array(new_edge_labels_bi)
    edge_labels_bi = np.concatenate((edge_labels_bi, new_edge_labels_bi))
    np.save(path + '/label_bi_updated.npy', edge_labels_bi)

np.save(path + '/edge_feat_scaled_updated.npy', edge_features)
adj = np.vstack((adj, new_edges))
np.save(path + '/adj_updated.npy', adj)
print("数组维度:", adj.shape)
