import numpy as np
import networkx as nx
from neo4j import GraphDatabase
import random
import re

# 加载数据
path = '../datasets/iot/data'
#path = 'mul'
# path = 'darknet'
# path = 'unswGan'
multiclass = True # 只有tor需要区分是否是multiclass


nodes = np.load(path+'/nodes.npy', allow_pickle=True)
adj = np.load(path+'/adj.npy', allow_pickle=True)
# print("数据类型:", type(adj))
print("数组维度:", adj.shape)
edge_features = np.load(path+'/edge_feat_scaled.npy', allow_pickle=True)

if multiclass:
    edge_labels = np.load(path+'/label_mul.npy', allow_pickle=True)
    data_class = 'mul'
else:
    edge_labels = np.load(path+'/label_bi.npy', allow_pickle=True)
    data_class = 'bi'

unique_dips = np.load(path+'/unique_dips.npy', allow_pickle=True)

# 创建 NetworkX 图
G = nx.DiGraph()
for node in nodes:
    G.add_node(node)
#edge_feature_map = {}
edge_feature_label_map = {node: [] for node in nodes}  # 为每个节点初始化一个特征列表
for i, edge in enumerate(adj):
    u, v = edge[0], edge[1]
    G.add_edge(u, v)
    edge_feature_label_map[u].append((edge_features[i], edge_labels[i])) # 将边特征和标签作为元组添加到起点的列表中
    #edge_feature_map[u].append(edge_features[i])  # 将边特征添加到起点的特征列表中
    #edge_feature_map[(u, v)] = edge_features[i]
num_connected_components = nx.number_weakly_connected_components(G)
print('初始的连通量：', num_connected_components)

# 更新图和 adj.npy
new_edges = []
new_edge_features = []
new_edge_labels = []  
# 创建一个全是 0.5 的默认向量
feature_dim = edge_features.shape[1]
# default_feature = np.full(feature_dim, 0.5)
default_feature = np.ones(feature_dim)
for ip in unique_dips:
    # 使用正则表达式来匹配确切的IP前缀
    ip_regex = re.compile(r'^' + re.escape(ip) + r':\d+$')
    ip_nodes = [node for node in G.nodes if ip_regex.match(node)]
    # ip_nodes = [node for node in G.nodes if node.startswith(ip)]
    # print(len(ip_nodes))
    ip_nodes = sorted(ip_nodes, key=lambda x: int(x.split(':')[1]), reverse=True)
    if len(ip_nodes) > 1:
        for i in range(len(ip_nodes) - 1):
            new_edge = [ip_nodes[i], ip_nodes[i + 1]]
            new_edges.append(new_edge)
            # G.add_edge(*new_edge)
            G.add_edge(ip_nodes[i], ip_nodes[i + 1])

           # 从边的起点复制特征和标签
            if edge_feature_label_map[ip_nodes[i]]:
                selected_feature, selected_label = random.choice(edge_feature_label_map[ip_nodes[i]])
            else:
                # 如果没有其他边，可以选择一个默认特征或跳过
                selected_feature = default_feature  # 以默认向量作为默认特征
                selected_label = 0  # 默认标签

            new_edge_features.append(selected_feature)
            new_edge_labels.append(selected_label)
            
        new_edge = [ip_nodes[len(ip_nodes) - 1], ip_nodes[0]] # 最后一个连接第一个
        new_edges.append(new_edge)
        # G.add_edge(*new_edge)
        G.add_edge(ip_nodes[len(ip_nodes) - 1], ip_nodes[0])
        # 从边的起点复制特征和标签
        if edge_feature_label_map[ip_nodes[len(ip_nodes) - 1]]:
            selected_feature, selected_label = random.choice(edge_feature_label_map[ip_nodes[len(ip_nodes) - 1]])
        else:
            # 如果没有其他边，可以选择一个默认特征或跳过
            selected_feature = default_feature  # 以默认向量作为默认特征
            selected_label = 0  # 默认标签

        new_edge_features.append(selected_feature)
        new_edge_labels.append(selected_label)

num_connected_components = nx.number_weakly_connected_components(G)
print('更改后的连通量：', num_connected_components)

# 更新 edge_feat.npy 和 adj.npy
new_edge_features = np.array(new_edge_features)
edge_features = np.vstack((edge_features, new_edge_features))
new_edge_labels = np.array(new_edge_labels)
edge_labels = np.concatenate((edge_labels, new_edge_labels))
np.save(path+'/edge_feat_scaled_updated.npy', edge_features)
np.save(path+'/label_'+data_class+'_updated.npy', edge_labels)

#new_adj = [[np.where(nodes == edge[0])[0][0], np.where(nodes == edge[1])[0][0]] for edge in new_edges]
adj = np.vstack((adj, new_edges))
np.save(path+'/adj_updated.npy', adj)  # 保存更新后的文件
# print("数据类型:", type(adj))
print("数组维度:", adj.shape)
# print(adj)

# # 连接到 Neo4j 数据库
# uri = "bolt://localhost:7687"  # 修改为你的 Neo4j 实例的 URI
# username = "neo4j"             # 修改为你的用户名
# password = "12345678"          # 修改为你的密码
# driver = GraphDatabase.driver(uri, auth=(username, password))

# # 将更新后的 NetworkX 图存储到 Neo4j
# def add_graph_to_neo4j(driver, graph):
#     with driver.session() as session:
#         # 添加节点
#         for node in graph.nodes:
#             session.run("MERGE (a:Node {id: $id})", id=node)
#         # 添加边
#         for edge in graph.edges:
#             session.run("MATCH (a:Node {id: $src}), (b:Node {id: $dst}) "
#                         "MERGE (a)-[:CONNECTED]->(b)", src=edge[0], dst=edge[1])

# # 执行存储操作
# add_graph_to_neo4j(driver, G)

# # 关闭数据库连接
# driver.close()
