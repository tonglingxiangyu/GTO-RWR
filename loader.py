import numpy as np
import torch
import torch.nn as nn
import json
import pickle
from collections import defaultdict
from models import (MeanAggregator, Encoder)


def load_sage(path, binary, GTO, rwr, cuda, device):
    # nodes
    nodes = np.load(path+"nodes.npy", allow_pickle=True)
    num_nodes = len(nodes)

    # features node_feat: all one; edge_feat: scaled
    node_feat = np.ones((num_nodes, 64))
    node_features = nn.Embedding(node_feat.shape[0], node_feat.shape[1])
    node_features.weight = nn.Parameter(torch.FloatTensor(node_feat), requires_grad=False)
    node_features = node_features.to(device)
    if GTO:
        edge_feat = np.load(path+"edge_feat_scaled_updated.npy", allow_pickle=True)
    else:
        edge_feat = np.load(path+"edge_feat_scaled.npy", allow_pickle=True)  # (n,f)
    edge_features = nn.Embedding(edge_feat.shape[0], edge_feat.shape[1])
    edge_features.weight = nn.Parameter(torch.FloatTensor(edge_feat), requires_grad=False)
    edge_features = edge_features.to(device)

    # label
    if binary:
        if GTO:
            label = np.load(path+"label_bi_updated.npy", allow_pickle=True)
        else:
            label = np.load(path+"label_bi.npy", allow_pickle=True)  # (n,1)
    else:
        if GTO:
            label = np.load(path+"label_mul_updated.npy", allow_pickle=True)
        else:
            label = np.load(path+"label_mul.npy", allow_pickle=True)

    # mapping function from node ip to node id 即ip映射到索引
    node_map = {}
    for i, node in enumerate(nodes):
        node_map[node] = i

    # adjacency adj: edge -> (node1, node2); adj_lists: {node: edge1, ..., edgen}以节点ID为键，值为与该节点相邻的边的id集合
    if GTO:
        adj = np.load(path+"adj_updated.npy", allow_pickle=True)
    else:
        adj = np.load(path+"adj.npy", allow_pickle=True)
    adj_lists = defaultdict(set)
    for i, line in enumerate(adj):
        node1 = node_map[line[0]]
        node2 = node_map[line[1]]
        adj_lists[node1].add(i)  # mutual neighbor
        adj_lists[node2].add(i)

    # 节点对到边的索引
    edge_index_map = {}
    for edge_index, (node1, node2) in enumerate(adj):
        sorted_node_pair = tuple(sorted([node_map[node1], node_map[node2]]))
        edge_index_map[sorted_node_pair] = edge_index
    
    if rwr:
        with open(path+'gto_rwr_neighbors_probs.json', 'r') as f:
            rwr_neighbors_probs = json.load(f)
        # 将字符串键转换回整数
        rwr_neighbors_probs = {int(outer_key): {int(inner_key): val for inner_key, val in inner_dict.items()} for outer_key, inner_dict in rwr_neighbors_probs.items()}
        #print(rwr_neighbors_probs[14273])
    else:
        rwr_neighbors_probs = {}
    # Define two layer aggregators and encoders
    # 2-hop
    agg1 = MeanAggregator(edge_features,  edge_index_map, rwr_neighbors_probs, rwr=rwr, gcn=False, cuda=cuda)
    enc1 = Encoder(node_features, edge_feat.shape[1], 64, adj_lists,
                   agg1, num_sample=8, gcn=True, cuda=cuda)
    agg2 = MeanAggregator(edge_features,  edge_index_map, rwr_neighbors_probs, rwr=rwr, gcn=False, cuda=cuda)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), edge_feat.shape[1], 64,
                   adj_lists, agg2, num_sample=8, base_model=enc1, gcn=True, cuda=cuda)
    # 增加hop
    # agg3 = MeanAggregator(edge_features, edge_index_map, rwr_neighbors_probs, rwr=rwr, gcn=False, cuda=cuda)
    # enc3 = Encoder(lambda nodes: enc2(nodes).t(), edge_feat.shape[1], 64,
    #                adj_lists, agg3, num_sample=8, base_model=enc2, gcn=True, cuda=cuda)
    # agg4 = MeanAggregator(edge_features, edge_index_map, rwr_neighbors_probs, rwr=rwr, gcn=False, cuda=cuda)
    # enc4 = Encoder(lambda nodes: enc3(nodes).t(), edge_feat.shape[1], 64,
    #                adj_lists, agg4, num_sample=8, base_model=enc3, gcn=True, cuda=cuda)
    # agg5 = MeanAggregator(edge_features, edge_index_map, rwr_neighbors_probs, rwr=rwr, gcn=False, cuda=cuda)
    # enc5 = Encoder(lambda nodes: enc4(nodes).t(), edge_feat.shape[1], 64,
    #                adj_lists, agg5, num_sample=8, base_model=enc4, gcn=True, cuda=cuda)
    # agg6 = MeanAggregator(edge_features, edge_index_map, rwr_neighbors_probs, rwr=rwr, gcn=False, cuda=cuda)
    # enc6 = Encoder(lambda nodes: enc5(nodes).t(), edge_feat.shape[1], 64,
    #                adj_lists, agg6, num_sample=8, base_model=enc5, gcn=True, cuda=cuda)

    #return enc2, edge_feat, label, node_map, adj
    return enc2, edge_feat, label, node_map, adj


def load_gat(path, device, binary):
    # feature
    edge_feat = np.load(path + "edge_feat_scaled.npy", allow_pickle=True)  # （n,f)
    edge_feat = torch.tensor(edge_feat, dtype=torch.float, device=device)
    #edge_feat = torch.tensor(edge_feat, dtype=torch.float, device='cpu')

    # label
    if binary:
        label = np.load(path + "label_bi.npy", allow_pickle=True)  # (n,1)
    else:
        label = np.load(path+"label_mul.npy", allow_pickle=True)
    label = torch.tensor(label, dtype=torch.long, device=device)  # Cross entropy expects a long int
    #label = torch.tensor(label, dtype=torch.long, device='cpu')

    # adjacency
    adj = np.load(path + "adj_random.npy", allow_pickle=True)
    with open(path + 'adj_random_list.dict', 'rb') as file:
        adj_lists = pickle.load(file)

    # configuration
    config = {
        "num_of_layers": 3,  # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
        "num_heads_per_layer": [6, 6, 6],
        "num_features_per_layer": [edge_feat.shape[1], 8, 8, 8], # edge_feat.shape[0]表示边数，edge_feat.shape[1]表示边特征的维度
        "num_identity_feats": 8,
        "add_skip_connection": False,
        "bias": True,  # result is not so sensitive to bias
        "dropout": 0.2  # result is sensitive to dropout
    }

    return edge_feat, label, adj, adj_lists, config
