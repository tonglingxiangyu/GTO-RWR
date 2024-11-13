import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random

global_edge_features = []
global_count = 0

class EGraphSage(nn.Module):

    def __init__(self, num_classes, enc, edge_feat, node_map, adj, residual):
        #super(EGraphSage, self).__init__()
        super().__init__()
        self.enc = enc # encoder
        self.edge_features = edge_feat
        self.node_map = node_map
        self.adj = adj # edges -> (node1, node2)
        self.xent = nn.CrossEntropyLoss()
        self.residual = residual

        if self.residual:
            self.weight = nn.Parameter(torch.FloatTensor(num_classes, 2 * enc.embed_dim + edge_feat.shape[1]))
        else:
            self.weight = nn.Parameter(torch.FloatTensor(num_classes, 2 * enc.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, edges):
        # E-GraphSAGE
        nodes = self.adj[edges] #一个batch的边的两个节点
        nodes = [set(x) for x in nodes]
        nodes = list(set.union(*nodes))
        nodes_id = [self.node_map[node] for node in nodes] #得到节点的索引

        # construct mapping from node_id to index 建立从节点索引到其在特定操作中的索引的映射
        unique_map = {}
        for idx, id in enumerate(nodes_id):
            unique_map[id] = idx

        node_embeds = self.enc(nodes_id).t().cpu().detach().numpy() # (N,e)  e: embed_dim

        if self.residual:
            edge_embeds = np.array([np.concatenate(
                (node_embeds[unique_map[self.node_map[self.adj[edge][0]]]],
                 node_embeds[unique_map[self.node_map[self.adj[edge][1]]]],
                 self.edge_features[edge])) 
                 for edge in edges])
        else:
            edge_embeds = np.array([np.concatenate(
                (node_embeds[unique_map[self.node_map[self.adj[edge][0]]]],
                 node_embeds[unique_map[self.node_map[self.adj[edge][1]]]])) 
                for edge in edges])
        edge_embeds = torch.FloatTensor(edge_embeds)
        scores = self.weight.mm(edge_embeds.t()) # W * embed: (c,2e) * (2e,E)  --> (c,E)  c : num of classes

        # Applying the additional layers

        return scores.t(), edge_embeds  # (c,E) -->(E,c)

    def loss(self, edges, labels):
        scores, edge_embeds = self.forward(edges)
        global global_count
        global global_edge_features
        # if global_count % 2 == 1:
        #     global_edge_features.pop()
        # print('edge_embeds', edge_embeds.shape)
        labels = labels.view(-1, 1)  # 假设labels是Tensor且初试维度是1x500
        edge_embeds = np.hstack([edge_embeds, labels.cpu().numpy()])  # 将labels添加到每个edge_embeds
        # print('edge_embeds', edge_embeds.shape)
        global_edge_features.append(edge_embeds)
        # global_count += 1
        return self.xent(scores, labels.squeeze())


class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighboring edges' embeddings
    """

    def __init__(self, edge_features,  edge_index_map, rwr_neighbors_probs, rwr=False, cuda=False, gcn=False):
        """
        Initializes the aggregator for a specific graph.

        edge_features -- function mapping LongTensor of edge ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        #super(MeanAggregator, self).__init__()
        super().__init__()

        self.edge_features = edge_features
        self.cuda = cuda
        self.gcn = gcn
        self.rwr = rwr
        self.edge_index_map = edge_index_map
        self.rwr_neighbors_probs = rwr_neighbors_probs

    # 通过节点对查询边的索引
    def get_edge_index(node1, node2, edge_index_map):
        sorted_node_pair = tuple(sorted([node1, node2]))
        return edge_index_map.get(sorted_node_pair)
    
    def forward(self, nodes, to_neighs, num_sample=None):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbor edges for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        if self.rwr:
            if num_sample is not None:
                samp_neighs = []
                #print('nodes：', nodes)
                for index, node in enumerate(nodes):
                    neighs = set([])
                    if  len(to_neighs[index]) <= num_sample:
                        neighs = to_neighs[index]
                    else:
                        # node = str(node)
                        # print(self.rwr_neighbors_probs[node])
                        # 边采样
                        sorted_neighbors = sorted(self.rwr_neighbors_probs[node], key=self.rwr_neighbors_probs[node].get, reverse=True)[:num_sample]
                        # print(sorted_neighbors)
                        for n in sorted_neighbors:
                            node_pair = (node, n)
                            edge_indices = self.edge_index_map.get(tuple(sorted(node_pair)))
                            neighs.add(edge_indices)
                    samp_neighs.append(neighs)
                #print(samp_neighs)
            else:
                samp_neighs = to_neighs
        else:
            _set = set  # Local pointers to functions (speed hack)
            if num_sample is not None:
                _sample = random.sample
                samp_neighs = [
                    _set(_sample(
                        to_neigh,
                        num_sample,
                    )) if len(to_neigh) >= num_sample else to_neigh
                    for to_neigh in to_neighs
                ]  # sample = neighbor, if neighboorhood size is less than num_sample
            else:
                samp_neighs = to_neighs

        if self.gcn:  # gcn=True, add self node into neighbor
            samp_neighs = [
                samp_neigh.union(set([nodes[i]]))
                for i, samp_neigh in enumerate(samp_neighs)
            ]
        #print('边：', to_neighs)
        #print('采样边：', samp_neighs)
        unique_edges_list = list(set.union(*samp_neighs))
        unique_edges = {n: i for i, n in enumerate(unique_edges_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_edges))) # mask张量形状为节点数量乘以唯一边的数量
        column_indices = [
            unique_edges[n] for samp_neigh in samp_neighs for n in samp_neigh
        ]

        row_indices = [
            i for i in range(len(samp_neighs))
            for j in range(len(samp_neighs[i]))
        ]

        mask[row_indices, column_indices] = 1 # mask中相邻边标记为1

        if self.cuda:
            mask = mask.cuda()

        num_neigh = mask.sum(1, keepdim=True)  # torch.sum()  (n,m) --> (n, 1)
        mask = mask.div(num_neigh)  # normalization 归一化平均值，表示邻居的均匀贡献

        if self.cuda:
            embed_matrix = self.edge_features(
                torch.LongTensor(unique_edges_list).cuda())
        else:
            embed_matrix = self.edge_features(torch.LongTensor(unique_edges_list))
        to_feats = mask.mm(embed_matrix) # 通过掩码矩阵和边嵌入矩阵的乘积得到
        return to_feats


class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    2-hop in loader.py
    """
    def __init__(self,
                 node_features,
                 feature_dim,
                 embed_dim,
                 adj_lists,
                 aggregator,
                 num_sample=None,
                 base_model=None,
                 gcn=False,
                 cuda=False):
        #super(Encoder, self).__init__()
        super().__init__()

        self.node_features = node_features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn  # True: Mean-agg  False: GCN-agg
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.aggregator.cuda = cuda
        self.weight = nn.Parameter(
            torch.FloatTensor(
                embed_dim, self.feat_dim + self.embed_dim if self.gcn else self.feat_dim
            ).to(self.device)  # Mean-agg: [embed_dim,feat_dim] ; GCN-agg: [embed_dim, 2*feat_dim]
        )

        init.xavier_uniform_(self.weight) # 初始化权重

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.
        nodes     -- list of nodes
        """
        if isinstance(nodes, torch.Tensor):
            # 将张量转换为列表
            nodes = nodes.tolist()
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes],
                                              self.num_sample)  # (nodes，neighbor edges，num samples) 聚合的邻居特征
        if self.gcn:
            if self.cuda:
                self_feats = self.node_features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = self.node_features(torch.LongTensor(nodes)) # (n , f) n : num of nodes
            neigh_feats[torch.isnan(neigh_feats)] = 1e-2
            combined = torch.cat([self_feats, neigh_feats], dim=1).to(self.device)  # (n , 2f) 更新的节点特征
        else:
            combined = neigh_feats
        combined = self.weight.matmul(combined.t())                                                                                                            
        combined = F.relu(combined) # relu W* F^T : (e, 2f) * (2f, n)  --> (e,n)  e: embed_dim
        return combined
