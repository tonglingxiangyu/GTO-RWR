import numpy as np
from neo4j import GraphDatabase

# 加载节点和边的数据
nodes = np.load("nodes.npy", allow_pickle=True)
print('nodes:', nodes.shape)
edge_feat = np.load("edge_feat.npy", allow_pickle=True)
print('edge_feat:', edge_feat.shape)

adj = np.load('adj.npy', allow_pickle=True)
print('adj:', adj.shape)

dips = np.load('unique_dips.npy', allow_pickle=True)
print('dips:', dips.shape)


# 连接到 Neo4j 数据库，确保使用正确的 URL、用户名和密码
uri = "bolt://localhost:7687"
# user = "neo4j"
# password = "123456789"
user = "neo4j"
password = "12345678"
driver = GraphDatabase.driver(uri, auth=(user, password))
# g = Graph('http://localhost:7474', auth=('neo4j', 'ABC123'))

# 定义一个函数来清空数据库
def clear_db(tx):
    tx.run("MATCH (n) DETACH DELETE n")
# 清空边
def clear_edge(tx):
    tx.run("MATCH ()-[r]->() DELETE r")

# 定义一个函数来创建节点
def create_node(tx, node_id):
    #id = node_id.split(':')[0]
    port = node_id.split(':')[1]
    #tx.run("CREATE (:Node {id: $node_id})", node_id=node_id)
    tx.run("CREATE (:Node {id: $node_id, port: $port})", node_id=node_id, port=port)

# 定义一个函数来创建边
def create_edge(tx, src, dst, edge_attributes):
    query = """
    MATCH (a:Node {id: $src})
    MATCH (b:Node {id: $dst})
    CREATE (a)-[:CONNECTS {attributes: $edge_attributes}]->(b)
    """
    tx.run(query, src=src, dst=dst, edge_attributes=edge_attributes)

# 连接具有相同 IP 地址的节点
# def connect_same_ip_nodes(tx, ip):
#     query = """
#     MATCH (p1), (p2)
#     WHERE p1.id STARTS WITH $ip AND p2.id STARTS WITH $ip AND p1 <> p2 AND p1.id < p2.id
#     MERGE (p1)-[:CONNECTED]->(p2)
#     """
#     tx.run(query, ip=ip)
#     print(ip)

# 按端口排序：首先，对具有相同 IP 前缀的所有节点按其端口号进行排序。
# 连接相邻节点：然后，创建连接，使得每个节点都与按端口排序的下一个节点相连。
# 连接最大和最小端口的节点：最后，特别处理以连接端口号最大和最小的节点。
def connect_same_ip_nodes(tx, ip):
    # query = """
    # MATCH (p1:Node), (p2:Node)
    # WHERE p1.id STARTS WITH $ip AND p2.id STARTS WITH $ip
    # WITH p1, p2
    # ORDER BY p1.port, p2.port
    # WITH COLLECT(p1) AS nodes
    # FOREACH (i IN RANGE(0, LENGTH(nodes)-2) |
    #     MERGE (nodes[i])-[:CONNECTED]->(nodes[i+1]))
    # MERGE (nodes[LENGTH(nodes)-1])-[:CONNECTED]->(nodes[0])
    # """
    query = """
    MATCH (p:Node)
    WHERE p.id STARTS WITH $ip
    WITH p
    ORDER BY p.port
    WITH COLLECT(p) AS nodes
    WITH nodes, nodes[0] AS firstNode, SIZE(nodes) AS len
    WITH nodes, firstNode, CASE WHEN len > 1 THEN nodes[len - 1] END AS lastNode, len
    CALL apoc.nodes.link(nodes, 'CONNECTED')
    WITH firstNode, lastNode, len
    WHERE len > 1
    MERGE (lastNode)-[:CONNECTED]->(firstNode)
    """
    tx.run(query, ip=ip)
    print(ip)


# # 连接到数据库并清空旧数据
# with driver.session() as session:
#     session.write_transaction(clear_db)
# print('clear')

# # 清空边
# with driver.session() as session:
#     session.write_transaction(clear_edge)

# # 将节点添加到数据库
# for node_id in nodes:
#     with driver.session() as session:
#         session.execute_write(create_node, node_id)
# print("node over")

# # 将边添加到数据库
# start = 0
# for i, edge in enumerate(adj):
#     if i < start:
#         continue
#     src = edge[0]
#     dst = edge[1]
#     attributes = edge_feat[i]

#     with driver.session() as session:
#         session.execute_write(create_edge, src, dst, attributes)
#     # print(i)
# print("edge over")

# 连接具有相同 IP 的节点
inode = 0
for ip in dips:
    inode += 1
    ip_only = ip.split(':')[0]
    with driver.session() as session:
        session.execute_write(connect_same_ip_nodes, ip_only)
    print(inode)
print("over")


# 关闭数据库连接
driver.close()