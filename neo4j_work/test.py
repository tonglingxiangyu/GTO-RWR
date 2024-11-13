from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
# user = "neo4j"
# password = "123456789"
user = "neo4j"
password = "12345678"
driver = GraphDatabase.driver(uri, auth=(user, password))

# 创建第一个子图
tx = driver.session().begin_transaction()
tx.run("""
CREATE (n1:Node {ip: '192.168.1.1', port: '8080'})
CREATE (n2:Node {ip: '192.168.1.2', port: '8081'}) 
CREATE (n3:Node {ip: '192.168.1.3', port: '8081'}) 
CREATE (n2)-[:CONNECTED_TO]->(n1)
CREATE (n3)-[:CONNECTED_TO]->(n1)
""")
tx.commit()

# 创建第二个子图,其中一个节点ip和第一个子图相同
tx = driver.session().begin_transaction()
tx.run("""
CREATE (n3:Node {ip: '192.168.1.1', port: '8082'})
CREATE (n4:Node {ip: '192.168.1.4', port: '8083'})  
CREATE (n5:Node {ip: '192.168.1.5', port: '8083'})
CREATE (n4)-[:CONNECTED_TO]->(n3) 
CREATE (n5)-[:CONNECTED_TO]->(n3)  
""")
tx.commit()

# 创建第三个子图
tx = driver.session().begin_transaction()  
tx.run("""
CREATE (n5:Node {ip: '192.168.1.1', port: '8084'})
CREATE (n6:Node {ip: '192.168.1.6', port: '8085'})
CREATE (n7:Node {ip: '192.168.1.7', port: '8085'})
CREATE (n6)-[:CONNECTED_TO]->(n5)
CREATE (n7)-[:CONNECTED_TO]->(n5)
""")
tx.commit()