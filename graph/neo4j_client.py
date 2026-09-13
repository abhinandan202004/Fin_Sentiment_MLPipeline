import logging
import os
from typing import Dict, Any, List, Optional, Tuple
import networkx as nx
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Neo4jClient")


class Neo4jGraphClient:
    """
    Dual-engine graph client:
    - Connects to Neo4j database via bolt:// if live.
    - Transparently falls back to an in-memory NetworkX DiGraph if Neo4j is offline.
    """
    _instance: Optional["Neo4jGraphClient"] = None

    def __init__(
        self,
        uri: str = NEO4J_URI or "bolt://localhost:7687",
        user: str = NEO4J_USER or "neo4j",
        password: str = NEO4J_PASSWORD or "password",
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.is_neo4j_active = False
        self.driver = None

        # NetworkX in-memory fallback graph
        self.nx_graph = nx.MultiDiGraph()

        # Try connecting to Neo4j
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            with driver.session() as session:
                session.run("RETURN 1 AS test")
            self.driver = driver
            self.is_neo4j_active = True
            logger.info(f"Connected to live Neo4j instance at {self.uri}")
        except Exception as e:
            self.is_neo4j_active = False
            logger.info(
                f"Neo4j instance not reachable at {self.uri} ({e}). "
                "Active engine: In-Memory MultiDiGraph (NetworkX fallback with zero data loss)."
            )

    @classmethod
    def get_instance(cls) -> "Neo4jGraphClient":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def add_node(self, node_id: str, label: str, properties: Optional[Dict[str, Any]] = None):
        """Adds or updates a node in the graph."""
        props = properties or {}
        props["label"] = label
        props["id"] = node_id

        # Update in-memory graph
        self.nx_graph.add_node(node_id, **props)

        # Update Neo4j if active
        if self.is_neo4j_active and self.driver:
            try:
                with self.driver.session() as session:
                    prop_str = ", ".join([f"n.{k} = ${k}" for k in props.keys()])
                    query = f"MERGE (n:{label} {{id: $id}}) SET {prop_str}"
                    session.run(query, **props)
            except Exception as e:
                logger.warning(f"Neo4j add_node error: {e}")

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ):
        """Adds an edge with metadata between two nodes."""
        props = properties or {}
        props["relation"] = relation_type
        props.setdefault("strength", 1.0)
        props.setdefault("evidence", "Direct market relationship")

        # Update in-memory graph
        self.nx_graph.add_edge(source_id, target_id, key=relation_type, **props)

        # Update Neo4j if active
        if self.is_neo4j_active and self.driver:
            try:
                with self.driver.session() as session:
                    prop_str = ", ".join([f"r.{k} = ${k}" for k in props.keys()])
                    query = (
                        f"MATCH (a {{id: $source_id}}), (b {{id: $target_id}}) "
                        f"MERGE (a)-[r:{relation_type}]->(b) "
                        f"SET {prop_str}"
                    )
                    session.run(query, source_id=source_id, target_id=target_id, **props)
            except Exception as e:
                logger.warning(f"Neo4j add_relationship error: {e}")

    def find_connected_subgraph(
        self,
        target_id: str,
        max_depth: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Discovers incoming and outgoing causal chains connected to target node.
        """
        results = []
        if target_id not in self.nx_graph:
            return results

        # In-memory graph traversal (works in both Neo4j and offline modes)
        for predecessor in self.nx_graph.predecessors(target_id):
            edge_data = self.nx_graph.get_edge_data(predecessor, target_id)
            for key, data in edge_data.items():
                p_label = self.nx_graph.nodes[predecessor].get("label", "Node")
                p_name = self.nx_graph.nodes[predecessor].get("name", predecessor)
                results.append({
                    "source": p_name,
                    "source_type": p_label,
                    "target": target_id,
                    "relation": key,
                    "strength": data.get("strength", 1.0),
                    "evidence": data.get("evidence", ""),
                    "direction": "upstream",
                })

        for successor in self.nx_graph.successors(target_id):
            edge_data = self.nx_graph.get_edge_data(target_id, successor)
            for key, data in edge_data.items():
                s_label = self.nx_graph.nodes[successor].get("label", "Node")
                s_name = self.nx_graph.nodes[successor].get("name", successor)
                results.append({
                    "source": target_id,
                    "relation": key,
                    "target": s_name,
                    "target_type": s_label,
                    "strength": data.get("strength", 1.0),
                    "evidence": data.get("evidence", ""),
                    "direction": "downstream",
                })

        return results

    def find_multi_hop_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 4,
    ) -> List[List[Tuple[str, str, str]]]:
        """
        Finds multi-hop paths between source and target:
        [(nodeA, RELATION, nodeB), (nodeB, RELATION, nodeC), ...]
        """
        if source_id not in self.nx_graph or target_id not in self.nx_graph:
            return []

        paths = []
        try:
            for p in nx.all_simple_paths(self.nx_graph, source=source_id, target=target_id, cutoff=max_hops):
                path_edges = []
                for i in range(len(p) - 1):
                    u, v = p[i], p[i + 1]
                    edge_dict = self.nx_graph.get_edge_data(u, v)
                    rel_name = list(edge_dict.keys())[0] if edge_dict else "RELATED_TO"
                    path_edges.append((u, rel_name, v))
                paths.append(path_edges)
        except Exception as e:
            logger.warning(f"Error finding paths: {e}")

        return paths

    def close(self):
        if self.driver:
            self.driver.close()
