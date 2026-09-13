import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List, Dict, Any
from graph.neo4j_client import Neo4jGraphClient
from graph.graph_builder import build_financial_knowledge_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("GraphQueries")


class GraphQueryEngine:
    _instance = None

    def __init__(self):
        self.client = build_financial_knowledge_graph()

    @classmethod
    def get_instance(cls) -> "GraphQueryEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_ticker_graph_impacts(self, ticker: str) -> List[str]:
        """
        Retrieves qualitative causal and supply chain impacts for the ticker
        from the Knowledge Graph.
        """
        subgraph = self.client.find_connected_subgraph(ticker)
        impact_statements = []

        for item in subgraph:
            if item["direction"] == "upstream":
                # item["source"] affects/supplies ticker
                rel = item["relation"]
                src = item["source"]
                ev = item.get("evidence", "")
                if rel == "SUPPLIES":
                    impact_statements.append(f"Supply Chain Dependency: {src} supplies {ticker} ({ev})")
                elif rel == "AFFECTS":
                    impact_statements.append(f"Macro / Sector Catalyst: {src} affects {ticker} ({ev})")
                elif rel == "BELONGS_TO":
                    impact_statements.append(f"Sector Membership: {ticker} belongs to {src} sector ({ev})")
                else:
                    impact_statements.append(f"{src} -[:{rel}]-> {ticker} ({ev})")
            else:
                # ticker affects/supplies item["target"]
                rel = item["relation"]
                tgt = item["target"]
                ev = item.get("evidence", "")
                if rel == "SUPPLIES":
                    impact_statements.append(f"Downstream Customer Demand: {ticker} supplies key systems to {tgt} ({ev})")
                elif rel == "BELONGS_TO":
                    impact_statements.append(f"Sector Peer Correlation: {ticker} anchored within {tgt} ({ev})")

        # Also search for sector-level upstream events
        company_node = self.client.nx_graph.nodes.get(ticker, {})
        sector = company_node.get("sector")
        if sector and sector in self.client.nx_graph:
            sector_subgraph = self.client.find_connected_subgraph(sector)
            for s_item in sector_subgraph:
                if s_item["direction"] == "upstream" and s_item["relation"] == "AFFECTS":
                    src = s_item["source"]
                    ev = s_item.get("evidence", "")
                    statement = f"Sector-wide Catalyst: {src} strengthening {sector} sector ({ev})"
                    if statement not in impact_statements:
                        impact_statements.append(statement)

        return impact_statements

    def explain_multi_hop_chain(self, source_node: str, target_node: str) -> str:
        """Explains how an external event cascades into the target stock."""
        paths = self.client.find_multi_hop_path(source_node, target_node)
        if not paths:
            return f"No direct or multi-hop path detected between {source_node} and {target_node}."

        first_path = paths[0]
        steps = []
        for u, rel, v in first_path:
            edge_data = self.client.nx_graph.get_edge_data(u, v)
            rel_name = rel
            ev = ""
            if edge_data and rel in edge_data:
                ev = edge_data[rel].get("evidence", "")
            steps.append(f"{u} -[:{rel_name}]-> {v} ({ev})" if ev else f"{u} -[:{rel_name}]-> {v}")

        return " -> ".join(steps)


if __name__ == "__main__":
    engine = GraphQueryEngine.get_instance()
    impacts = engine.get_ticker_graph_impacts("NVDA")
    print("\nNVDA Knowledge Graph Impacts:")
    for imp in impacts:
        print(f" - {imp}")

    print("\nCausal chain (Middle_East_Conflict -> XOM):")
    print(engine.explain_multi_hop_chain("Middle_East_Conflict", "XOM"))

    print("\nCausal chain (ASML -> NVDA):")
    print(engine.explain_multi_hop_chain("ASML", "NVDA"))
