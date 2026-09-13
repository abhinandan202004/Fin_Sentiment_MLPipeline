import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.neo4j_client import Neo4jGraphClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("GraphBuilder")


def build_financial_knowledge_graph(client: Neo4jGraphClient = None) -> Neo4jGraphClient:
    """
    Constructs the standard Financial Knowledge Graph schema:
      Nodes: Company, Sector, Event, Commodity, Country, Article
      Edges: MENTIONS, AFFECTS, BELONGS_TO, SUPPLIES, IMPACTS
    """
    if client is None:
        client = Neo4jGraphClient.get_instance()

    logger.info("Populating Financial Knowledge Graph nodes...")

    # 1. Company Nodes
    companies = [
        ("NVDA", "NVIDIA Corporation", "Semiconductors", 2800000000000),
        ("AAPL", "Apple Inc.", "Consumer Technology", 3400000000000),
        ("MSFT", "Microsoft Corporation", "Cloud Computing", 3100000000000),
        ("AMZN", "Amazon.com Inc.", "Cloud Computing", 1900000000000),
        ("GOOGL", "Alphabet Inc.", "Cloud Computing", 2000000000000),
        ("TSM", "Taiwan Semiconductor Manufacturing Co.", "Semiconductor Foundry", 750000000000),
        ("ASML", "ASML Holding N.V.", "Semiconductor Equipment", 350000000000),
        ("XOM", "Exxon Mobil Corporation", "Energy", 480000000000),
    ]
    for ticker, name, sector, mcap in companies:
        client.add_node(ticker, "Company", {"name": name, "sector": sector, "market_cap": mcap})

    # 2. Sector Nodes
    sectors = [
        ("Semiconductors", "Semiconductors & Semiconductor Equipment"),
        ("Cloud Computing", "Software & Cloud Infrastructure Services"),
        ("Consumer Technology", "Consumer Electronics & Services"),
        ("Energy", "Oil, Gas & Consumable Fuels"),
    ]
    for sec_id, desc in sectors:
        client.add_node(sec_id, "Sector", {"name": sec_id, "description": desc})

    # 3. Commodity Nodes
    commodities = [
        ("Crude_Oil", "Brent & WTI Crude Oil", "Energy"),
        ("Silicon_Wafers", "300mm Semiconductor Substrates", "Materials"),
        ("HBM_Memory", "High Bandwidth Memory (HBM3e)", "Hardware"),
    ]
    for com_id, name, cat in commodities:
        client.add_node(com_id, "Commodity", {"name": name, "category": cat})

    # 4. Country / Region Nodes
    countries = [
        ("USA", "United States of America"),
        ("Taiwan", "Taiwan"),
        ("Middle_East", "Middle East Geopolitical Region"),
    ]
    for c_id, name in countries:
        client.add_node(c_id, "Country", {"name": name})

    # 5. Macro Event Nodes
    events = [
        ("AI_Demand_Surge", "Hyperscale Generative AI Infrastructure Buildout", "macro_positive"),
        ("Middle_East_Conflict", "Regional Geopolitical Conflict & Supply Disruption", "macro_risk"),
        ("Taiwan_Strait_Tension", "Geopolitical Semiconductor Supply Chain Vulnerability", "macro_risk"),
        ("Fed_Rate_Cut", "Federal Reserve Monetary Easing Cycle", "macro_positive"),
    ]
    for ev_id, desc, polarity in events:
        client.add_node(ev_id, "Event", {"description": desc, "polarity": polarity})

    logger.info("Populating causal relationships & multi-hop dependency edges...")

    # Company -> Sector (BELONGS_TO)
    client.add_relationship("NVDA", "Semiconductors", "BELONGS_TO", {
        "strength": 1.0,
        "evidence": "NVIDIA is the leading designer of enterprise AI GPUs and data center accelerators."
    })
    client.add_relationship("TSM", "Semiconductors", "BELONGS_TO", {
        "strength": 1.0,
        "evidence": "TSMC is the dominant global semiconductor manufacturing foundry."
    })
    client.add_relationship("ASML", "Semiconductors", "BELONGS_TO", {
        "strength": 1.0,
        "evidence": "ASML provides critical extreme ultraviolet (EUV) photolithography equipment."
    })
    client.add_relationship("MSFT", "Cloud Computing", "BELONGS_TO", {
        "strength": 1.0,
        "evidence": "Microsoft operates the Azure enterprise cloud computing platform."
    })
    client.add_relationship("AMZN", "Cloud Computing", "BELONGS_TO", {
        "strength": 1.0,
        "evidence": "Amazon Web Services (AWS) is the world's largest public cloud infrastructure provider."
    })
    client.add_relationship("GOOGL", "Cloud Computing", "BELONGS_TO", {
        "strength": 1.0,
        "evidence": "Google Cloud Platform (GCP) provides enterprise compute, TPU, and AI services."
    })
    client.add_relationship("AAPL", "Consumer Technology", "BELONGS_TO", {
        "strength": 1.0,
        "evidence": "Apple designs consumer hardware, operating systems, and services ecosystem."
    })
    client.add_relationship("XOM", "Energy", "BELONGS_TO", {
        "strength": 1.0,
        "evidence": "Exxon Mobil explores, produces, and refines crude oil and petrochemicals."
    })

    # Supply Chain Dependencies (SUPPLIES)
    client.add_relationship("ASML", "TSM", "SUPPLIES", {
        "strength": 0.95,
        "evidence": "Exclusive provider of EUV machines required for TSMC 3nm and 4nm advanced nodes."
    })
    client.add_relationship("TSM", "NVDA", "SUPPLIES", {
        "strength": 0.98,
        "evidence": "TSMC manufactures 100% of NVIDIA Blackwell (B200) and Hopper (H100/H200) GPU silicon."
    })
    client.add_relationship("HBM_Memory", "NVDA", "SUPPLIES", {
        "strength": 0.90,
        "evidence": "SK Hynix and Micron supply critical HBM3e high-bandwidth memory for GPU packaging."
    })
    client.add_relationship("NVDA", "MSFT", "SUPPLIES", {
        "strength": 0.92,
        "evidence": "NVIDIA supplies GPU clusters for Microsoft Azure OpenAI training and inference clusters."
    })
    client.add_relationship("NVDA", "AMZN", "SUPPLIES", {
        "strength": 0.88,
        "evidence": "NVIDIA provides HGX/DGX compute systems for AWS Bedrock and EC2 UltraClusters."
    })

    # Event Impacts (AFFECTS & IMPACTS)
    client.add_relationship("AI_Demand_Surge", "Semiconductors", "AFFECTS", {
        "strength": 0.95,
        "evidence": "Soaring demand for generative AI accelerators drives sector-wide multiple expansion."
    })
    client.add_relationship("AI_Demand_Surge", "NVDA", "AFFECTS", {
        "strength": 0.98,
        "evidence": "Hyperscaler capex growth translates directly into record data center GPU revenue."
    })
    client.add_relationship("Middle_East_Conflict", "Crude_Oil", "IMPACTS", {
        "strength": 0.85,
        "evidence": "Supply disruption risk through Persian Gulf shipping routes boosts crude spot prices."
    })
    client.add_relationship("Crude_Oil", "Energy", "IMPACTS", {
        "strength": 0.90,
        "evidence": "Higher benchmark crude prices immediately expand upstream energy operating margins."
    })
    client.add_relationship("Energy", "XOM", "AFFECTS", {
        "strength": 0.85,
        "evidence": "Energy sector earnings revisions directly elevate Exxon Mobil free cash flow."
    })
    client.add_relationship("Fed_Rate_Cut", "Cloud Computing", "AFFECTS", {
        "strength": 0.75,
        "evidence": "Lower cost of capital encourages enterprise IT cloud migration and corporate capex."
    })

    logger.info("Financial Knowledge Graph construction complete.")
    return client


if __name__ == "__main__":
    client = build_financial_knowledge_graph()
    print("\nKnowledge Graph verification:")
    print(f"Total Nodes: {client.nx_graph.number_of_nodes()}")
    print(f"Total Edges: {client.nx_graph.number_of_edges()}")

    # Verify Multi-hop paths
    print("\nVerifying multi-hop path: ASML -> NVDA:")
    paths = client.find_multi_hop_path("ASML", "NVDA")
    for p in paths:
        print("  " + " -> ".join([f"{u} -[:{rel}]-> {v}" for u, rel, v in p]))

    print("\nVerifying multi-hop path: Middle_East_Conflict -> XOM:")
    paths = client.find_multi_hop_path("Middle_East_Conflict", "XOM")
    for p in paths:
        print("  " + " -> ".join([f"{u} -[:{rel}]-> {v}" for u, rel, v in p]))
