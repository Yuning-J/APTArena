# data_loader.py
import json
import os
import sys
import networkx as nx
import logging
from typing import Dict, List, Optional, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from classes.state import System, Asset, Component, Vulnerability, Connection

logger = logging.getLogger(__name__)


def load_data(file_path: str) -> System:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        system = build_system_from_json(data)
        calculate_centrality(system)
        return system
    except FileNotFoundError:
        logger.error(f"File '{file_path}' not found")
        raise
    except json.JSONDecodeError:
        logger.error(f"File '{file_path}' is not valid JSON")
        raise

def build_system_from_json(data: Dict) -> System:
    system = System()
    asset_dict = {}
    vuln_count = 0
    for i, asset_data in enumerate(data.get("Assets", [])):
        asset_id = str(asset_data.get("asset_id", f"asset_{i}"))
        business_value = asset_data.get("business_value", asset_data.get("criticality_level", 0) * 200000)
        if business_value < 5000 or business_value > 45000:
            logger.warning(f"Invalid business value {business_value} for asset {asset_id}. Using default.")
            business_value = asset_data.get("criticality_level", 3) * 5000
        asset = Asset(
            asset_id=asset_id,
            asset_type=asset_data.get("type", ""),
            name=asset_data.get("name", ""),
            criticality_level=asset_data.get("criticality_level", 0),
            ip_address=asset_data.get("ip_address", "0.0.0.0"),
            mac_address=asset_data.get("mac_address", "00:00:00:00:00:00"),
            business_value=business_value,
            dependency_count=asset_data.get("dependency_count", 0)
        )
        asset_dict[asset_id] = asset
        asset_dict[str(i)] = asset
        for comp_data in asset_data.get("components", []):
            comp_id = str(comp_data.get("id", f"comp_{i}_{len(asset.components)}"))
            component = Component(
                comp_id=comp_id,
                comp_type=comp_data.get("type", ""),
                vendor=comp_data.get("vendor", ""),
                name=comp_data.get("name", ""),
                version=comp_data.get("version", ""),
                embedded_in=comp_data.get("embedded_in")
            )
            for vuln_data in comp_data.get("vulnerabilities", []):
                cve_id = vuln_data.get("cve_id", "")
                if not cve_id or cve_id.lower() == "unknown":
                    logger.warning(f"Skipping invalid CVE '{cve_id}' in asset {asset_id}, component {comp_id}: {vuln_data}")
                    continue
                vulnerability = Vulnerability(
                    cve_id=cve_id,
                    cvss=vuln_data.get("cvss", 0.0),
                    cvssV3Vector=vuln_data.get("cvssV3Vector", ""),
                    scopeChanged=vuln_data.get("scopeChanged", False),
                    likelihood=vuln_data.get("likelihood", 0.0),
                    impact=vuln_data.get("impact", 0.0),
                    exploit=bool(vuln_data.get("exploit", 0)),
                    epss=vuln_data.get("epss", 0.0),
                    ransomWare=bool(vuln_data.get("ransomWare", 0)),
                    component_id=comp_id,
                    is_patched=vuln_data.get("is_patched", False),
                    is_exploited=vuln_data.get("is_exploited", False),
                    cwe_id=vuln_data.get("cwe_id", []),
                    exploitability=vuln_data.get("exploitability", 0.0),
                    mitre_techniques=vuln_data.get("mitre_techniques", []),
                    complexity=vuln_data.get("complexity", "medium")
                )
                component.add_vulnerability(vulnerability)
                vuln_count += 1
            asset.add_component(component)
        if "adjacency_matrix" in asset_data:
            asset.set_adjacency_matrix(asset_data["adjacency_matrix"])
        system.add_asset(asset)

    connections_data = data.get("Connections", [])
    created_connections = 0
    for conn_data in connections_data:
        from_id = str(conn_data.get("source_asset_id", conn_data.get("from", conn_data.get("from_asset_id", ""))))
        to_id = str(conn_data.get("destination_asset_id", conn_data.get("to", conn_data.get("to_asset_id", ""))))
        from_asset = asset_dict.get(from_id)
        to_asset = asset_dict.get(to_id)
        if from_asset and to_asset:
            connection = Connection(
                from_asset=from_asset,
                to_asset=to_asset,
                connection_type=conn_data.get("connection_type", conn_data.get("type", "network")),
                bidirectional=True
            )
            system.add_connection(connection)
            created_connections += 1
        else:
            logger.warning(f"Skipping invalid connection: from={from_id}, to={to_id}")

    logger.info(f"Created {created_connections} connections from connection data")
    if not system.connections:
        logger.info("No connections found in data, generating default connections")
        connections_created = generate_default_connections(system)
        logger.info(f"Generated {connections_created} default connections")

    num_assets = len(system.assets)
    num_connections = len(system.connections)
    logger.info(
        f"Loaded system with {num_assets} assets, {vuln_count} vulnerabilities, and {num_connections} connections")

    has_mitre = any(
        vuln.mitre_techniques for asset in system.assets for comp in asset.components for vuln in comp.vulnerabilities)
    has_exploitability = any(
        vuln.exploitability is not None for asset in system.assets for comp in asset.components for vuln in comp.vulnerabilities)
    if has_mitre and has_exploitability:
        logger.info("Data is enriched with MITRE techniques and exploitability scores")
    else:
        logger.warning("Data may not be fully enriched")

    for asset in system.assets:
        if not hasattr(asset, 'contains_sensitive_data') or asset.contains_sensitive_data is None:
            asset.contains_sensitive_data = asset.criticality_level >= 4
        if not hasattr(asset, 'security_controls') or asset.security_controls is None:
            asset.security_controls = min(5, asset.criticality_level)
        if not hasattr(asset, 'dependency_count') or asset.dependency_count is None:
            asset.dependency_count = sum(1 for conn in system.connections if conn.from_asset == asset)

    return system


def calculate_centrality(system: System) -> None:
    """
    Calculate various centrality measures for assets in the system
    and store them as attributes of the assets.
    
    Args:
        system: System object with assets and connections
    """
    # Create a directed graph from system connections
    G = nx.DiGraph()
    
    # Create a mapping from asset objects to node IDs
    asset_to_node = {}
    node_to_asset = {}
    
    # Add all assets as nodes
    for i, asset in enumerate(system.assets):
        # Use asset ID if available, otherwise generate one
        node_id = asset.asset_id if asset.asset_id else f"asset_{i}"
        G.add_node(node_id)
        asset_to_node[asset] = node_id
        node_to_asset[node_id] = asset
    
    # Add all connections as edges
    edge_count = 0
    for conn in system.connections:
        if conn.from_asset and conn.to_asset:
            try:
                from_id = asset_to_node[conn.from_asset]
                to_id = asset_to_node[conn.to_asset]
                G.add_edge(from_id, to_id)
                edge_count += 1
            except KeyError:
                # Skip connections with assets not in our mapping
                continue
    
    print(f"Created graph with {len(G.nodes())} nodes and {edge_count} edges for centrality calculation")
    
    # If graph is empty or has no edges, assign default centrality
    if len(G) == 0 or G.number_of_edges() == 0:
        print("Warning: Graph has no edges. Setting default centrality values.")
        for asset in system.assets:
            asset.centrality = 0.5
        return
        
    try:
        # Calculate various centrality measures
        in_degree_dict = nx.in_degree_centrality(G)
        out_degree_dict = nx.out_degree_centrality(G)
        
        # Only calculate betweenness if there are enough nodes and edges
        if len(G) > 1 and G.number_of_edges() > 0:
            try:
                betweenness_dict = nx.betweenness_centrality(G, normalized=True, endpoints=True)
            except:
                print("Warning: Betweenness centrality calculation failed. Using default values.")
                betweenness_dict = {node: 0.0 for node in G.nodes()}
        else:
            betweenness_dict = {node: 0.0 for node in G.nodes()}
            
        # Try to calculate PageRank
        try:
            if G.number_of_edges() > 0:
                pagerank_dict = nx.pagerank(G, alpha=0.85)
            else:
                pagerank_dict = {node: 1.0/len(G) for node in G.nodes()}
        except:
            print("Warning: PageRank calculation failed. Using default values.")
            pagerank_dict = {node: 1.0/len(G) for node in G.nodes()}
        
        # Combine centrality measures
        combined_centrality = {}
        for node in G.nodes():
            combined_centrality[node] = (
                in_degree_dict.get(node, 0) +
                out_degree_dict.get(node, 0) +
                betweenness_dict.get(node, 0) +
                pagerank_dict.get(node, 0)
            ) / 4
        
        # Normalize to 0-1 scale
        if combined_centrality:
            max_centrality = max(combined_centrality.values()) if combined_centrality.values() else 1.0
            if max_centrality > 0:  # Avoid division by zero
                for node in combined_centrality:
                    combined_centrality[node] /= max_centrality
            else:
                for node in combined_centrality:
                    combined_centrality[node] = 0.5
        
        # Store centrality values on assets
        for node, centrality in combined_centrality.items():
            if node in node_to_asset:
                asset = node_to_asset[node]
                asset.centrality = centrality
                
                # Also store individual centrality measures for potential future use
                asset.in_degree_centrality = in_degree_dict.get(node, 0)
                asset.out_degree_centrality = out_degree_dict.get(node, 0)
                asset.betweenness_centrality = betweenness_dict.get(node, 0)
                asset.pagerank = pagerank_dict.get(node, 0)
                
        # Set default centrality for any assets not in the graph
        for asset in system.assets:
            if not hasattr(asset, 'centrality'):
                asset.centrality = 0.5
                asset.in_degree_centrality = 0
                asset.out_degree_centrality = 0
                asset.betweenness_centrality = 0
                asset.pagerank = 0.25
                
        # Print centrality summary
        centrality_values = [asset.centrality for asset in system.assets]
        avg_centrality = sum(centrality_values) / len(centrality_values) if centrality_values else 0
        max_centrality_asset = max(system.assets, key=lambda a: getattr(a, 'centrality', 0))
        print(f"Centrality calculation complete. Average: {avg_centrality:.2f}, "
              f"Max: {max_centrality_asset.centrality:.2f} ({max_centrality_asset.name})")
              
        return
        
    except Exception as e:
        # Fallback if centrality calculation fails
        print(f"Warning: Centrality calculation failed: {e}")
        for asset in system.assets:
            asset.centrality = 0.5

def generate_default_connections(system: System) -> int:
    """
    Generate default connections between assets when no connections are specified.
    
    Args:
        system: The system to generate connections for
        
    Returns:
        int: Number of connections created
    """
    connections_created = 0
    
    # Create a dictionary of assets by type for easier lookup
    assets_by_type = {}
    for asset in system.assets:
        asset_type = asset.type.lower()
        if asset_type not in assets_by_type:
            assets_by_type[asset_type] = []
        assets_by_type[asset_type].append(asset)
    
    # Define connection patterns based on typical ICS network architecture
    connection_patterns = [
        # VPN Server connects to all workstations
        ("vpn server", ["workstation", "engineering workstation"], "network"),
        
        # Web Server connects to historians and databases
        ("web server", ["historian database"], "data"),
        
        # Workstations connect to HMI and historians
        ("workstation", ["human machine interface", "hmi", "historian database"], "management"),
        
        # Engineering workstations connect to PLCs
        ("engineering workstation", ["programmable logic controller", "plc"], "control"),
        
        # HMI connects to PLCs
        ("human machine interface", ["programmable logic controller", "plc"], "control"),
        ("hmi", ["programmable logic controller", "plc"], "control")
    ]
    
    # Create connections based on patterns
    for from_type, to_types, conn_type in connection_patterns:
        if from_type in assets_by_type:
            from_assets = assets_by_type[from_type]
            
            for to_type in to_types:
                if to_type in assets_by_type:
                    to_assets = assets_by_type[to_type]
                    
                    for from_asset in from_assets:
                        for to_asset in to_assets:
                            connection = Connection(
                                from_asset=from_asset,
                                to_asset=to_asset,
                                connection_type=conn_type
                            )
                            system.add_connection(connection)
                            connections_created += 1

    return connections_created
