# attack_graph.py
import random
import logging
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional
from classes.state import State, Asset, Component, Vulnerability, System, KillChainStage
from classes.mitre import MitreMapper, MitreTTP, mitre_to_ckc_mapping
import numpy as np
from scipy.stats import multivariate_normal
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for pygraphviz availability
try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
    PYGRAPHVIZ_AVAILABLE = True
except ImportError:
    PYGRAPHVIZ_AVAILABLE = False
    logger.warning("pygraphviz is not installed. Falling back to spring_layout for attack tree visualization.")

class AttackNode:
    """
    Represents a node in the attack graph.
    A node can be an asset, a vulnerability, or an abstract concept like 'external network'.
    """
    def __init__(self, node_id: str, node_type: str, name: str, **attributes):
        self.id = node_id
        self.type = node_type  # 'asset', 'vulnerability', 'abstract'
        self.name = name
        self.attributes = attributes
        self.exploited = False
        self.patched = False

    def __repr__(self):
        return f"AttackNode({self.id}, {self.type}, {self.name})"

class AttackEdge:
    """
    Represents an edge in the attack graph.
    An edge represents a potential attack vector between nodes.
    """
    def __init__(self, source_id: str, target_id: str, edge_type: str, cost: float, probability: float, **attributes):
        self.source_id = source_id
        self.target_id = target_id
        self.type = edge_type  # 'exploit', 'lateral_movement', 'network_access'
        self.cost = cost  # Cost to the attacker to traverse this edge
        self.probability = probability  # Probability of success
        self.attributes = attributes
        self.traversed = False

    def __repr__(self):
        return f"AttackEdge({self.source_id} -> {self.target_id}, {self.type}, cost={self.cost:.2f}, prob={self.probability:.2f})"


class AttackGraph:
    def __init__(self, system_graph=None, mitre_mapper=None, cwe_canfollow_path=None):
        self.system_graph = system_graph
        self.graph = None
        self.high_value_targets = []
        self.entry_nodes = []
        self.asset_nodes = []
        self.epsilon = 0.01
        self.cwe_canfollow_data = []
        self.vuln_correlations = {}  # Store correlation matrices for vulnerabilities
        if cwe_canfollow_path:
            self.load_cwe_canfollow(cwe_canfollow_path)
        self._load_mitre_mappings(mitre_mapper)

    def _load_mitre_mappings(self, mitre_mapper=None):
        if mitre_mapper:
            self.technique_to_tactic = mitre_mapper.generate_flat_technique_to_tactic()
            self.tactic_to_techniques = mitre_mapper.tactic_to_techniques
        else:
            from .mitre import technique_to_tactic
            self.technique_to_tactic = technique_to_tactic
            self.tactic_to_techniques = defaultdict(list)
            for technique_id, tactic in technique_to_tactic.items():
                self.tactic_to_techniques[tactic].append(technique_id)

    def load_cwe_canfollow(self, file_path="../data/CTI/raw/canfollow.json", force_reload=False):
        import json
        import os
        if self.cwe_canfollow_data and not force_reload:
            return self.cwe_canfollow_data
        if not os.path.exists(file_path):
            logger.warning(f"CWE CanFollow file not found: {file_path}")
            self.cwe_canfollow_data = []
            return self.cwe_canfollow_data
        try:
            with open(file_path, 'r') as f:
                self.cwe_canfollow_data = json.load(f)
            logger.info(f"Loaded {len(self.cwe_canfollow_data)} CWE CanFollow relationships from {file_path}")
            return self.cwe_canfollow_data
        except Exception as e:
            logger.error(f"Error loading CWE CanFollow data: {e}")
            self.cwe_canfollow_data = []
            return self.cwe_canfollow_data

    def _identify_internet_connections(self):
        """Identify internet connections in the graph to mark internet-facing assets."""
        for u, v, data in self.graph.edges(data=True):
            if data.get('src_ip') == '0.0.0.0' or data.get('src_subnet') == '0.0.0.0/0':
                if self.graph.nodes[u].get('type') == 'asset':
                    self.graph.edges[u, v]['internet_facing'] = True
                    self.graph.nodes[u]['is_internet_facing'] = True
            if data.get('dst_ip') == '0.0.0.0' or data.get('dst_subnet') == '0.0.0.0/0':
                if self.graph.nodes[v].get('type') == 'asset':
                    self.graph.edges[u, v]['internet_facing'] = True
                    self.graph.nodes[v]['is_internet_facing'] = True
            # Exclude OT network assets (192.168.3.0/24)
            if 'dst_ip' in data and data['dst_ip'].startswith('192.168.3.'):
                self.graph.nodes[v]['is_internet_facing'] = False
                data['internet_facing'] = False

    def build_attack_graph(self, system_graph=None, initial_ckc_stage=KillChainStage.RECONNAISSANCE):
        if system_graph:
            self.system_graph = system_graph
        if not self.system_graph:
            raise ValueError("System graph not provided")
        self.graph = self.system_graph.copy()

        self._identify_internet_connections()

        # Validate node attributes
        for node_id, data in self.graph.nodes(data=True):
            logger.debug(f"Node: {node_id}, Type: {data.get('type')}, Attributes: {data}")
            if 'type' not in data:
                logger.warning(f"Node {node_id} missing 'type' attribute, setting to 'unknown'")
                data['type'] = 'unknown'
            if 'ckc_stage' not in data:
                data['ckc_stage'] = initial_ckc_stage

        for u, v, data in self.graph.edges(data=True):
            logger.debug(f"Edge: {u} -> {v}, Type: {data.get('type')}, Attributes: {data}")

        self._build_component_to_asset_mapping()
        self._assign_ckc_stages(initial_ckc_stage)
        self._compute_vulnerability_correlations()
        self._adjust_edge_weights_for_ckc()
        self.precompute_mappings()
        if self.cwe_canfollow_data:
            self.incorporate_cwe_canfollow(self.cwe_canfollow_data)
        self.identify_high_value_targets()
        self.identify_entry_nodes()

        # Log graph summary
        logger.debug(f"Built attack graph: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
        logger.debug(f"Entry nodes: {self.entry_nodes}")
        logger.debug(f"High-value targets: {self.high_value_targets}")
        return self.graph

    def _build_component_to_asset_mapping(self):
        self.component_to_asset = {}
        for node_id, data in self.system_graph.nodes(data=True):
            if data.get('type') == 'asset':
                for component in data.get('components', []):
                    comp_id = component.get('id')
                    if comp_id:
                        self.component_to_asset[comp_id] = node_id

    def _compute_vulnerability_correlations(self):
        """Compute correlation matrices for vulnerabilities on the same asset or with CWE relationships."""
        vuln_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'vulnerability']
        self.vuln_correlations = {}
        for asset_id in self.asset_nodes:
            asset_vulns = [n for n in vuln_nodes if self.graph.nodes[n].get('asset_id') == asset_id]
            if len(asset_vulns) > 1:
                # Assume moderate positive correlation (0.3) for vulnerabilities on the same asset
                corr_matrix = np.ones((len(asset_vulns), len(asset_vulns))) * 0.3
                np.fill_diagonal(corr_matrix, 1.0)
                self.vuln_correlations[asset_id] = {'nodes': asset_vulns, 'matrix': corr_matrix}
        for cwe_data in self.cwe_canfollow_data:
            cwe_id = cwe_data.get('cwe_id')
            canfollow_cwes = cwe_data.get('cwe_canfollow', [])
            cwe_vulns = [n for n in vuln_nodes if cwe_id in (self.graph.nodes[n].get('cwe_id', []) if isinstance(self.graph.nodes[n].get('cwe_id'), list) else [self.graph.nodes[n].get('cwe_id')])]
            for follow_cwe in canfollow_cwes:
                follow_vulns = [n for n in vuln_nodes if follow_cwe in (self.graph.nodes[n].get('cwe_id', []) if isinstance(self.graph.nodes[n].get('cwe_id'), list) else [self.graph.nodes[n].get('cwe_id')])]
                if cwe_vulns and follow_vulns:
                    all_vulns = list(set(cwe_vulns + follow_vulns))
                    corr_matrix = np.ones((len(all_vulns), len(all_vulns))) * 0.5  # Higher correlation for CWE relationships
                    np.fill_diagonal(corr_matrix, 1.0)
                    self.vuln_correlations[f"cwe_{cwe_id}_{follow_cwe}"] = {'nodes': all_vulns, 'matrix': corr_matrix}

    def _assign_ckc_stages(self, initial_ckc_stage):
        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'entry':
                data['ckc_stage'] = KillChainStage.DELIVERY
            elif data.get('type') == 'vulnerability':
                techniques = data.get('techniques', [])
                stages = []
                for technique in techniques:
                    tactic = self.technique_to_tactic.get(technique)
                    if tactic:
                        stage = mitre_to_ckc_mapping.get(tactic)
                        if stage:
                            stages.append(stage)
                if stages:
                    stage_counts = {}
                    for stage in stages:
                        stage_counts[stage] = stage_counts.get(stage, 0) + 1
                    data['ckc_stage'] = max(stage_counts.items(), key=lambda x: x[1])[0]
                else:
                    data['ckc_stage'] = KillChainStage.EXPLOITATION
            elif data.get('type') == 'asset':
                data['ckc_stage'] = KillChainStage.ACTIONS_ON_OBJECTIVES
            else:
                data['ckc_stage'] = initial_ckc_stage

    def _adjust_edge_weights_for_ckc(self):
        for u, v, data in self.graph.edges(data=True):
            u_stage = self.graph.nodes[u].get('ckc_stage')
            v_stage = self.graph.nodes[v].get('ckc_stage')
            if u_stage and v_stage:
                stage_diff = abs(v_stage.value - u_stage.value)
                if v_stage.value == u_stage.value + 1:
                    data['weight'] = data.get('weight', 1.0) * 0.8
                elif v_stage.value > u_stage.value + 1:
                    data['weight'] = data.get('weight', 1.0) * (1.0 + 0.2 * stage_diff)
                elif v_stage.value < u_stage.value:
                    data['weight'] = data.get('weight', 1.0) * (1.0 + 0.5 * stage_diff)

    def precompute_mappings(self):
        self.technique_to_tactics = {}
        for tactic, techniques in self.tactic_to_techniques.items():
            for technique in techniques:
                if technique not in self.technique_to_tactics:
                    self.technique_to_tactics[technique] = []
                self.technique_to_tactics[technique].append(tactic)
        for node_id, data in self.graph.nodes(data=True):
            if data.get('type') == 'vulnerability':
                techniques = data.get('techniques', [])
                tactics = set()
                for technique in techniques:
                    tactic = self.technique_to_tactic.get(technique)
                    if tactic:
                        tactics.add(tactic)
                    tech_tactics = self.technique_to_tactics.get(technique, [])
                    tactics.update(tech_tactics)
                self.graph.nodes[node_id]['associated_tactics'] = list(tactics)

    def incorporate_cwe_canfollow(self, cwe_canfollow_data):
        cwe_to_vuln_nodes = defaultdict(list)
        for node_id, data in self.graph.nodes(data=True):
            if data.get('type') == 'vulnerability' and 'cwe_id' in data:
                cwe_ids = data['cwe_id'] if isinstance(data['cwe_id'], list) else [data['cwe_id']]
                for cwe_id in cwe_ids:
                    cwe_to_vuln_nodes[cwe_id].append(node_id)
        for cwe_data in cwe_canfollow_data:
            cwe_id = cwe_data.get('cwe_id')
            canfollow_cwes = cwe_data.get('cwe_canfollow', [])
            if not canfollow_cwes:
                continue
            vuln_nodes = cwe_to_vuln_nodes.get(cwe_id, [])
            for vuln_node in vuln_nodes:
                self.graph.nodes[vuln_node]['canfollow_cwes'] = canfollow_cwes
                # Bayesian update for edges involving this vulnerability
                for u, v, edge_data in self.graph.edges(vuln_node, data=True):
                    if self.graph.nodes[v].get('type') == 'vulnerability':
                        target_cwe = self.graph.nodes[v].get('cwe_id', [])
                        if any(cwe in canfollow_cwes for cwe in (target_cwe if isinstance(target_cwe, list) else [target_cwe])):
                            source_epss = self.graph.nodes[vuln_node].get('epss', 0.1)
                            target_epss = self.graph.nodes[v].get('epss', 0.1)
                            conditional_prob = 0.7 * source_epss * target_epss + 0.3
                            edge_data['probability'] = min(1.0, edge_data.get('probability', 0.1) * (1 + conditional_prob))

    def _calculate_path_probability(self, path):
        """
        Calculate path probability with logarithmic aggregation and robust correlation handling.
        """
        logger.debug(f"Calculating path probability for path: {path}")
        log_path_prob = 0.0  # Use log probabilities to prevent underflow
        vuln_indices = []
        vuln_nodes = []

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.graph.has_edge(u, v):
                edge_prob = self.graph.edges[u, v].get('probability', 0.5)
                if not 0 < edge_prob <= 1:
                    logger.warning(f"Invalid edge probability {edge_prob} for {u} -> {v}, using 0.5")
                    edge_prob = 0.5
                log_path_prob += np.log(max(self.epsilon, edge_prob))
                if self.graph.nodes[u].get('type') == 'vulnerability':
                    vuln_indices.append(i)
                    vuln_nodes.append(u)

        if len(vuln_nodes) > 1:
            for asset_id, corr_data in self.vuln_correlations.items():
                if all(v in corr_data['nodes'] for v in vuln_nodes):
                    try:
                        indices = [corr_data['nodes'].index(v) for v in vuln_nodes]
                        sub_matrix = corr_data['matrix'][np.ix_(indices, indices)]
                        # Validate correlation matrix
                        if not np.all(np.isfinite(sub_matrix)) or np.any(np.diag(sub_matrix) != 1):
                            logger.warning(f"Invalid correlation matrix for {asset_id}, skipping adjustment")
                            continue
                        mean = [0] * len(vuln_nodes)
                        mv_dist = multivariate_normal(mean=mean, cov=sub_matrix, allow_singular=True)
                        probs = [max(0.01, min(1.0, self.graph.nodes[v].get('epss', 0.1))) for v in vuln_nodes]
                        adjustment = mv_dist.pdf(probs)
                        if adjustment <= 0 or np.isnan(adjustment) or np.isinf(adjustment):
                            logger.debug(f"Invalid PDF adjustment for {asset_id}, using 1.0")
                            adjustment = 1.0
                        else:
                            prod_probs = np.prod(
                                [max(0.01, multivariate_normal.pdf([p], mean=[0], cov=1.0)) for p in probs])
                            if prod_probs > 0:
                                adjustment = min(10.0, max(0.1, adjustment / prod_probs))
                            else:
                                adjustment = 1.0
                        log_path_prob += np.log(max(self.epsilon, adjustment))
                        logger.debug(f"Applied correlation adjustment: {adjustment}")
                    except Exception as e:
                        logger.error(f"Error in correlation adjustment for {asset_id}: {e}", exc_info=True)
                        continue

        path_prob = np.exp(log_path_prob)
        final_prob = max(self.epsilon, min(1.0, path_prob))
        logger.debug(f"Final path probability: {final_prob}")
        return final_prob

    def _vulnerabilities_are_related(self, source_vuln_data, target_vuln_data):
        """
        Check if two vulnerabilities are related (on same or connected assets).

        Args:
            source_vuln_data: Data for source vulnerability
            target_vuln_data: Data for target vulnerability

        Returns:
            bool: True if vulnerabilities are on same or connected assets
        """
        # Get the component/asset IDs for both vulnerabilities
        source_component = source_vuln_data.get('component_id')
        target_component = target_vuln_data.get('component_id')

        # If on same component, they're related
        if source_component == target_component:
            return True

        # If components are on connected assets, they're related
        source_asset = self.component_to_asset.get(source_component)
        target_asset = self.component_to_asset.get(target_component)

        if source_asset and target_asset:
            # Check if assets are connected in the system graph
            try:
                path_exists = nx.has_path(self.system_graph, source_asset, target_asset)
                return path_exists
            except nx.NetworkXError:
                # Handle case where nodes might not exist
                return False

        return False

    def identify_high_value_targets(self):
        """Identify high-value target assets based on criticality and business value."""
        self.high_value_targets = []
        self.asset_nodes = []

        for node_id, data in self.graph.nodes(data=True):
            if data.get('type') == 'asset':
                self.asset_nodes.append(node_id)

                # Check if this is a high-value target
                if (data.get('criticality_level', 0) >= 4 or
                        data.get('business_value', 0) >= 20000):
                    self.high_value_targets.append(node_id)

    def old_identify_entry_nodes(self):
        """
        Identify entry point nodes in the attack graph with priority on internet-facing assets.
        """
        self.entry_nodes = []
        internet_facing = []

        # First pass: Explicitly identify internet-facing assets from connections
        for node_id, data in self.graph.nodes(data=True):
            if data.get('type') == 'asset':
                # Check if this asset has internet connectivity based on connections
                for u, v, edge_data in self.graph.edges(data=True):
                    if (u == node_id or v == node_id) and edge_data.get('internet_facing', False):
                        internet_facing.append(node_id)
                        data['is_internet_facing'] = True
                        data['is_entry_point'] = True
                        data['is_compromised'] = True  # Internet-facing assets are assumed compromisable
                        break

        # Second pass: Check for asset types suggesting internet exposure
        for node_id, data in self.graph.nodes(data=True):
            if data.get('type') == 'asset' and node_id not in internet_facing:
                asset_type = data.get('type', '').lower()
                if ('web' in asset_type or 'firewall' in asset_type or 'external' in asset_type
                        or 'gateway' in asset_type or 'proxy' in asset_type):
                    internet_facing.append(node_id)
                    data['is_internet_facing'] = True
                    data['is_entry_point'] = True
                    data['is_compromised'] = True  # Internet-facing assets are assumed compromisable

        # Add all internet-facing assets as entry nodes
        self.entry_nodes.extend(internet_facing)

        # If still no entry nodes identified, use fallback methods
        if not self.entry_nodes:
            # Use nodes marked as 'entry' type
            for node_id, data in self.graph.nodes(data=True):
                if data.get('type') == 'entry':
                    self.entry_nodes.append(node_id)

            # If still no entry nodes, use nodes with no incoming edges
            if not self.entry_nodes:
                for node_id in self.graph.nodes():
                    if (self.graph.in_degree(node_id) == 0 and
                            self.graph.out_degree(node_id) > 0 and
                            self.graph.nodes[node_id].get('type') == 'asset'):
                        self.entry_nodes.append(node_id)

    def identify_entry_nodes(self):
        self.entry_nodes = []
        internet_facing = set()
        for u, v, edge_data in self.graph.edges(data=True):
            if edge_data.get('internet_facing', False):
                if self.graph.nodes[u].get('type') == 'asset':
                    internet_facing.add(u)
                    self.graph.nodes[u]['is_internet_facing'] = True
                if self.graph.nodes[v].get('type') == 'asset':
                    internet_facing.add(v)
                    self.graph.nodes[v]['is_internet_facing'] = True
        for node_id, data in self.graph.nodes(data=True):
            if data.get('type') == 'asset' and node_id not in internet_facing:
                asset_name = data.get('asset_name', '').lower()
                if any(keyword in asset_name for keyword in ['web server', 'vpn']):
                    internet_facing.add(node_id)
                    data['is_internet_facing'] = True
                    data['is_entry_point'] = True
                if 'workstation' in asset_name and 'phishing' in data.get('apt3_entry_justification', ''):
                    internet_facing.add(node_id)
                    data['is_internet_facing'] = True
                    data['is_entry_point'] = True
        self.entry_nodes = list(internet_facing)
        if not self.entry_nodes:
            for node_id, data in self.graph.nodes(data=True):
                if data.get('type') == 'entry':
                    self.entry_nodes.append(node_id)
        logger.info(f"Identified entry nodes: {self.entry_nodes}")

    def _adjust_edge_weights_for_attacker_stage(self, current_stage):
        """
        Adjust edge weights based on attacker's current kill chain stage.

        Args:
            current_stage: Current kill chain stage of the attacker
        """
        for u, v, data in self.graph.edges(data=True):
            u_stage = self.graph.nodes[u].get('ckc_stage')
            v_stage = self.graph.nodes[v].get('ckc_stage')

            if u_stage and v_stage:
                # Favor edges aligned with current or next stage
                if u_stage == current_stage or v_stage == current_stage:
                    data['weight'] = data.get('weight', 1.0) * 0.7
                elif v_stage.value == current_stage.value + 1:
                    data['weight'] = data.get('weight', 1.0) * 0.8

    def _calculate_tactic_fit(self, vuln_node_id, tactic):
        """
        Calculate tactic fit score for a vulnerability and tactic.

        Args:
            vuln_node_id: ID of vulnerability node
            tactic: Tactic name

        Returns:
            float: Tactic fit score between 0 and 1
        """
        vuln_data = self.graph.nodes[vuln_node_id]

        # Get techniques associated with this vulnerability
        techniques = vuln_data.get('techniques', [])

        # If we have techniques and a tactic, calculate actual technique-tactic alignment
        if techniques and tactic:
            # Get all techniques associated with this tactic from our mapping
            tactic_techniques = self.tactic_to_techniques.get(tactic, [])

            # Calculate alignment score: percentage of vulnerability's techniques
            # that match this tactic's techniques
            matching_techniques = [t for t in techniques if t in tactic_techniques]
            if techniques:  # Avoid division by zero
                alignment_score = len(matching_techniques) / len(techniques)
            else:
                alignment_score = 0.0

            # Weight alignment score more heavily (70%) but also consider EPSS (30%)
            # Use EPSS directly without normalization
            base_similarity = 0.7 * alignment_score + 0.3 * vuln_data.get('epss', 0.1)
        else:
            # If no techniques or tactic available, fall back to just EPSS
            base_similarity = vuln_data.get('epss', 0.1)

        # Augment with phi(s) (system-specific factor, e.g., complexity)
        complexity_map = {'low': 1.0, 'medium': 0.8, 'high': 0.6}
        phi_s = complexity_map.get(vuln_data.get('complexity', 'medium'), 0.8)

        # Calculate final tactic fit
        tf = base_similarity * phi_s + self.epsilon

        return min(tf, 1.0)  # Ensure TF <= 1

    def _calculate_roi(self, current_node, next_node):
        """
        Calculate ROI for moving from current node to next node.

        Args:
            current_node: Current node ID
            next_node: Next node ID

        Returns:
            float: ROI value
        """
        # Get node data
        current_data = self.graph.nodes[current_node]
        next_data = self.graph.nodes[next_node]

        # Get edge data
        edge_data = self.graph[current_node][next_node]

        # Calculate cost C(a)
        cost = edge_data.get('cost', 1.0)

        # Get vulnerability success probability P(v)
        # Use EPSS if available, otherwise fall back to default
        vuln_prob = current_data.get('epss', 0.1)

        # Adjust for exploit availability
        if current_data.get('exploit', False):
            vuln_prob *= 1.5

        # Calculate business value BV(a)
        asset_data = next_data
        if next_data.get('type') != 'asset':
            # If next node is not an asset, try to find associated asset
            asset_id = self.component_to_asset.get(next_data.get('component_id'))
            if asset_id and asset_id in self.graph:
                asset_data = self.graph.nodes[asset_id]

        # Get business value
        BV = asset_data.get('business_value', 100000)

        # Calculate ROI
        ROI = (vuln_prob * BV) / (cost + self.epsilon)

        return ROI

    def old_select_attack_path(self, attacker_current_stage=None):
        """
        Select the optimal attack path considering TTPs, CKC stages, and business value.
        Ensures paths follow asset → vulnerability → asset chains.
        """
        if attacker_current_stage:
            self._adjust_edge_weights_for_attacker_stage(attacker_current_stage)

        paths = []
        for target in self.high_value_targets:
            for entry in self.entry_nodes:
                try:
                    target_paths = list(nx.shortest_simple_paths(self.graph, entry, target, weight='weight'))

                    # Filter paths to ensure they follow asset → vulnerability → asset pattern
                    valid_paths = []
                    for path in target_paths[:30]:  # Check more paths to find valid ones
                        is_valid = True
                        for i in range(len(path) - 1):
                            u, v = path[i], path[i + 1]
                            u_type = self.graph.nodes[u].get('type')
                            v_type = self.graph.nodes[v].get('type')

                            # Rules for valid paths:
                            # 1. An asset should be followed by its own vulnerability
                            # 2. A vulnerability should be followed by a different asset
                            if u_type == 'asset' and v_type == 'asset':
                                # Direct asset to asset - invalid
                                is_valid = False
                                break
                            elif u_type == 'vulnerability' and v_type == 'vulnerability':
                                # Direct vulnerability to vulnerability - invalid
                                is_valid = False
                                break
                            elif u_type == 'asset' and v_type == 'vulnerability':
                                # Asset to vulnerability - only valid if asset owns the vulnerability
                                if self.graph.nodes[v].get('asset_id') != u:
                                    is_valid = False
                                    break
                            elif u_type == 'vulnerability' and v_type == 'asset':
                                # Vulnerability to asset - only valid if asset is NOT the parent
                                if self.graph.nodes[u].get('asset_id') == v:
                                    is_valid = False
                                    break

                        if is_valid:
                            valid_paths.append(path)
                            if len(valid_paths) >= 10:  # Limit to top 10 valid paths
                                break

                    paths.extend(valid_paths)

                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

        if not paths:
            # Fallback to any valid path between assets through vulnerabilities
            for entry in self.entry_nodes:
                for asset in self.asset_nodes:
                    if asset not in self.high_value_targets and asset != entry:
                        try:
                            asset_paths = []
                            for path in nx.all_simple_paths(self.graph, entry, asset, cutoff=8):
                                # Check if the path follows asset → vulnerability → asset pattern
                                is_valid = True
                                for i in range(len(path) - 1):
                                    u, v = path[i], path[i + 1]
                                    u_type = self.graph.nodes[u].get('type')
                                    v_type = self.graph.nodes[v].get('type')

                                    if u_type == 'asset' and v_type == 'asset':
                                        is_valid = False
                                        break
                                    elif u_type == 'vulnerability' and v_type == 'vulnerability':
                                        is_valid = False
                                        break
                                    elif u_type == 'asset' and v_type == 'vulnerability':
                                        # Asset to vulnerability - only valid if asset owns the vulnerability
                                        if self.graph.nodes[v].get('asset_id') != u:
                                            is_valid = False
                                            break
                                    elif u_type == 'vulnerability' and v_type == 'asset':
                                        # Vulnerability to asset - only valid if asset is NOT the parent
                                        if self.graph.nodes[u].get('asset_id') == v:
                                            is_valid = False
                                            break

                                if is_valid:
                                    asset_paths.append(path)
                                    if len(asset_paths) >= 3:  # Limit to top 3 paths per asset
                                        break

                            paths.extend(asset_paths)
                            if len(paths) >= 10:
                                break

                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            continue

        # If still no paths, try to create a path manually
        if not paths:
            logger.warning("No valid paths found through graph search, attempting to construct manually")
            for entry in self.entry_nodes:
                entry_vulns = []
                for node in self.graph.successors(entry):
                    if (self.graph.nodes[node].get('type') == 'vulnerability' and
                            self.graph.nodes[node].get('asset_id') == entry):
                        entry_vulns.append(node)

                for vuln in entry_vulns:
                    for target in self.graph.successors(vuln):
                        if (self.graph.nodes[target].get('type') == 'asset' and
                                target != entry):
                            # Found a simple path: entry -> vulnerability -> target
                            paths.append([entry, vuln, target])
                            break
                    if paths:
                        break
                if paths:
                    break

        # Score the paths
        path_scores = []
        for path in paths:
            path_roi = 0.0
            path_probability = 1.0
            asset_count = 0
            vuln_count = 0

            for i in range(len(path) - 1):
                current_node = path[i]
                next_node = path[i + 1]

                # Calculate edge probability
                if self.graph.has_edge(current_node, next_node):
                    edge_prob = self.graph.edges[current_node, next_node].get('probability', 0.5)
                    path_probability *= edge_prob

                # Count assets and vulnerabilities in path
                if self.graph.nodes[current_node].get('type') == 'asset':
                    asset_count += 1
                elif self.graph.nodes[current_node].get('type') == 'vulnerability':
                    vuln_count += 1
                    current_vuln_data = self.graph.nodes[current_node]
                    # Calculate ROI for vulnerabilities
                    epss = current_vuln_data.get('epss', 0.1)
                    edge_data = self.graph[current_node][next_node]
                    vuln_roi = epss * edge_data.get('probability', 0.5)
                    path_roi += vuln_roi

            # Add last node to counts
            if self.graph.nodes[path[-1]].get('type') == 'asset':
                asset_count += 1
            elif self.graph.nodes[path[-1]].get('type') == 'vulnerability':
                vuln_count += 1

            # Calculate final score with adjustments
            if vuln_count > 0:
                path_roi /= vuln_count  # Normalize by number of vulnerabilities

            # Prefer paths with reasonable number of steps
            length_factor = 1.0
            if len(path) < 4:  # Too short
                length_factor = 0.7
            elif len(path) > 10:  # Too long
                length_factor = 0.5

            # Calculate final score - balance between ROI and probability
            final_score = (0.7 * path_roi + 0.3 * path_probability) * length_factor

            # Bonus for paths that reach high-value targets
            if path[-1] in self.high_value_targets:
                final_score *= 1.5

            path_scores.append((path, final_score))

        if path_scores:
            path_scores.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"Selected best path: {path_scores[0][0]} with score {path_scores[0][1]:.4f}")
            return path_scores[0][0]

        # If no valid paths, return any available path (fallback)
        logger.warning("No valid attack paths found with asset → vulnerability → asset pattern.")
        return self._fallback_attack_path()

    def select_attack_path(self, attacker_current_stage=None, prioritize_apt3=True):
        """
        Select an attack path, allowing direct asset-to-asset edges if no vulnerabilities exist.
        Optionally prioritizes APT3-relevant vulnerabilities.
        """
        if attacker_current_stage:
            self._adjust_edge_weights_for_attacker_stage(attacker_current_stage)

        apt3_mitre_techniques = {'T1203', 'T1068', 'T1055', 'T1548', 'T1190', 'T1040', 'T1557', 'T1134',
                                 'T1027', 'T1562.003', 'T1574.006', 'T1574.007', 'T1210', 'T1059.001',
                                 'T1199', 'T1078', 'T1133', 'T1195', 'T1021', 'T1505'}
        priority_cves = {'CVE-2018-13379', 'ZERO-DAY-001', 'CVE-2015-3113'}

        # Get compromised assets and validate connectivity
        system = getattr(self.system_graph, 'system', None)
        compromised_assets = [str(asset.asset_id) for asset in system.assets if asset.is_compromised] if system else []
        start_nodes = []
        for node in set(self.entry_nodes + compromised_assets):
            if self.graph.out_degree(node) > 0:  # Ensure node has outgoing edges
                start_nodes.append(node)
        if not start_nodes:
            logger.error("No valid start nodes with outgoing edges")
            return self._fallback_attack_path()

        logger.info(f"Start nodes: {start_nodes}")
        paths = []

        try:
            for entry in start_nodes:
                for target in self.high_value_targets + self.asset_nodes:
                    if entry == target:
                        continue
                    try:
                        # Use edge-validated path finding
                        for path in nx.all_simple_paths(self.graph, entry, target, cutoff=12):
                            full_path = []
                            valid = True
                            for i in range(len(path) - 1):
                                src = path[i]
                                dst = path[i + 1]
                                if not self.graph.has_edge(src, dst):
                                    logger.debug(f"No edge {src} -> {dst} in path {path}")
                                    valid = False
                                    break
                                full_path.append(src)
                                # Optionally insert vulnerability node
                                if self.graph.nodes[src].get('type') == 'asset':
                                    vuln_nodes = [
                                        n for n in self.graph.successors(src)
                                        if self.graph.nodes[n].get('type') == 'vulnerability' and
                                           self.graph.nodes[n].get('asset_id') == src and
                                           not self.graph.nodes[n].get('is_patched', False) and
                                           not self.graph.nodes[n].get('is_exploited', False)
                                    ]
                                    best_vuln = None
                                    best_vuln_score = -1
                                    for vuln_node in vuln_nodes:
                                        vuln_data = self.graph.nodes[vuln_node]
                                        cve_id = vuln_data.get('cve_id', '')
                                        techniques = set(vuln_data.get('mitre_techniques', []))
                                        score = 0.5
                                        if prioritize_apt3:
                                            if cve_id in priority_cves:
                                                score += 0.3
                                            if techniques.intersection(apt3_mitre_techniques):
                                                score += 0.2
                                        score += vuln_data.get('epss', 0.1)  # General exploitability
                                        if score > best_vuln_score and self.graph.has_edge(vuln_node, dst):
                                            best_vuln = vuln_node
                                            best_vuln_score = score
                                    if best_vuln:
                                        full_path.append(best_vuln)
                                        logger.debug(f"Inserted vulnerability {best_vuln} in path")
                            if valid:
                                full_path.append(path[-1])
                                paths.append(full_path)
                                path_str = " -> ".join(
                                    [f"{node} ({self.graph.nodes[node].get('type', 'unknown')})" for node in full_path])
                                logger.debug(f"Generated path: {path_str}")
                            if len(paths) >= 30:
                                break
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        logger.debug(f"No path from {entry} to {target}")
                        continue
        except Exception as e:
            logger.error(f"Error generating paths: {e}", exc_info=True)
            return self._fallback_attack_path()

        if not paths:
            logger.warning("No valid paths found, resorting to fallback")
            return self._fallback_attack_path()

        # Score paths
        path_scores = []
        for path in paths:
            path_roi = 0.0
            path_probability = 1.0
            vuln_count = 0
            techniques = set()
            cves = set()
            try:
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    if self.graph.has_edge(u, v):
                        edge_prob = self.graph.edges[u, v].get('probability', 0.5)
                        if not 0 <= edge_prob <= 1:
                            logger.warning(f"Invalid edge probability {edge_prob} for {u} -> {v}, using 0.5")
                            edge_prob = 0.5
                        path_probability *= edge_prob
                        if self.graph.nodes[u].get('type') == 'vulnerability':
                            vuln_count += 1
                            vuln_data = self.graph.nodes[u]
                            cve_id = vuln_data.get('cve_id', '')
                            cves.add(cve_id)
                            vuln_roi = vuln_data.get('epss', 0.1)
                            if prioritize_apt3:
                                if cve_id in priority_cves:
                                    vuln_roi *= 1.3
                                if set(vuln_data.get('mitre_techniques', [])).intersection(apt3_mitre_techniques):
                                    vuln_roi *= 1.2
                                    techniques.update(vuln_data.get('mitre_techniques', []))
                            path_roi += vuln_roi
                if vuln_count > 0:
                    path_roi /= vuln_count
                asset_count = sum(1 for node in path if self.graph.nodes[node].get('type') == 'asset')
                length_factor = 1.0 if 3 <= len(path) <= 12 else 0.8
                technique_bonus = 1.0 + 0.1 * len(
                    techniques.intersection(apt3_mitre_techniques)) if prioritize_apt3 else 1.0
                cve_bonus = 1.0 + 0.2 * len(cves.intersection(priority_cves)) if prioritize_apt3 else 1.0
                # Normalize scores to prevent extreme values
                path_roi = min(path_roi, 1.0)
                path_probability = max(self.epsilon, min(1.0, path_probability))
                final_score = (0.6 * path_roi + 0.4 * path_probability) * length_factor * technique_bonus * cve_bonus
                if path[-1] in self.high_value_targets:
                    final_score *= 1.5
                if str(path[-1]) == '8':  # RTU priority
                    final_score *= 1.2
                path_scores.append((path, final_score))
                path_str = " -> ".join([f"{node} ({self.graph.nodes[node].get('type', 'unknown')})" for node in path])
                logger.debug(f"Scored path: {path_str}, Score: {final_score:.4f}")
            except Exception as e:
                logger.error(f"Error scoring path {path}: {e}", exc_info=True)
                continue

        if path_scores:
            path_scores.sort(key=lambda x: x[1], reverse=True)
            selected_path = path_scores[0][0]
            path_str = " -> ".join(
                [f"{node} ({self.graph.nodes[node].get('type', 'unknown')})" for node in selected_path])
            logger.info(f"Selected best path: {path_str}, Score: {path_scores[0][1]:.4f}")
            return selected_path

        logger.warning("No valid attack paths scored")
        return self._fallback_attack_path()

    def _fallback_attack_path(self):
        """Fallback method to find any valid path when normal selection fails."""
        # Try to find a simple path from any entry to any high-value target
        for entry in self.entry_nodes:
            for target in self.high_value_targets:
                try:
                    path = nx.shortest_path(self.graph, entry, target)
                    logger.info(f"Fallback: Found direct path from {entry} to {target}")
                    return path
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

        # If no path to high-value targets, try any path between entry points and assets
        for entry in self.entry_nodes:
            for asset in self.asset_nodes:
                if asset != entry:
                    try:
                        path = nx.shortest_path(self.graph, entry, asset)
                        logger.info(f"Fallback: Found path from {entry} to asset {asset}")
                        return path
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue

        # If still no path, create a simple path with entry point and connected node
        if self.entry_nodes:
            entry = self.entry_nodes[0]
            for successor in self.graph.successors(entry):
                logger.info(f"Fallback: Creating minimal path from {entry} to {successor}")
                return [entry, successor]

        # Last resort: just return a list of asset nodes (up to 3)
        assets = [n for n in self.graph.nodes() if self.graph.nodes[n].get('type') == 'asset']
        if assets:
            logger.warning("Fallback: Returning arbitrary asset nodes as path")
            return assets[:min(3, len(assets))]

        # If everything fails, return empty path
        logger.error("Fallback: Could not create any valid attack path")
        return []

    def old_find_top_attack_paths(self, num_paths=3, attacker_current_stage=None):
        """
        Select the top N attack paths with highest probabilities.

        Args:
            num_paths: Number of attack paths to return
            attacker_current_stage: Current kill chain stage of the attacker (KillChainStage enum)

        Returns:
            list: List of attack paths, each as a list of node IDs
        """
        if attacker_current_stage:
            logger.debug(f"Adjusting edge weights for CKC stage {attacker_current_stage.name}")
            self._adjust_edge_weights_for_attacker_stage(attacker_current_stage)

        all_paths = []
        all_probs = []

        # Search for paths from all entry points to all high-value targets
        for entry in self.entry_nodes:
            for target in self.high_value_targets:
                if entry == target:
                    continue  # Skip if entry is the target

                try:
                    # Find all paths from entry to target
                    target_paths = list(nx.all_shortest_paths(self.graph, entry, target, weight='weight'))

                    # Calculate probability for each path
                    for path in target_paths:
                        path_prob = 1.0
                        for i in range(len(path) - 1):
                            if self.graph.has_edge(path[i], path[i + 1]):
                                edge_data = self.graph.edges[path[i], path[i + 1]]
                                prob = edge_data.get('probability', 0.5)  # Default to 0.5
                                if prob <= 0 or prob > 1:
                                    logger.warning(
                                        f"Invalid edge probability {prob} for edge {path[i]} -> {path[i + 1]}, using 0.5")
                                    prob = 0.5
                                path_prob *= prob

                        all_paths.append(path)
                        all_probs.append(path_prob)
                except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                    logger.debug(f"No path from {entry} to {target}: {e}")
                    continue

        # If no paths between entry and high-value targets, try entry to any asset
        if not all_paths:
            for entry in self.entry_nodes:
                for asset in self.asset_nodes:
                    if entry == asset or asset in self.high_value_targets:
                        continue

                    try:
                        target_paths = list(nx.all_shortest_paths(self.graph, entry, asset, weight='weight'))

                        for path in target_paths:
                            path_prob = 1.0
                            for i in range(len(path) - 1):
                                if self.graph.has_edge(path[i], path[i + 1]):
                                    edge_data = self.graph.edges[path[i], path[i + 1]]
                                    prob = edge_data.get('probability', 0.5)
                                    if prob <= 0 or prob > 1:
                                        logger.warning(
                                            f"Invalid edge probability {prob} for edge {path[i]} -> {path[i + 1]}, using 0.5")
                                        prob = 0.5
                                    path_prob *= prob

                            all_paths.append(path)
                            all_probs.append(path_prob)
                        if len(all_paths) >= num_paths:
                            break
                    except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                        logger.debug(f"No path from {entry} to {asset}: {e}")
                        continue

        # Return the top N paths by probability
        if all_paths:
            # Sort paths by probability
            sorted_paths = [x for _, x in sorted(zip(all_probs, all_paths), reverse=True)]

            # Take the top N unique paths
            unique_paths = []
            for path in sorted_paths:
                # Check if this path is sufficiently different from existing paths
                is_unique = True
                for existing_path in unique_paths:
                    overlap = set(path) & set(existing_path)
                    # If paths share more than 50% of nodes, consider them similar
                    if len(overlap) > min(len(path), len(existing_path)) * 0.5:
                        is_unique = False
                        break

                if is_unique:
                    unique_paths.append(path)
                    if len(unique_paths) >= num_paths:
                        break

            logger.info(f"Found {len(unique_paths)} unique attack paths")
            return unique_paths

        logger.warning("No attack paths found")
        return []

    def find_top_attack_paths(self, num_paths=3, attacker_current_stage=None) -> List[Tuple[List[str], float]]:
        if attacker_current_stage:
            logger.debug(f"Adjusting edge weights for CKC stage {attacker_current_stage.name}")
            self._adjust_edge_weights_for_attacker_stage(attacker_current_stage)

        all_paths = []
        all_scores = []

        logger.info("Starting path enumeration for attack graph")
        for entry in self.entry_nodes:
            for target in self.high_value_targets:
                if entry == target:
                    logger.debug(f"Skipping path from {entry} to itself")
                    continue
                try:
                    target_paths = list(nx.all_shortest_paths(self.graph, entry, target, weight='weight'))

                    #logger.info(f"Found {len(target_paths)} paths from {entry} to {target}")
                    for path in target_paths:
                        score = self._calculate_path_probability(path)
                        path_str = " -> ".join(
                            [f"{node} ({self.graph.nodes[node].get('type', 'unknown')})" for node in path])
                        logger.debug(f"Path: {path_str}, Score: {score:.4f}")
                        all_paths.append(path)
                        all_scores.append(score)
                except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                    logger.debug(f"No path from {entry} to {target}: {e}")
                    continue

        # Fallback: entry to any asset
        if not all_paths:
            logger.warning("No paths to high-value targets, trying any asset")
            for entry in self.entry_nodes:
                for asset in self.asset_nodes:
                    if asset == entry or asset in self.high_value_targets:
                        continue
                    try:
                        target_paths = list(nx.all_shortest_paths(self.graph, entry, asset, weight='weight'))
                        logger.info(f"Found {len(target_paths)} fallback paths from {entry} to {asset}")
                        for path in target_paths:
                            score = self._calculate_path_probability(path)
                            path_str = " -> ".join(
                                [f"{node} ({self.graph.nodes[node].get('type', 'unknown')})" for node in path])
                            logger.debug(f"Fallback Path: {path_str}, Score: {score:.4f}")
                            all_paths.append(path)
                            all_scores.append(score)
                        if len(all_paths) >= num_paths:
                            break
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue

        # Rank and filter unique top paths
        if all_paths:
            sorted_paths = sorted(zip(all_paths, all_scores), key=lambda x: x[1], reverse=True)
            unique_paths = []
            for path, score in sorted_paths:
                is_unique = all(self.jaccard_similarity(path, existing[0]) <= 0.5 for existing in unique_paths)
                path_str = " -> ".join([f"{node} ({self.graph.nodes[node].get('type', 'unknown')})" for node in path])
                if is_unique:
                    #logger.info(f"Selected unique path: {path_str}, Score: {score:.4f}")
                    unique_paths.append((path, score))
                    if len(unique_paths) >= num_paths:
                        break
                else:
                    logger.debug(f"Discarded similar path: {path_str}, Score: {score:.4f}")

            logger.info(f"Returning {len(unique_paths)} unique attack paths")
            return unique_paths

        logger.warning("No attack paths found")
        return []
    def jaccard_similarity(self, path1, path2):
        """Calculate Jaccard similarity between two paths, handling non-hashable elements."""
        try:
            # Ensure paths are lists of hashable elements
            def to_hashable(path):
                return [str(node) if not isinstance(node, (str, int, tuple)) else node for node in path]

            set1 = set(to_hashable(path1))
            set2 = set(to_hashable(path2))
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            similarity = intersection / union if union > 0 else 0.0
            logger.debug(f"Jaccard similarity between {path1} and {path2}: {similarity}")
            return similarity
        except Exception as e:
            logger.error(f"Error computing Jaccard similarity for paths {path1} and {path2}: {e}")
            return 0.0  # Return 0 similarity on error to avoid blocking path selection

    def validate_graph(self):
        """
        Validate the attack graph for required attributes and connectivity.
        Returns True if valid, False otherwise.
        """
        invalid_nodes = []
        missing_attrs = defaultdict(list)
        required_node_attrs = ['type', 'ckc_stage']
        required_edge_attrs = ['probability', 'cost', 'weight']

        # Check node attributes
        for node in self.graph.nodes:
            node_data = self.graph.nodes[node]
            for attr in required_node_attrs:
                if attr not in node_data:
                    missing_attrs[attr].append(node)
            # Set default type if missing
            if 'type' not in node_data:
                node_data['type'] = 'unknown'
                logger.warning(f"Node {node} missing 'type', set to 'unknown'")

        # Check edge attributes
        for u, v in self.graph.edges:
            edge_data = self.graph.edges[u, v]
            for attr in required_edge_attrs:
                if attr not in edge_data:
                    missing_attrs[attr].append((u, v))
            # Set default values for missing edge attributes
            if 'probability' not in edge_data:
                edge_data['probability'] = 0.5
                logger.warning(f"Edge {u} -> {v} missing 'probability', set to 0.5")
            if 'cost' not in edge_data:
                edge_data['cost'] = 1.0
                logger.warning(f"Edge {u} -> {v} missing 'cost', set to 1.0")
            if 'weight' not in edge_data:
                edge_data['weight'] = 1.0
                logger.warning(f"Edge {u} -> {v} missing 'weight', set to 1.0")

        # Log missing attributes
        for attr, items in missing_attrs.items():
            logger.warning(f"Found {len(items)} items missing '{attr}' attribute: {items}")

        # Check connectivity
        undirected_graph = self.graph.to_undirected()
        if not nx.is_connected(undirected_graph):
            logger.error("Graph is not connected, attack progression may fail")
            return False

        # Check for isolated nodes
        isolated_nodes = [node for node in self.graph.nodes if self.graph.degree(node) == 0]
        if isolated_nodes:
            logger.warning(f"Found {len(isolated_nodes)} isolated nodes: {isolated_nodes}")

        # Verify entry nodes and targets
        if not self.entry_nodes:
            logger.error("No entry nodes defined")
            return False
        if not self.high_value_targets:
            logger.warning("No high-value targets defined, selecting top assets as fallback")
            self._fallback_high_value_targets()

        logger.info("Graph validation completed")
        return True

    def _fallback_high_value_targets(self):
        """Select top assets by criticality or business value as high-value targets."""
        assets = [(node, data.get('criticality_level', 0) + data.get('business_value', 0) / 100000)
                  for node, data in self.graph.nodes(data=True) if data.get('type') == 'asset']
        if assets:
            assets.sort(key=lambda x: x[1], reverse=True)
            self.high_value_targets = [node for node, _ in assets[:max(1, len(assets) // 2)]]
            logger.info(
                f"Selected {len(self.high_value_targets)} fallback high-value targets: {self.high_value_targets}")

    def find_diverse_attack_paths(self, num_paths=4, primary_path=None):
        """Find diverse attack paths using Yen's algorithm with Jaccard similarity."""
        if not self.entry_nodes or not self.high_value_targets:
            logger.warning("No entry nodes or high-value targets defined")
            return []

        result_paths = []
        if primary_path and all(node in self.graph for node in primary_path):
            result_paths.append(primary_path)  # Add primary path to ensure diversity against it
            logger.debug(f"Primary path added to diversity check: {primary_path}")

        for source in self.entry_nodes:
            for target in self.high_value_targets:
                try:
                    # Use Yen's algorithm to get k-shortest paths
                    k_shortest = list(nx.shortest_simple_paths(self.graph, source, target, weight='weight'))
                    for path in k_shortest[:num_paths * 2]:  # Consider more paths to filter diverse ones
                        # Ensure path is a flat list of nodes
                        if not all(isinstance(node, (str, int)) for node in path):
                            logger.warning(f"Path contains non-hashable nodes: {path}")
                            continue
                        # Skip if this path is the primary path
                        if primary_path and path == primary_path:
                            continue
                        # Only add path if it's sufficiently different (Jaccard similarity <= 0.8)
                        similar_paths = [existing for existing in result_paths
                                         if self.jaccard_similarity(path, existing) > 0.8]
                        if not similar_paths:
                            logger.debug(f"Adding diverse path: {path}")
                            result_paths.append(path)
                            if len(result_paths) >= num_paths + (1 if primary_path else 0):
                                return result_paths[1:] if primary_path else result_paths
                except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                    logger.debug(f"No path from {source} to {target}: {e}")
                    continue
        logger.info(f"Found {len(result_paths)} diverse paths, less than requested {num_paths}")
        return result_paths[1:] if primary_path and result_paths else result_paths



class AttackTreeGenerator:
    def __init__(self, attack_graph: AttackGraph):
        self.attack_graph = attack_graph
        self.attack_trees = {}

    def generate_tree(self, target_id: str, max_depth: Optional[int] = None) -> nx.DiGraph:
        """Generate an attack tree with dynamic depth adjustment."""
        if not self.attack_graph or not self.attack_graph.graph:
            logger.error("No attack graph available to generate tree")
            return nx.DiGraph()

        # Validate target
        if target_id not in self.attack_graph.graph.nodes:
            logger.error(f"Target {target_id} not in graph")
            return nx.DiGraph()

        # Check if target has predecessors
        predecessors = list(self.attack_graph.graph.predecessors(target_id))
        if not predecessors:
            logger.warning(f"Target {target_id} has no predecessors, returning empty tree")
            tree = nx.DiGraph()
            root_attrs = dict(self.attack_graph.graph.nodes[target_id])
            tree.add_node(target_id, **root_attrs)
            return tree

        # Calculate dynamic max_depth
        if max_depth is None:
            try:
                diameter = nx.diameter(self.attack_graph.graph.to_undirected())
                max_depth = min(diameter + 2, 10)
                criticality = self.attack_graph.graph.nodes[target_id].get('criticality_level', 1)
                max_depth = max(max_depth, int(criticality * 2))
            except nx.NetworkXError:
                max_depth = 5
        logger.info(f"Using max_depth={max_depth} for target {target_id}")

        tree = nx.DiGraph()
        root_attrs = dict(self.attack_graph.graph.nodes[target_id])
        tree.add_node(target_id, **root_attrs)
        truncated_nodes = set()

        def add_predecessors(node_id, current_depth=0, visited=None):
            if visited is None:
                visited = set()
            if current_depth >= max_depth:
                truncated_nodes.add(node_id)
                return
            if node_id in visited:
                return
            visited.add(node_id)
            predecessors = list(self.attack_graph.graph.predecessors(node_id))
            predecessors.sort(key=lambda p: self.attack_graph.graph[p][node_id].get('probability', 0.0), reverse=True)
            for pred in predecessors:
                if pred in tree and tree.has_edge(pred, node_id):
                    continue
                if pred not in self.attack_graph.graph.nodes:
                    logger.debug(f"Predecessor {pred} not in graph, skipping")
                    continue
                pred_attrs = dict(self.attack_graph.graph.nodes[pred])
                tree.add_node(pred, **pred_attrs)
                edge_attrs = dict(self.attack_graph.graph.edges[pred, node_id])
                tree.add_edge(pred, node_id, **edge_attrs)
                add_predecessors(pred, current_depth + 1, visited)

        add_predecessors(target_id)
        self.attack_trees[target_id] = tree
        if truncated_nodes:
            logger.warning(f"Truncated {len(truncated_nodes)} nodes at depth {max_depth} for target {target_id}")
        logger.info(
            f"Generated attack tree for target {target_id} with {len(tree.nodes)} nodes, {len(tree.edges)} edges")
        return tree

    def find_attack_vectors(self, target_id):
        """
        Find all possible attack vectors (paths) to a target.

        Args:
            target_id: ID of the target node

        Returns:
            list: List of attack paths as lists of node IDs
        """
        if target_id not in self.attack_trees:
            self.generate_tree(target_id)

        tree = self.attack_trees[target_id]
        attack_vectors = []

        # Get all leaf nodes (nodes with no predecessors in the tree)
        leaf_nodes = [n for n in tree.nodes() if tree.in_degree(n) == 0]

        # For each leaf, find a path to the target
        for leaf in leaf_nodes:
            try:
                paths = list(nx.all_simple_paths(tree, leaf, target_id))
                attack_vectors.extend(paths)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

        return attack_vectors

    def get_critical_components(self, target_id):
        """
        Identify critical components in the attack tree.

        Args:
            target_id: ID of the target node

        Returns:
            list: List of critical component IDs
        """
        if target_id not in self.attack_trees:
            self.generate_tree(target_id)

        tree = self.attack_trees[target_id]
        critical_nodes = []

        # Find nodes with high betweenness centrality
        try:
            centrality = nx.betweenness_centrality(tree)
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

            # Take top 20% of nodes as critical
            num_critical = max(1, int(len(sorted_nodes) * 0.2))
            critical_nodes = [node for node, score in sorted_nodes[:num_critical]]
        except:
            # Fallback if centrality calculation fails
            # Identify nodes with multiple incoming edges
            critical_nodes = [n for n in tree.nodes() if tree.in_degree(n) > 1]

        return critical_nodes

    def get_min_cost_vector(self, target_id):
        """
        Find the attack vector with minimum cost.

        Args:
            target_id: ID of the target node

        Returns:
            tuple: (path, cost) where path is the minimum-cost path
        """
        attack_vectors = self.find_attack_vectors(target_id)

        min_cost = float('inf')
        min_cost_path = None

        for path in attack_vectors:
            path_cost = 0
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if self.attack_graph.graph.has_edge(u, v):
                    edge_cost = self.attack_graph.graph.edges[u, v].get('cost', 1.0)
                    path_cost += edge_cost

            if path_cost < min_cost:
                min_cost = path_cost
                min_cost_path = path

        return min_cost_path, min_cost

    def get_max_probability_vector(self, target_id):
        """
        Find the attack vector with maximum success probability.

        Args:
            target_id: ID of the target node

        Returns:
            tuple: (path, probability) where path is the maximum-probability path
        """
        attack_vectors = self.find_attack_vectors(target_id)

        max_prob = 0.0
        max_prob_path = None

        for path in attack_vectors:
            path_prob = 1.0
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if self.attack_graph.graph.has_edge(u, v):
                    edge_prob = self.attack_graph.graph.edges[u, v].get('probability', 1.0)
                    path_prob *= edge_prob

            if path_prob > max_prob:
                max_prob = path_prob
                max_prob_path = path

        return max_prob_path, max_prob

class AttackPathAnalyzer:
    """
    Analyzes attack paths to identify critical vulnerabilities, assets, and attack vectors.
    """
    def __init__(self, attack_graph: AttackGraph):
        self.attack_graph = attack_graph
        self.tree_generator = AttackTreeGenerator(attack_graph)
        self.critical_nodes = {}
        self.critical_edges = {}
        self.vulnerability_rankings = {}
        self.asset_rankings = {}

    def analyze_all_targets(self) -> Dict:
        """
        Analyze all high-value targets in the attack graph.
        """
        results = {}
        for target_id in self.attack_graph.high_value_targets:
            results[target_id] = self.analyze_target(target_id)
        return results

    def analyze_target(self, target_id: str) -> Dict:
        """
        Analyze attack paths to a specific target.
        """
        tree = self.tree_generator.generate_tree(target_id)
        attack_vectors = self.tree_generator.find_attack_vectors(target_id)
        critical_components = self.tree_generator.get_critical_components(target_id)
        min_cost_vector, min_cost = self.tree_generator.get_min_cost_vector(target_id)
        max_prob_vector, max_prob = self.tree_generator.get_max_probability_vector(target_id)
        vuln_stats = self._calculate_vulnerability_statistics(target_id, attack_vectors)
        asset_stats = self._calculate_asset_statistics(target_id, attack_vectors)
        self.critical_nodes[target_id] = critical_components
        self._identify_critical_edges(target_id, attack_vectors)
        self._update_rankings(vuln_stats, asset_stats)
        return {
            'target_id': target_id,
            'num_attack_vectors': len(attack_vectors),
            'critical_components': critical_components,
            'min_cost_vector': {
                'path': min_cost_vector,
                'cost': min_cost
            },
            'max_probability_vector': {
                'path': max_prob_vector,
                'probability': max_prob
            },
            'vulnerability_statistics': vuln_stats,
            'asset_statistics': asset_stats
        }

    def _calculate_vulnerability_statistics(self, target_id: str, attack_vectors: List[List[str]]) -> Dict:
        """
        Calculate statistics about vulnerabilities in attack vectors.
        Uses a weighted sum approach for the final vulnerability score.
        """
        stats = {}
        vuln_counts = defaultdict(int)
        vuln_paths = defaultdict(list)

        for path_idx, path in enumerate(attack_vectors):
            for node in path:
                node_data = self.attack_graph.graph.nodes.get(node, {})
                if node_data.get('type') == 'vulnerability':
                    cve_id = node_data.get('cve_id', '')
                    if cve_id:
                        vuln_counts[cve_id] += 1
                        vuln_paths[cve_id].append(path_idx)

        vuln_scores = {}
        max_count = max(vuln_counts.values()) if vuln_counts else 1

        for cve_id, count in vuln_counts.items():
            vuln_node = None
            for node, data in self.attack_graph.graph.nodes(data=True):
                if data.get('type') == 'vulnerability' and data.get('cve_id') == cve_id:
                    vuln_node = node
                    break
            if not vuln_node:
                continue

            vuln_data = self.attack_graph.graph.nodes[vuln_node]

            # Calculate individual factors
            base_score = count / max_count
            cvss = vuln_data.get('cvss', 5.0)
            cvss_factor = cvss / 10.0
            exploit_factor = 1.5 if vuln_data.get('exploited', False) else 1.0
            epss = vuln_data.get('epss', 0.1)
            epss_factor = epss
            exploit_availability_factor = 1.2 if vuln_data.get('has_exploit', False) else 1.0

            # Define weights for each factor
            weights = {
                'base': 0.2,
                'cvss': 0.3,
                'exploit': 0.2,
                'epss': 0.2,
                'exploit_avail': 0.1
            }

            # Calculate weighted sum
            final_score = (
                    weights['base'] * base_score +
                    weights['cvss'] * cvss_factor +
                    weights['exploit'] * exploit_factor +
                    weights['epss'] * epss_factor +
                    weights['exploit_avail'] * exploit_availability_factor
            )

            # Store all data
            vuln_scores[cve_id] = {
                'count': count,
                'paths': vuln_paths[cve_id],
                'base_score': base_score,
                'cvss': cvss,
                'epss': epss,
                'has_exploit': vuln_data.get('has_exploit', False),
                'is_exploited': vuln_data.get('exploited', False),
                'is_patched': vuln_data.get('is_patched', False),
                'final_score': final_score
            }
        return vuln_scores

    def _calculate_asset_statistics(self, target_id: str, attack_vectors: List[List[str]]) -> Dict:
        """
        Calculate statistics about assets in attack vectors.
        """
        stats = {}
        asset_counts = defaultdict(int)
        asset_paths = defaultdict(list)

        for path_idx, path in enumerate(attack_vectors):
            for node in path:
                node_data = self.attack_graph.graph.nodes.get(node, {})
                if node_data.get('type') == 'asset':
                    asset_id = node
                    asset_counts[asset_id] += 1
                    asset_paths[asset_id].append(path_idx)

        asset_scores = {}
        max_count = max(asset_counts.values()) if asset_counts else 1

        for asset_id, count in asset_counts.items():
            asset_data = self.attack_graph.graph.nodes[asset_id]
            base_score = count / max_count
            criticality = asset_data.get('criticality', 1)
            criticality_factor = criticality / 5.0
            business_value = asset_data.get('business_value', 0)
            value_factor = min(1.0, business_value / 100000) if business_value > 0 else 0.2
            compromise_factor = 2.0 if asset_data.get('is_compromised', False) else 1.0
            final_score = base_score * criticality_factor * value_factor * compromise_factor
            asset_scores[asset_id] = {
                'count': count,
                'paths': asset_paths[asset_id],
                'base_score': base_score,
                'criticality': criticality,
                'business_value': business_value,
                'is_compromised': asset_data.get('is_compromised', False),
                'is_entry_point': asset_data.get('is_entry_point', False),
                'final_score': final_score
            }
        return asset_scores

    def _identify_critical_edges(self, target_id: str, attack_vectors: List[List[str]]) -> None:
        """
        Identify critical edges in attack vectors.
        """
        edge_counts = defaultdict(int)
        for path in attack_vectors:
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                edge_counts[edge] += 1
        edge_scores = {}
        max_count = max(edge_counts.values()) if edge_counts else 1

        for edge, count in edge_counts.items():
            source, target = edge
            if not self.attack_graph.graph.has_edge(source, target):
                continue
            edge_data = self.attack_graph.graph.edges[source, target]
            base_score = count / max_count
            type_factor = 1.0
            if edge_data.get('type') == 'vulnerability_exists':
                type_factor = 1.5
            elif edge_data.get('type') == 'initial_access':
                type_factor = 1.3
            elif edge_data.get('type') == 'lateral_movement':
                type_factor = 1.2
            traversal_factor = 2.0 if edge_data.get('traversed', False) else 1.0
            final_score = base_score * type_factor * traversal_factor
            edge_scores[edge] = final_score
        sorted_edges = sorted(edge_scores.items(), key=lambda x: x[1], reverse=True)
        self.critical_edges[target_id] = sorted_edges

    def _update_rankings(self, vuln_stats: Dict, asset_stats: Dict) -> None:
        """
        Update global vulnerability and asset rankings.
        """
        for cve_id, stats in vuln_stats.items():
            if cve_id not in self.vulnerability_rankings:
                self.vulnerability_rankings[cve_id] = {
                    'total_score': 0.0,
                    'path_count': 0,
                    'target_count': 0,
                    'metadata': {}
                }
            self.vulnerability_rankings[cve_id]['total_score'] += stats['final_score']
            self.vulnerability_rankings[cve_id]['path_count'] += len(stats['paths'])
            self.vulnerability_rankings[cve_id]['target_count'] += 1
            self.vulnerability_rankings[cve_id]['metadata'] = {
                'cvss': stats['cvss'],
                'epss': stats['epss'],
                'has_exploit': stats['has_exploit'],
                'is_exploited': stats['is_exploited'],
                'is_patched': stats['is_patched']
            }
        for asset_id, stats in asset_stats.items():
            if asset_id not in self.asset_rankings:
                self.asset_rankings[asset_id] = {
                    'total_score': 0.0,
                    'path_count': 0,
                    'target_count': 0,
                    'metadata': {}
                }
            self.asset_rankings[asset_id]['total_score'] += stats['final_score']
            self.asset_rankings[asset_id]['path_count'] += len(stats['paths'])
            self.asset_rankings[asset_id]['target_count'] += 1
            self.asset_rankings[asset_id]['metadata'] = {
                'criticality': stats['criticality'],
                'business_value': stats['business_value'],
                'is_compromised': stats['is_compromised'],
                'is_entry_point': stats['is_entry_point']
            }

    def get_critical_vulnerabilities(self, top_n: int = 10) -> List[Tuple[str, Dict]]:
        """
        Get the top N critical vulnerabilities across all targets.
        """
        sorted_vulns = sorted(self.vulnerability_rankings.items(),
                              key=lambda x: x[1]['total_score'],
                              reverse=True)
        return sorted_vulns[:top_n]

    def get_critical_assets(self, top_n: int = 10) -> List[Tuple[str, Dict]]:
        """
        Get the top N critical assets across all targets.
        """
        sorted_assets = sorted(self.asset_rankings.items(),
                               key=lambda x: x[1]['total_score'],
                               reverse=True)
        return sorted_assets[:top_n]

    def recommend_patches(self, defender_budget: float, max_patches: int = 5) -> List[Tuple[str, float]]:
        """
        Recommend vulnerabilities to patch based on their criticality.
        """
        critical_vulns = self.get_critical_vulnerabilities(top_n=20)
        patch_costs = []
        for cve_id, stats in critical_vulns:
            if stats['metadata'].get('is_patched', False):
                continue
            patch_cost = float('inf')
            for node, data in self.attack_graph.graph.nodes(data=True):
                if data.get('type') == 'vulnerability' and data.get('cve_id') == cve_id:
                    asset_id = data.get('asset_id', '')
                    component_id = data.get('component_id', '')
                    vuln_key = f"{cve_id}:{asset_id}:{component_id}"
                    if hasattr(self.attack_graph, 'cost_cache') and self.attack_graph.cost_cache:
                        patch_cost = self.attack_graph.cost_cache['patch_costs'].get(vuln_key, 200.0)
                    else:
                        cvss = data.get('cvss', 5.0)
                        patch_cost = 50 + cvss * 30
                    break
            patch_costs.append((cve_id, patch_cost, stats['total_score']))
        patch_costs.sort(key=lambda x: x[2] / x[1], reverse=True)
        selected_patches = []
        remaining_budget = defender_budget
        for cve_id, cost, score in patch_costs:
            if cost <= remaining_budget and len(selected_patches) < max_patches:
                selected_patches.append((cve_id, cost))
                remaining_budget -= cost
        return selected_patches

    def export_analysis_to_json(self, output_file: str = 'attack_analysis.json') -> None:
        """
        Export analysis results to a JSON file.
        """
        import json
        analysis_data = {
            'targets': {target_id: self.analyze_target(target_id)
                        for target_id in self.attack_graph.high_value_targets},
            'critical_vulnerabilities': {cve_id: stats
                                         for cve_id, stats in self.get_critical_vulnerabilities(20)},
            'critical_assets': {asset_id: stats
                                for asset_id, stats in self.get_critical_assets(20)}
        }
        with open(output_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        logger.info(f"Attack analysis exported to {output_file}")