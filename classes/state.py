# state.py
from typing import Union, Optional
from classes.mitre import KillChainStage  # Import from mitre.py
from classes.mitre import APT3TacticMapping
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_vuln_key(vuln_id, asset_id, component_id="0"):
    """
    Create a standardized vulnerability key.

    Args:
        vuln_id: The CVE ID or vulnerability identifier
        asset_id: The asset identifier
        component_id: Component identifier (default "0" if not provided)

    Returns:
        str: A standardized vulnerability key in the format {vuln_id}:{asset_id}:{component_id}
    """
    # Ensure all parts are converted to strings
    vuln_id_str = str(vuln_id) if vuln_id is not None else "unknown"
    asset_id_str = str(asset_id) if asset_id is not None else "unknown"
    component_id_str = str(component_id) if component_id is not None else "0"

    # Create the key
    return f"{vuln_id_str}:{asset_id_str}:{component_id_str}"

class Vulnerability:
    def __init__(self, cve_id: str = "", cvss: float = 0.0, cvssV3Vector: str = "",
                 scopeChanged: bool = False, likelihood: float = 0.0, impact: float = 0.0,
                 exploit: bool = False, epss: float = 0.0, ransomWare: bool = False,
                 component_id: str = "", is_patched: bool = False, is_exploited: bool = False,
                 cwe_id: str = "", exploitability: float = 0.0, mitre_techniques: list = None,
                 complexity: str = None, requires_reboot: bool = False,
                 _is_patched: bool = False, _is_exploited: bool = False,
                 patch_complexity: str = None, disruption_minutes: int = 0):
        self.cve_id = cve_id
        self.cvss = cvss
        self.cvssV3Vector = cvssV3Vector
        self.scopeChanged = scopeChanged
        self.likelihood = likelihood
        self.impact = impact
        self.exploit = exploit
        self.epss = epss
        self.ransomWare = ransomWare
        self.component_id = component_id
        self.is_patched = is_patched
        self.is_exploited = is_exploited
        self.cwe_id = cwe_id
        self.exploitability = exploitability
        self.mitre_techniques = mitre_techniques or []
        self.exploit_likelihood = exploitability
        self.complexity = complexity
        self.requires_reboot = requires_reboot
        self.patch_complexity = patch_complexity
        self.disruption_minutes = disruption_minutes
        self._is_patched = is_patched
        self._is_exploited = is_exploited

    def __repr__(self):
        return f"Vulnerability(cve_id={self.cve_id}, cvss={self.cvss}, patched={self.is_patched}, exploited={self.is_exploited})"

    @property
    def is_patched(self):
        return self._is_patched

    @is_patched.setter
    def is_patched(self, value):
        self._is_patched = value
        if value and self._is_exploited:
            self._is_exploited = False

    @property
    def is_exploited(self):
        return self._is_exploited

    @is_exploited.setter
    def is_exploited(self, value):
        self._is_exploited = value

    def apply_patch(self):
        self.is_patched = True

    def mark_as_exploited(self):
        if not self.is_patched:
            self.is_exploited = True
            return True
        return False

class Component:
    def __init__(self, comp_id: str = "", comp_type: str = "", vendor: str = "",
                 name: str = "", version: str = "", embedded_in: str = None):
        self.id = comp_id
        self.type = comp_type
        self.vendor = vendor
        self.name = name
        self.version = version
        self.embedded_in = embedded_in
        self.vulnerabilities = []

    def add_vulnerability(self, vulnerability: Vulnerability):
        self.vulnerabilities.append(vulnerability)

    def __repr__(self):
        return f"Component(id={self.id}, name={self.name}, version={self.version})"

class Asset:
    def __init__(self, asset_id: str = "", asset_type: str = "", name: str = "",
                 criticality_level: int = 0, ip_address: str = "0.0.0.0",
                 mac_address: str = "00:00:00:00:00:00", business_value: float = 0.0,
                 centrality: float = 0.5, security_controls: int = 0,
                 contains_sensitive_data: bool = False, dependency_count: int = 0):
        self.id = asset_id
        self.asset_id = asset_id
        self.type = asset_type
        self.name = name
        self.criticality_level = criticality_level
        self.business_value = business_value
        self.ip_address = ip_address
        self.mac_address = mac_address
        self.centrality = centrality
        self.security_controls = security_controls
        self.contains_sensitive_data = contains_sensitive_data
        self.dependency_count = dependency_count
        self.updated_criticality = criticality_level
        self.final_criticality = criticality_level
        self.total_propagation_risk = 0
        self.components = []
        self.adjacency_matrix = []
        self.vulnerabilities = []
        self._compromise_time = None
        self._last_attack_time = None
        self._is_compromised = False

    def mark_as_compromised(self, compromised=True):
        self._is_compromised = compromised

    def add_component(self, component: Component):
        self.components.append(component)

    def set_adjacency_matrix(self, adjacency_matrix: list):
        self.adjacency_matrix = adjacency_matrix

    def __repr__(self):
        return f"Asset(id={self.id}, name={self.name}, criticality={self.criticality_level}, business_value={self.business_value})"

    @property
    def is_compromised(self):
        return self._is_compromised

    def record_compromise(self, time_step):
        self._is_compromised = True
        if self._compromise_time is None:
            self._compromise_time = time_step

    def record_attack(self, time_step):
        self._last_attack_time = time_step

    @property
    def time_since_compromise(self, current_time_step):
        if self._compromise_time is None:
            return None
        return current_time_step - self._compromise_time

    @property
    def time_since_last_attack(self, current_time_step):
        if self._last_attack_time is None:
            return None
        return current_time_step - self._last_attack_time

class Connection:
    def __init__(self, from_asset: Asset, to_asset: Asset, connection_type: str = "",
                 bidirectional: bool = False, bandwidth: float = 1.0):
        self.from_asset = from_asset
        self.to_asset = to_asset
        self.connection_type = connection_type
        self.bidirectional = bidirectional
        self.bandwidth = bandwidth

    def __repr__(self):
        direction = "<->" if self.bidirectional else "->"
        return f"Connection({self.from_asset.id} {direction} {self.to_asset.id}, type={self.connection_type})"

class System:
    def __init__(self):
        self.assets = []
        self.connections = []
        self.time_step = 0
        self.action_history = []

    def add_asset(self, asset: Asset):
        self.assets.append(asset)

    def add_connection(self, connection: Connection):
        self.connections.append(connection)

    def increment_time(self):
        self.time_step += 1
        return self.time_step

    def get_connections_from(self, asset_id):
        return [conn for conn in self.connections if conn.from_asset.id == asset_id]

    def get_connections_to(self, asset_id):
        return [conn for conn in self.connections if conn.to_asset.id == asset_id]

    def get_asset_by_id(self, asset_id):
        for asset in self.assets:
            if asset.id == asset_id:
                return asset
        return None

    def __repr__(self):
        return f"System(assets={len(self.assets)}, connections={len(self.connections)}, time_step={self.time_step})"

class State:
    def __init__(self, system: System, k: Union[int, KillChainStage] = None, attacker_belief=None,
                 defender_belief=None, attacker_cost: float = 0.0, defender_cost: float = 0.0,
                 detection_prob: float = 0.1):
        """
        Initialize the state with a kill chain stage as an integer.

        Args:
            system: The system object containing assets and connections
            k: Kill chain stage (integer or KillChainStage enum)
            attacker_belief: Attacker's belief state
            defender_belief: Defender's belief state
            attacker_cost: Accumulated attacker cost
            defender_cost: Accumulated defender cost
            detection_prob: Current detection probability
        """
        self.system = system
        if k is None:
            self.k = KillChainStage.RECONNAISSANCE.value
        elif isinstance(k, KillChainStage):
            self.k = k.value  # Store as integer
        else:
            self.k = int(k)  # Ensure k is an integer
        assert isinstance(self.k, int), f"State.k must be an integer, got {type(self.k)}"
        self.attacker_belief = attacker_belief
        self.defender_belief = defender_belief
        self.attacker_cost = attacker_cost
        self.defender_cost = defender_cost
        self.detection_prob = detection_prob
        self.attacker_suggested_stage: Optional[KillChainStage] = None
        self.attack_history = []
        self.defense_history = []
        self.temp_attacker_actions = []
        self.temp_successful_exploits = 0
        self.lateral_movement_targets = []
        self.lateral_movement_chain = []

    def record_attack(self, attacker_actions, success_count=0):
        self.attack_history.append({
            'time_step': self.system.time_step,
            'kill_chain_stage': self.k,
            'actions': attacker_actions,
            'success_count': success_count
        })
        for vuln in attacker_actions:
            for asset in self.system.assets:
                for comp in asset.components:
                    if vuln in comp.vulnerabilities:
                        asset.record_attack(self.system.time_step)
                        break

    def record_defense(self, defender_actions):
        self.defense_history.append({
            'time_step': self.system.time_step,
            'kill_chain_stage': self.k,
            'actions': defender_actions
        })

    def update_kill_chain_stage(self):
        """Update kill chain stage based on system state."""
        current_stage_value = int(self.k)  # k is an integer
        compromised_assets = [a for a in self.system.assets if a.is_compromised]
        compromised_count = len(compromised_assets)
        new_stage_value = current_stage_value
        if compromised_count > 0:
            if compromised_count >= 3:
                new_stage_value = max(current_stage_value, KillChainStage.ACTIONS_ON_OBJECTIVES.value)
            elif compromised_count >= 2:
                new_stage_value = max(current_stage_value, KillChainStage.COMMAND_AND_CONTROL.value)
            else:
                new_stage_value = max(current_stage_value, KillChainStage.INSTALLATION.value)
        elif hasattr(self, 'attack_history') and len(self.attack_history) > 0:
            if any(record['success_count'] > 0 for record in self.attack_history[-3:]):
                new_stage_value = max(current_stage_value, KillChainStage.EXPLOITATION.value)
            else:
                new_stage_value = max(current_stage_value, KillChainStage.DELIVERY.value)
        else:
            new_stage_value = KillChainStage.RECONNAISSANCE.value
        self.k = new_stage_value  # Store as integer

    def process_attacker_stage_suggestion(self):
        """Process attacker-suggested kill chain stage."""
        if self.attacker_suggested_stage is None:
            return
        current_stage_value = int(self.k)  # k is an integer
        try:
            suggested_stage_value = (self.attacker_suggested_stage.value
                                     if isinstance(self.attacker_suggested_stage, KillChainStage)
                                     else int(self.attacker_suggested_stage))
        except (AttributeError, ValueError):
            logger.warning(f"Invalid suggested stage {self.attacker_suggested_stage}. Ignoring suggestion")
            self.attacker_suggested_stage = None
            return
        if suggested_stage_value > current_stage_value:
            self.k = suggested_stage_value  # Store as integer
        self.attacker_suggested_stage = None

    def suggest_attacker_stage(self, stage: Union[int, KillChainStage]):
        """Suggest a kill chain stage for the attacker."""
        if isinstance(stage, KillChainStage):
            self.attacker_suggested_stage = stage
        else:
            try:
                self.attacker_suggested_stage = KillChainStage(stage)
            except ValueError:
                logger.warning(f"Invalid stage suggestion {stage}, ignoring")
                return
        logger.debug(f"Suggested attacker stage: {self.attacker_suggested_stage.name}")

    def compute_phi(self):
        """Compute system risk metrics."""
        total_vulns = 0
        unpatched_vulns = 0
        high_risk_vulns = 0
        for asset in self.system.assets:
            for comp in asset.components:
                for vuln in comp.vulnerabilities:
                    total_vulns += 1
                    if not vuln.is_patched:
                        unpatched_vulns += 1
                        if vuln.cvss >= 7.0:
                            high_risk_vulns += 1
        total_risk = sum(v.cvss for asset in self.system.assets
                        for comp in asset.components
                        for v in comp.vulnerabilities if not v.is_patched)
        compromised_assets = [a for a in self.system.assets if a.is_compromised]
        num_compromised = len(compromised_assets)
        high_value_compromised = sum(1 for a in compromised_assets
                                   if hasattr(a, 'business_value') and a.business_value >= 7)
        critical_compromised = sum(1 for a in compromised_assets
                                 if a.criticality_level >= 4)
        return {
            "total_risk": total_risk,
            "num_compromised": num_compromised,
            "high_value_compromised": high_value_compromised,
            "critical_compromised": critical_compromised,
            "unpatched_vulns": unpatched_vulns,
            "high_risk_vulns": high_risk_vulns,
            "patch_percentage": (1.0 - (unpatched_vulns / total_vulns)) * 100 if total_vulns > 0 else 100,
            "kill_chain_stage": self.k,
            "time_step": self.system.time_step
        }
