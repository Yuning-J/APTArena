"""
MITRE ATT&CK framework integration module.
Provides mapping between MITRE tactics/techniques and the Cyber Kill Chain.
Dynamically loads data from MITRE ATT&CK Excel files.
"""
from enum import Enum
from collections import defaultdict
import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KillChainStage(Enum):
    """Enumeration of the seven stages of the Cyber Kill Chain (CKC)."""
    RECONNAISSANCE = 1
    WEAPONIZATION = 2
    DELIVERY = 3
    EXPLOITATION = 4
    INSTALLATION = 5
    COMMAND_AND_CONTROL = 6
    ACTIONS_ON_OBJECTIVES = 7


class MitreTTP:
    """Represents a MITRE ATT&CK tactic and its corresponding CKC stage."""
    __slots__ = ('name', 'description', 'kill_chain_stage')
    
    def __init__(self, name: str, description: str = "", kill_chain_stage: KillChainStage = None):
        self.name = name
        self.description = description
        self.kill_chain_stage = kill_chain_stage

    def __repr__(self):
        return f"MitreTTP(name={self.name}, stage={self.kill_chain_stage.name if self.kill_chain_stage else 'Unknown'})"

# In mitre.py
class APT3TacticMapping:
    """APT3-specific TTP mappings and preferences."""
    # Known APT3 CVEs mapped to MITRE techniques
    APT3_CVE_TO_TECHNIQUE = {
        'CVE-2017-0199': ['T1566.001', 'T1204.002'],  # Spearphishing Attachment, User Execution
        'CVE-2019-0708': ['T1210'],                    # Exploitation of Remote Services
        'CVE-2017-0144': ['T1210', 'T1021.001'],       # Exploitation, Remote Desktop Protocol
        'CVE-2018-13379': ['T1190'],                   # Exploit Public-Facing Application
        'CVE-2015-3113': ['T1203']                     # Exploitation for Client Execution
    }

    # Preferred MITRE tactics and techniques by CKC stage
    APT3_PREFERRED_TTPS = {
        KillChainStage.RECONNAISSANCE: [
            ('Reconnaissance', ['T1598', 'T1595']),  # Phishing for Information, Active Scanning
        ],
        KillChainStage.WEAPONIZATION: [
            ('Phishing', ['T1566.001', 'T1566.002']),  # Spearphishing Attachment/Link
        ],
        KillChainStage.DELIVERY: [
            ('Initial Access', ['T1566.001', 'T1566.002', 'T1190']),  # Spearphishing, Exploit Public-Facing App
        ],
        KillChainStage.EXPLOITATION: [
            ('Execution', ['T1204.002', 'T1059']),      # User Execution, Command and Scripting
            ('Credential Access', ['T1003']),           # OS Credential Dumping
            ('Privilege Escalation', ['T1068']),        # Exploitation for Privilege Escalation
        ],
        KillChainStage.INSTALLATION: [
            ('Persistence', ['T1547', 'T1543']),        # Boot or Logon Autostart, Create/Modify System Process
        ],
        KillChainStage.COMMAND_AND_CONTROL: [
            ('Command and Control', ['T1071', 'T1573']), # Application Layer Protocol, Encrypted Channel
        ],
        KillChainStage.ACTIONS_ON_OBJECTIVES: [
            ('Collection', ['T1005', 'T1114']),         # Data from Local System, Email Collection
            ('Exfiltration', ['T1041']),                # Exfiltration Over C2 Channel
        ]
    }

    @staticmethod
    def get_preferred_techniques(stage: KillChainStage) -> list:
        """Return APT3-preferred techniques for a given CKC stage."""
        return [tech for _, techs in APT3TacticMapping.APT3_PREFERRED_TTPS.get(stage, []) for tech in techs]

# Mapping from MITRE ATT&CK tactics to CKC stages
mitre_to_ckc_mapping = {
    "Reconnaissance": KillChainStage.RECONNAISSANCE,
    "Resource Development": KillChainStage.WEAPONIZATION,
    "Initial Access": KillChainStage.DELIVERY,
    "Discovery": KillChainStage.DELIVERY,
    "Execution": KillChainStage.EXPLOITATION,
    "Credential Access": KillChainStage.EXPLOITATION,
    "Lateral Movement": KillChainStage.EXPLOITATION,
    "Persistence": KillChainStage.INSTALLATION,
    "Privilege Escalation": KillChainStage.INSTALLATION,
    "Defense Evasion": KillChainStage.INSTALLATION,
    "Command and Control": KillChainStage.COMMAND_AND_CONTROL,
    "Collection": KillChainStage.ACTIONS_ON_OBJECTIVES,
    "Exfiltration": KillChainStage.ACTIONS_ON_OBJECTIVES,
    "Impact": KillChainStage.ACTIONS_ON_OBJECTIVES
}

# Dictionary to store special case descriptions for non-standard tactics
# (only used if not found in MITRE data)
special_case_descriptions = {
    "Phishing": "Sending malicious emails or creating fraudulent websites.",
    "Drive-by Compromise": "Target victim visits website and malware is downloaded.",
    "Spearphishing Attachment": "Specific targeting with malicious files.",
    "Exploitation": "Exploiting software vulnerabilities."
}


class MitreMapper:
    """
    Class to generate mappings between MITRE technique IDs and tactics,
    and to build MitreTTP objects from the ATT&CK data.
    """

    def __init__(self, data_folder=None):
        """
        Initialize the mapper with the path to data files.
        
        Args:
            data_folder: Path to the folder containing MITRE ATT&CK Excel files.
                         If None, tries to locate the data folder automatically.
        """
        # Locate the data folder
        if data_folder is None:
            # Try to locate data folder automatically
            base_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(base_dir)  # Go up one level to project root
            self.data_folder = os.path.join(project_dir, 'data', 'CTI', 'raw')
        else:
            self.data_folder = data_folder
            
        # Initialize file paths and data structures
        self.techniques_file = None
        self.tactics_file = None
        self.technique_to_tactic = {}
        self.tactic_to_techniques = {}
        self.tactic_details = {}  # Stores tactic details for MitreTTP creation
        
        # Find the data files
        self._locate_files()

    def _locate_files(self):
        """Find the latest MITRE ATT&CK Excel files in the data folder."""
        if not os.path.exists(self.data_folder):
            logger.warning(f"Data folder not found: {self.data_folder}")
            return
            
        # Find technique and tactic files
        for file in os.listdir(self.data_folder):
            if file.startswith("enterprise-attack") and file.endswith("techniques.xlsx"):
                self.techniques_file = os.path.join(self.data_folder, file)
            elif file.startswith("enterprise-attack") and file.endswith("tactics.xlsx"):
                self.tactics_file = os.path.join(self.data_folder, file)
                
        if not self.techniques_file:
            logger.warning("MITRE ATT&CK techniques Excel file not found")
        if not self.tactics_file:
            logger.warning("MITRE ATT&CK tactics Excel file not found")
        
        if self.techniques_file and self.tactics_file:
            logger.info(f"Located techniques file: {os.path.basename(self.techniques_file)}")
            logger.info(f"Located tactics file: {os.path.basename(self.tactics_file)}")

    def generate_mappings(self):
        """
        Generate mappings between MITRE technique IDs and tactics,
        and load tactic details.
        
        Returns:
            tuple: (technique_to_tactic, tactic_to_techniques, tactic_details) dictionaries
        """
        logger.info("Generating MITRE technique-to-tactic mappings...")
        
        # Check if files are available
        if not self.techniques_file or not self.tactics_file:
            logger.warning("MITRE ATT&CK Excel files not found. Using default mappings.")
            return {}, {}, {}
        
        # Process the tactics data file
        try:
            tactics_df = pd.read_excel(self.tactics_file)
            logger.info(f"Loaded {len(tactics_df)} tactics from Excel")
            
            # Process the tactics data
            tactic_id_to_name = self._process_tactics_data(tactics_df)
        except Exception as e:
            logger.error(f"Error loading tactics file: {e}")
            return {}, {}, {}
        
        # Process the techniques data file
        try:
            techniques_df = pd.read_excel(self.techniques_file)
            logger.info(f"Loaded {len(techniques_df)} techniques from Excel")
            
            # Process the techniques data
            self._process_techniques_data(techniques_df, tactic_id_to_name)
        except Exception as e:
            logger.error(f"Error loading techniques file: {e}")
            return {}, {}, self.tactic_details
        
        logger.info(f"Generated mappings for {len(self.technique_to_tactic)} techniques across {len(self.tactic_to_techniques)} tactics")
        return self.technique_to_tactic, self.tactic_to_techniques, self.tactic_details
    
    def _process_tactics_data(self, tactics_df):
        """
        Process tactics data from the dataframe.
        
        Args:
            tactics_df: Pandas DataFrame with tactics data
            
        Returns:
            dict: Mapping from tactic IDs to names
        """
        tactic_id_to_name = {}
        
        # Extract tactic details for MitreTTP creation
        for _, row in tactics_df.iterrows():
            if 'ID' in row and 'name' in row:
                tactic_id = row['ID']
                tactic_name = row['name']
                
                # Get description if available
                description = row.get('description', "")
                
                # Store tactic details
                self.tactic_details[tactic_name] = {
                    'id': tactic_id,
                    'name': tactic_name,
                    'description': description
                }
                
                # Add to ID-to-name mapping
                tactic_id_to_name[tactic_id] = tactic_name
        
        return tactic_id_to_name

    def _process_techniques_data(self, techniques_df, tactic_id_to_name):
        """
        Process techniques data from the dataframe.

        Args:
            techniques_df: Pandas DataFrame with techniques data
            tactic_id_to_name: Mapping from tactic IDs to names
        """
        # Process each technique
        for _, row in techniques_df.iterrows():
            # Skip incomplete rows
            if 'ID' not in row or pd.isna(row['ID']) or 'tactics' not in row or pd.isna(row['tactics']):
                continue

            technique_id = row['ID']

            # Handle sub-techniques (e.g., T1548.001)
            is_sub_technique = '.' in technique_id

            # Get tactic(s) for this technique
            tactics = row['tactics']

            # Some tactics might be comma-separated
            if isinstance(tactics, str):
                tactic_list = [t.strip() for t in tactics.split(',')]
            else:
                tactic_list = [tactics]

            # Add to mappings
            self.technique_to_tactic[technique_id] = []

            for tactic in tactic_list:
                # Skip empty tactics
                if pd.isna(tactic) or not tactic:
                    continue

                # Get the tactic name if we have an ID
                tactic_name = tactic_id_to_name.get(tactic, tactic)

                # Add to technique-to-tactic mapping
                self.technique_to_tactic[technique_id].append(tactic_name)

                # Add to tactic-to-techniques mapping
                if tactic_name not in self.tactic_to_techniques:
                    self.tactic_to_techniques[tactic_name] = []
                if technique_id not in self.tactic_to_techniques[tactic_name]:
                    self.tactic_to_techniques[tactic_name].append(technique_id)

                # If this is a sub-technique, also add the parent technique to the tactic
                if is_sub_technique:
                    parent_id = technique_id.split('.')[0]
                    if parent_id not in self.tactic_to_techniques[tactic_name]:
                        self.tactic_to_techniques[tactic_name].append(parent_id)
    
    def generate_flat_technique_to_tactic(self):
        """
        Generate a flat technique_to_tactic dictionary for backward compatibility.
        Each technique maps to its first associated tactic (string, not list).
        
        Returns:
            dict: Technique ID to tactic name
        """
        if not self.technique_to_tactic:
            self.generate_mappings()
            
        # For compatibility with the original format, map each technique to its first tactic
        flat_mapping = {}
        for technique_id, tactics in self.technique_to_tactic.items():
            if tactics:
                flat_mapping[technique_id] = tactics[0]  # Use the first tactic
        
        return flat_mapping
    
    def generate_mitre_ttps(self, include_special_cases=True):
        """
        Generate MitreTTP objects from the tactics data.
        
        Args:
            include_special_cases: Whether to include special case TTPs not in ATT&CK data
        
        Returns:
            list: List of MitreTTP objects
        """
        if not self.tactic_details:
            _, _, self.tactic_details = self.generate_mappings()
        
        ttps = []
        
        # Create MitreTTP objects from tactic details
        for tactic_name, details in self.tactic_details.items():
            # Map tactic to kill chain stage
            kill_chain_stage = mitre_to_ckc_mapping.get(tactic_name)
            
            if kill_chain_stage:
                description = details.get('description', "")
                ttps.append(MitreTTP(tactic_name, description, kill_chain_stage))
        
        # Add special case TTPs not in ATT&CK data if requested
        if include_special_cases:
            # Special cases and techniques that are not standard tactics
            special_ttps = [
                MitreTTP("Phishing", special_case_descriptions.get("Phishing", ""), KillChainStage.WEAPONIZATION),
                MitreTTP("Drive-by Compromise", special_case_descriptions.get("Drive-by Compromise", ""), KillChainStage.DELIVERY),
                MitreTTP("Spearphishing Attachment", special_case_descriptions.get("Spearphishing Attachment", ""), KillChainStage.DELIVERY),
                MitreTTP("Exploitation", special_case_descriptions.get("Exploitation", ""), KillChainStage.EXPLOITATION),
            ]
            
            # Check for duplicates before adding
            existing_names = {ttp.name for ttp in ttps}
            for ttp in special_ttps:
                if ttp.name not in existing_names:
                    ttps.append(ttp)
        
        # Sort TTPs by kill chain stage for consistency
        ttps.sort(key=lambda x: x.kill_chain_stage.value if x.kill_chain_stage else 999)
        
        return ttps


# Initialize the mapper
_mapper = MitreMapper()

# Try to dynamically generate mappings and TTPs
try:
    # Generate mappings
    technique_to_tactic_dynamic, tactic_to_techniques_dynamic, tactic_details = _mapper.generate_mappings()
    
    # Generate dynamic mitre_ttps list
    dynamic_mitre_ttps = _mapper.generate_mitre_ttps()
    
    # Use the dynamic TTPs if available, otherwise use static definitions
    if dynamic_mitre_ttps:
        mitre_ttps = dynamic_mitre_ttps
        logger.info(f"Loaded {len(mitre_ttps)} dynamic MITRE TTPs")
    else:
        # Fall back to static definitions (defined in the except block)
        raise ValueError("No dynamic TTPs generated")
    
    # Convert technique_to_tactic to flat format if successful
    if technique_to_tactic_dynamic:
        technique_to_tactic = {}
        for technique_id, tactics in technique_to_tactic_dynamic.items():
            if tactics:
                technique_to_tactic[technique_id] = tactics[0]  # Use the first tactic
        logger.info(f"Loaded dynamic technique-to-tactic mapping with {len(technique_to_tactic)} entries")
    else:
        # Fallback to empty mapping
        technique_to_tactic = {}
        logger.warning("Using empty technique-to-tactic mapping")
except Exception as e:
    # Fallback to static definitions
    technique_to_tactic = {}
    logger.error(f"Error loading dynamic mapping: {e}")
    
    # Define static mitre_ttps for fallback
    mitre_ttps = [
        # Reconnaissance
        MitreTTP("Reconnaissance", "The attacker is actively or passively gathering information about the target.",
                 KillChainStage.RECONNAISSANCE),

        # Weaponization
        MitreTTP("Phishing", "Sending malicious emails or creating fraudulent websites.", KillChainStage.WEAPONIZATION),
        MitreTTP("Resource Development", "Develop capabilities needed for an attack.", KillChainStage.WEAPONIZATION),

        # Delivery
        MitreTTP("Initial Access", "Techniques to get into a network.", KillChainStage.DELIVERY),
        MitreTTP("Drive-by Compromise", "Target victim visits website and malware is downloaded.", KillChainStage.DELIVERY),
        MitreTTP("Spearphishing Attachment", "Specific targeting with malicious files.", KillChainStage.DELIVERY),
        MitreTTP("Discovery", "Learn about the environment.", KillChainStage.DELIVERY),

        # Exploitation
        MitreTTP("Execution", "Running attacker-controlled code.", KillChainStage.EXPLOITATION),
        MitreTTP("Exploitation", "Exploiting software vulnerabilities.", KillChainStage.EXPLOITATION),
        MitreTTP("Credential Access", "Stealing account names and passwords.", KillChainStage.EXPLOITATION),
        MitreTTP("Privilege Escalation", "Gaining higher-level permissions.", KillChainStage.EXPLOITATION),
        MitreTTP("Lateral Movement", "Move through the network.", KillChainStage.EXPLOITATION),

        # Installation
        MitreTTP("Persistence", "Maintaining access to systems.", KillChainStage.INSTALLATION),
        MitreTTP("Defense Evasion", "Avoiding detection.", KillChainStage.INSTALLATION),

        # Command and Control
        MitreTTP("Command and Control", "Communicate with compromised systems.", KillChainStage.COMMAND_AND_CONTROL),

        # Actions on Objectives
        MitreTTP("Collection", "Gather data of interest.", KillChainStage.ACTIONS_ON_OBJECTIVES),
        MitreTTP("Exfiltration", "Steal data.", KillChainStage.ACTIONS_ON_OBJECTIVES),
        MitreTTP("Impact", "Manipulate, interrupt, or destroy systems and data.", KillChainStage.ACTIONS_ON_OBJECTIVES)
    ]
    logger.warning("Using static MITRE TTPs")

# Build mapping from CKC stages to MITRE tactics
ckc_to_mitre = defaultdict(list)
for ttp in mitre_ttps:
    ckc_to_mitre[ttp.kill_chain_stage].append(ttp.name)

# Build mapping from tactic names to MitreTTP objects for easier lookup
mitre_tactics_by_name = {ttp.name: ttp for ttp in mitre_ttps}


def update_mappings(force=False):
    """
    Update the technique-to-tactic mappings and mitre_ttps.
    
    Args:
        force: If True, force regeneration of mappings.
        
    Returns:
        tuple: (technique_to_tactic mapping, mitre_ttps list)
    """
    global technique_to_tactic, mitre_ttps, ckc_to_mitre, mitre_tactics_by_name
    
    try:
        # Reinitialize the mapper to ensure fresh data
        if force:
            global _mapper
            _mapper = MitreMapper()
        
        # Generate mappings
        dynamic_mapping, _, _ = _mapper.generate_mappings()
        dynamic_ttps = _mapper.generate_mitre_ttps()
        
        # Update TTPs if available
        if dynamic_ttps:
            mitre_ttps = dynamic_ttps
            # Rebuild ckc_to_mitre and mitre_tactics_by_name
            ckc_to_mitre = defaultdict(list)
            for ttp in mitre_ttps:
                ckc_to_mitre[ttp.kill_chain_stage].append(ttp.name)
            mitre_tactics_by_name = {ttp.name: ttp for ttp in mitre_ttps}
            logger.info(f"Updated MITRE TTPs with {len(mitre_ttps)} entries")
        
        # Update technique_to_tactic if available
        if dynamic_mapping:
            # Convert to flat format for compatibility
            new_mapping = {}
            for technique_id, tactics in dynamic_mapping.items():
                if tactics:
                    new_mapping[technique_id] = tactics[0]  # Use the first tactic
            
            # Update global mapping
            technique_to_tactic = new_mapping
            logger.info(f"Updated technique-to-tactic mapping with {len(technique_to_tactic)} entries")
        
        return technique_to_tactic, mitre_ttps
    except Exception as e:
        logger.error(f"Error updating mappings: {e}")
        return technique_to_tactic, mitre_ttps
