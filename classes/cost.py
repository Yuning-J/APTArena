import logging
from typing import Optional, Dict
from classes.state import System, Asset, Vulnerability, Component

logger = logging.getLogger(__name__)


class CostCalculator:
    """
    Static utility class for calculating various costs in cybersecurity simulation.
    """

    HOURLY_RATE = 100
    UPTIME_HOURS = 8760
    DEPENDENCY_COST = 200
    ATTACKER_HOURLY_RATE = 50
    KNOWN_EXPLOIT_COST = 250
    ZERO_DAY_EXPLOIT_COST = 1000

    @staticmethod
    def parse_cvss_vector(cvss_vector: str) -> Dict[str, str]:
        """
        Parse CVSS vector string into a dictionary of metrics.

        Args:
            cvss_vector: CVSS vector string (e.g., 'CVSS:3.0/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H')

        Returns:
            dict: Dictionary of CVSS metrics
        """
        if not cvss_vector or not isinstance(cvss_vector, str):
            return {}
        try:
            metrics = cvss_vector.split('/')[1:]
            result = {}
            for metric in metrics:
                key_value = metric.split(':')
                if len(key_value) == 2:
                    key, value = key_value
                    result[key] = value
            return result
        except Exception as e:
            logger.error(f"Error parsing CVSS vector {cvss_vector}: {e}")
            return {}

    @staticmethod
    def infer_requires_reboot(vuln: Vulnerability, asset: Asset, component: Component) -> bool:
        """
        Infer whether patching a vulnerability requires a system reboot.

        Args:
            vuln: Vulnerability object
            asset: Asset object
            component: Component object

        Returns:
            bool: True if reboot is required, False otherwise
        """
        try:
            component_type = getattr(component, 'type', '').lower()
            cvss = getattr(vuln, 'cvss', 5.0)
            exploit = getattr(vuln, 'exploit', False)
            cwe_ids = getattr(vuln, 'cwe_id', [])
            system_cwes = {'CWE-362', 'CWE-122', 'CWE-787', 'CWE-416', 'CWE-843', 'CWE-911'}
            is_os_or_firmware = component_type in ['operating system', 'firmware', 'firewall appliance']
            is_critical = any(cwe in system_cwes for cwe in cwe_ids)
            requires_reboot = is_os_or_firmware and is_critical
            logger.debug(f"Inferring requires_reboot for {getattr(vuln, 'cve_id', 'unknown')}: "
                         f"Component={component_type}, CVSS={cvss}, Exploit={exploit}, "
                         f"CWEs={cwe_ids}, Requires Reboot={requires_reboot}")
            return requires_reboot
        except Exception as e:
            logger.error(f"Error inferring requires_reboot for {getattr(vuln, 'cve_id', 'unknown')}: {e}")
            return False

    @staticmethod
    def calculate_patch_cost(vuln: Vulnerability, state: System, asset: Optional[Asset] = None,
                             component_id: Optional[str] = None, base_cost: float = 1.0) -> float:
        """
        Calculate the cost of patching a vulnerability.

        Args:
            vuln: Vulnerability object
            state: System state object
            asset: Asset object (optional)
            component_id: Component ID (optional)
            base_cost: Base cost multiplier (default: 1.0)

        Returns:
            float: Patch cost in dollars
        """
        cve_id = getattr(vuln, 'cve_id', 'unknown')
        if cve_id == 'unknown' or not cve_id:
            logger.warning(f"Invalid CVE ID '{cve_id}' provided. Using default cost.")
            return 200.0 * base_cost
        if not isinstance(vuln, Vulnerability):
            logger.error(f"Invalid vulnerability type for CVE {cve_id}: {type(vuln)}")
            return 200.0 * base_cost
        if asset is not None and not isinstance(asset, Asset):
            logger.error(f"Asset parameter for CVE {cve_id} is not an Asset object but {type(asset)}")
            return 200.0 * base_cost
        component = None
        if asset is None or component_id is None:
            if state is None or not hasattr(state, 'system') or state.system is None:
                logger.warning(f"State is None or has no system attribute for CVE {cve_id}. Using default cost.")
                return 200.0 * base_cost
            try:
                for a in state.system.assets:
                    if not isinstance(a, Asset):
                        logger.warning(f"Skipping invalid asset type {type(a)}")
                        continue
                    for comp in a.components:
                        if not isinstance(comp, Component):
                            logger.warning(f"Skipping invalid component type {type(comp)}")
                            continue
                        for v in comp.vulnerabilities:
                            if v.cve_id == cve_id:
                                if component_id is None or str(getattr(comp, 'id', None)) == str(component_id):
                                    asset = a
                                    component = comp
                                    break
                        if component:
                            break
                    if asset and component:
                        break
                if not asset or not component:
                    logger.warning(f"Asset/component not found for CVE {cve_id}")
                    return 200.0 * base_cost
            except Exception as e:
                logger.error(f"Error finding asset/component for CVE {cve_id}: {e}")
                return 200.0 * base_cost
        else:
            try:
                for comp in asset.components:
                    if str(comp.id) == str(component_id):
                        component = comp
                        break
                if not component:
                    logger.warning(f"Component {component_id} not found for CVE {cve_id} in asset {asset.asset_id}")
                    return 200.0 * base_cost
            except Exception as e:
                logger.error(f"Error validating component for CVE {cve_id}: {e}")
                return 200.0 * base_cost
        vuln_key = f"{cve_id}:{asset.asset_id}:{component.id}"
        try:
            cvss = getattr(vuln, 'cvss', 5.0)
            base_cost_cvss = 50.0 + (cvss * 30.0)
            reboot_cost = 200.0 if CostCalculator.infer_requires_reboot(vuln, asset, component) else 0.0
            downtime_hours = 1.0 if reboot_cost > 0 else 0.5
            business_value = getattr(asset, 'business_value', getattr(asset, 'criticality_level', 3) * 5000)
            if not isinstance(business_value, (int, float)) or business_value < 5000 or business_value > 35000:
                logger.warning(f"Invalid business value {business_value} for asset {asset.asset_id}")
                business_value = getattr(asset, 'criticality_level', 3) * 5000
            downtime_cost = (business_value / CostCalculator.UPTIME_HOURS) * downtime_hours
            dependency_count = getattr(asset, 'dependency_count', 0)
            dependency_cost = dependency_count * CostCalculator.DEPENDENCY_COST
            total_cost = base_cost_cvss + reboot_cost + downtime_cost + dependency_cost
            total_cost = round(min(max(total_cost, 100.0), 500.0), 2)
            logger.debug(f"Patch cost for {vuln_key}: Total=${total_cost:.2f}")
            return total_cost * base_cost
        except Exception as e:
            logger.error(f"Error in patch cost calculation for {vuln_key}: {e}")
            return 200.0 * base_cost

    @staticmethod
    def calculate_exploit_cost(vuln: Vulnerability, state: Optional[System] = None, asset: Optional[Asset] = None,
                               tactic: Optional[str] = None, component_id: Optional[str] = None,
                               base_cost: float = 1.0) -> float:
        """
        Calculate the cost of exploiting a vulnerability.

        Args:
            vuln: Vulnerability object
            state: System state object (optional)
            asset: Asset object (optional)
            tactic: MITRE tactic name (optional)
            component_id: Component ID (optional)
            base_cost: Base cost multiplier (default: 1.0)

        Returns:
            float: Exploit cost in dollars
        """
        cve_id = getattr(vuln, 'cve_id', 'unknown')
        if cve_id == 'unknown' or not cve_id:
            logger.warning(f"Invalid CVE ID '{cve_id}' provided. Using default cost.")
            return 1000.0 * base_cost
        if not isinstance(vuln, Vulnerability):
            logger.error(f"Invalid vulnerability type for CVE {cve_id}: {type(vuln)}")
            return 1000.0 * base_cost
        if asset is not None and not isinstance(asset, Asset):
            logger.error(f"Asset parameter for CVE {cve_id} is not an Asset object but {type(asset)}")
            return 1000.0 * base_cost
        if asset is None and state is not None and hasattr(state, 'system') and state.system is not None:
            try:
                for a in state.system.assets:
                    for comp in a.components:
                        for v in comp.vulnerabilities:
                            if v.cve_id == cve_id:
                                if component_id is None or str(getattr(comp, 'id', None)) == str(component_id):
                                    asset = a
                                    component_id = comp.id
                                    break
                        if asset:
                            break
                if asset is None:
                    logger.warning(f"Asset not found for CVE {cve_id}. Using default cost.")
                    return 1000.0 * base_cost
            except Exception as e:
                logger.error(f"Error finding asset for CVE {cve_id}: {e}")
                return 1000.0 * base_cost
        elif asset is None:
            logger.warning(f"No asset provided and state is invalid for CVE {cve_id}. Using default cost.")
            return 1000.0 * base_cost
        try:
            if state is None or not hasattr(state, 'system') or state.system is None:
                logger.debug(
                    f"State is None or invalid for CVE {cve_id}, but asset provided. Using default patch cost.")
                patch_cost = 200.0
            else:
                patch_cost = CostCalculator.calculate_patch_cost(vuln, state, asset, component_id, base_cost)
            exploit_available = getattr(vuln, 'exploit', False)
            complexity_multiplier = 1.0
            complexity_label = "unknown"
            pr_label = "unknown"
            if hasattr(vuln, 'cvssV3Vector') and isinstance(vuln.cvssV3Vector, str):
                vector_metrics = CostCalculator.parse_cvss_vector(vuln.cvssV3Vector)
                ac_value = vector_metrics.get('AC')
                if ac_value == "L":
                    complexity_multiplier = 1.0
                    complexity_label = "low"
                elif ac_value == "H":
                    complexity_multiplier = 1.5
                    complexity_label = "high"
                pr_value = vector_metrics.get('PR')
                if pr_value == "N":
                    pr_label = "none"
                elif pr_value == "L":
                    complexity_multiplier += 0.1
                    pr_label = "low"
                elif pr_value == "H":
                    complexity_multiplier += 0.2
                    pr_label = "high"
            else:
                cvss = getattr(vuln, 'cvss', 5.0)
                complexity_multiplier = 1.0 + (cvss / 20.0)
                logger.debug(f"cvssV3Vector missing for {cve_id}. Using CVSS-based multiplier.")
            detection_risk_factor = 0.1
            if asset and getattr(asset, 'criticality_level', 2) >= 4:
                detection_risk_factor = 0.15
            elif asset:
                detection_risk_factor = 0.05
            tactic_factor = 0.0
            if tactic:
                tactic_name = getattr(tactic, 'name', str(tactic)).lower()
                if 'initial access' in tactic_name:
                    tactic_factor = 0.1
                elif 'lateral movement' in tactic_name:
                    tactic_factor = 0.05
                elif 'exfiltration' in tactic_name:
                    tactic_factor = 0.15
                elif 'impact' in tactic_name:
                    tactic_factor = 0.2
                elif 'privilege escalation' in tactic_name:
                    tactic_factor = 0.07
            hours = 5 if exploit_available else 20
            exploit_cost = CostCalculator.KNOWN_EXPLOIT_COST if exploit_available else CostCalculator.ZERO_DAY_EXPLOIT_COST
            base_exploit_cost = (hours * CostCalculator.ATTACKER_HOURLY_RATE * complexity_multiplier) + exploit_cost
            detection_risk_cost = detection_risk_factor * base_exploit_cost
            tactic_specific_cost = tactic_factor * base_exploit_cost
            max_multiplier = 5.0 if exploit_available else 10.0
            total_cost = min(base_exploit_cost + detection_risk_cost + tactic_specific_cost,
                             patch_cost * max_multiplier)
            logger.debug(f"Exploit cost for {cve_id}: Total=${total_cost:.2f}")
            return round(total_cost * base_cost, 2)
        except Exception as e:
            logger.error(f"Error in exploit cost calculation for {cve_id}: {e}")
            return 1000.0 * base_cost

    @staticmethod
    def calculate_exploit_loss(vuln: Vulnerability, state: System, asset: Optional[Asset] = None,
                               component_id: Optional[str] = None, base_cost: float = 1.0) -> float:
        """
        Calculate the potential loss from exploiting a vulnerability.

        Args:
            vuln: Vulnerability object
            state: System state object
            asset: Asset object (optional)
            component_id: Component ID (optional)
            base_cost: Base cost multiplier (default: 1.0)

        Returns:
            float: Exploit loss in dollars
        """
        cve_id = getattr(vuln, 'cve_id', 'unknown')
        if cve_id == 'unknown' or not cve_id:
            logger.warning(f"Invalid CVE ID '{cve_id}' provided. Using default loss.")
            return 1000.0 * base_cost
        if not isinstance(vuln, Vulnerability):
            logger.error(f"Invalid vulnerability type for CVE {cve_id}: {type(vuln)}")
            return 1000.0 * base_cost
        if asset is not None and not isinstance(asset, Asset):
            logger.error(f"Asset parameter for CVE {cve_id} is not an Asset object but {type(asset)}")
            return 1000.0 * base_cost
        try:
            if asset is None:
                if state is None or not hasattr(state, 'system') or state.system is None:
                    logger.warning(f"State is None or has no system attribute for CVE {cve_id}. Using default loss.")
                    return 1000.0 * base_cost
                for a in state.system.assets:
                    for comp in a.components:
                        for v in comp.vulnerabilities:
                            if v.cve_id == cve_id and (
                                    component_id is None or str(getattr(comp, 'id', None)) == str(component_id)):
                                asset = a
                                component_id = comp.id
                                break
                        if asset:
                            break
                if asset is None:
                    logger.warning(f"Asset not found for CVE {cve_id}. Using default loss.")
                    return 1000.0 * base_cost
            business_value = getattr(asset, 'business_value', getattr(asset, 'criticality_level', 3) * 5000)
            if not isinstance(business_value, (int, float)) or business_value < 5000 or business_value > 45000:
                logger.warning(f"Invalid business value {business_value} for asset {asset.asset_id}")
                business_value = getattr(asset, 'criticality_level', 3) * 5000
            asset_vulns = []
            for comp in asset.components:
                asset_vulns.extend(comp.vulnerabilities)
            if not asset_vulns:
                logger.warning(f"No vulnerabilities found for asset {asset.asset_id}. Using default loss.")
                return 1000.0 * base_cost

            def get_impact_score(v):
                impact_score = 0.0
                cvss_vector = getattr(v, 'cvssV3Vector', None)
                if cvss_vector and isinstance(cvss_vector, str):
                    vector_metrics = CostCalculator.parse_cvss_vector(cvss_vector)
                    for metric in ['C', 'I', 'A']:
                        value = vector_metrics.get(metric, 'N')
                        if value == 'H':
                            impact_score += 0.56
                        elif value == 'M':
                            impact_score += 0.39
                        elif value == 'L':
                            impact_score += 0.22
                else:
                    cvss = getattr(v, 'cvss', 5.0)
                    impact_score = cvss / 3.33
                return max(impact_score, 0.1)

            vuln_impact_score = get_impact_score(vuln)
            total_impact_score = sum(get_impact_score(v) for v in asset_vulns)
            impact_weight = vuln_impact_score / total_impact_score if total_impact_score > 0 else 1.0 / len(asset_vulns)
            scope_multiplier = 1.0
            scope_label = "unchanged"
            cvss_vector = getattr(vuln, 'cvssV3Vector', None)
            if cvss_vector and isinstance(cvss_vector, str):
                vector_metrics = CostCalculator.parse_cvss_vector(cvss_vector)
                scope = vector_metrics.get('S')
                if scope == 'C':
                    scope_multiplier = 1.5
                    scope_label = "changed"
                elif scope == 'U':
                    scope_label = "unchanged"
            else:
                logger.debug(f"cvssV3Vector missing for {cve_id}. Using default scope multiplier.")
            impact_multiplier = impact_weight * scope_multiplier
            min_loss = 10.0
            expected_loss = business_value * impact_multiplier
            total_loss = max(expected_loss, min_loss)
            total_loss = round(total_loss, 2)
            setattr(vuln, 'calculated_loss', total_loss)
            logger.debug(f"Exploit loss for {cve_id}: Total=${total_loss:.2f}")
            return total_loss * base_cost
        except Exception as e:
            logger.error(f"Error in exploit loss calculation for {cve_id}: {e}")
            return 1000.0 * base_cost

    @staticmethod
    def calculate_risk_to_cost_ratio(vuln: Vulnerability, state: System, asset: Optional[Asset] = None,
                                     component_id: Optional[str] = None, cost_cache: Optional[Dict] = None) -> float:
        """
        Calculate the risk-to-cost ratio for a vulnerability.

        Args:
            vuln: Vulnerability object
            state: System state object
            asset: Asset object (optional)
            component_id: Component ID (optional)
            cost_cache: Cost cache dictionary (optional)

        Returns:
            float: Risk-to-cost ratio
        """
        cve_id = getattr(vuln, 'cve_id', 'unknown')
        if cve_id == 'unknown' or not cve_id:
            logger.warning(f"Invalid CVE ID '{cve_id}' provided. Returning default ratio.")
            return 0.0
        if not isinstance(vuln, Vulnerability):
            logger.error(f"Invalid vulnerability type for CVE {cve_id}: {type(vuln)}")
            return 0.0
        if asset is not None and not isinstance(asset, Asset):
            logger.error(f"Asset parameter for CVE {cve_id} is not an Asset object but {type(asset)}")
            return 0.0
        try:
            if asset is None or component_id is None:
                if state is None or not hasattr(state, 'system') or state.system is None:
                    logger.warning(
                        f"State is None or has no system attribute for CVE {cve_id}. Returning default ratio.")
                    return 0.0
                for a in state.system.assets:
                    for comp in a.components:
                        for v in comp.vulnerabilities:
                            if v.cve_id == cve_id:
                                if component_id is None or str(getattr(comp, 'id', None)) == str(component_id):
                                    asset = a
                                    component_id = comp.id
                                    break
                        if asset:
                            break
                if asset is None:
                    logger.warning(f"Asset not found for CVE {cve_id}. Returning default ratio.")
                    return 0.0
            cache_key = f"{cve_id}:{asset.asset_id}:{component_id}"
            if cost_cache and cache_key in cost_cache.get('vulnerability_info', {}):
                vuln_info = cost_cache['vulnerability_info'].get(cache_key, {})
                cached_asset = vuln_info.get('asset')
                if cached_asset and isinstance(cached_asset, Asset):
                    patch_cost = cost_cache.get('patch_costs', {}).get(cache_key, 200.0)
                    exploit_loss = cost_cache.get('exploit_losses', {}).get(cache_key, 0.0)
                    epss = vuln_info.get('epss', getattr(vuln, 'epss', 0.1))
                    risk = exploit_loss * epss
                    ratio = risk / patch_cost if patch_cost > 0 else risk * 10
                    logger.debug(f"Risk-to-cost ratio for {cache_key}: Ratio={ratio:.2f}")
                    return ratio
                else:
                    logger.warning(f"Invalid cached asset for {cache_key}: {type(cached_asset)}")
            likelihood = getattr(vuln, 'epss', getattr(vuln, 'exploitability', 0.5))
            exploit_loss = CostCalculator.calculate_exploit_loss(vuln, state, asset, component_id)
            risk = exploit_loss * likelihood
            patch_cost = CostCalculator.calculate_patch_cost(vuln, state, asset, component_id)
            if cost_cache:
                cost_cache.setdefault('patch_costs', {})[cache_key] = patch_cost
                cost_cache.setdefault('exploit_losses', {})[cache_key] = exploit_loss
                cost_cache.setdefault('vulnerability_info', {}).setdefault(cache_key, {}).update({
                    'epss': likelihood,
                    'asset': asset,
                    'component_id': component_id,
                    'risk_to_cost_ratio': risk / patch_cost if patch_cost > 0 else risk * 10
                })
            ratio = risk / patch_cost if patch_cost > 0 else risk * 10
            logger.debug(f"Risk-to-cost ratio for {cache_key}: Ratio={ratio:.2f}")
            return ratio
        except Exception as e:
            logger.error(f"Error in risk-to-cost ratio calculation for {cve_id}: {e}")
            return 0.0