"""
Synthetic enterprise / ICS topology generator.

Produces JSON topology files compatible with the APTArena data-loader,
following the Purdue-model zone architecture:

  * Corporate IT  (40 %)  – domain controllers, file servers, workstations
  * DMZ           (10 %)  – web servers, VPN gateways
  * Control Net   (30 %)  – HMI stations, engineering workstations, historians
  * Field Devices (20 %)  – PLCs, RTUs, sensors
"""

import json
import math
import random
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vulnerability template pool
# ---------------------------------------------------------------------------
_CVE_POOL: List[Dict[str, Any]] = [
    {"cve_id": "CVE-2024-21338", "cvss": 7.8, "cvssV3Vector": "CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H",
     "epss": 0.809, "exploit": 1, "cwe_id": ["CWE-822"], "mitre_techniques": ["T1499"], "complexity": "low"},
    {"cve_id": "CVE-2014-4113", "cvss": 7.8, "cvssV3Vector": "CVSS:3.1/AV:L/AC:L/PR:N/UI:R/S:U/C:H/I:H/A:H",
     "epss": 0.778, "exploit": 1, "cwe_id": ["CWE-269"], "mitre_techniques": ["T1548"], "complexity": "medium"},
    {"cve_id": "CVE-2023-36884", "cvss": 7.5, "cvssV3Vector": "CVSS:3.1/AV:N/AC:H/PR:N/UI:R/S:U/C:H/I:H/A:H",
     "epss": 0.931, "exploit": 1, "cwe_id": ["CWE-362"], "mitre_techniques": ["T1068", "T1203"], "complexity": "high"},
    {"cve_id": "CVE-2018-13379", "cvss": 9.8, "cvssV3Vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
     "epss": 0.975, "exploit": 1, "cwe_id": ["CWE-22"], "mitre_techniques": ["T1190"], "complexity": "low"},
    {"cve_id": "CVE-2021-34527", "cvss": 8.8, "cvssV3Vector": "CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H",
     "epss": 0.974, "exploit": 1, "cwe_id": ["CWE-269"], "mitre_techniques": ["T1068"], "complexity": "low"},
    {"cve_id": "CVE-2020-1472", "cvss": 10.0, "cvssV3Vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
     "epss": 0.976, "exploit": 1, "cwe_id": ["CWE-330"], "mitre_techniques": ["T1068", "T1003"], "complexity": "low"},
    {"cve_id": "CVE-2015-3113", "cvss": 9.8, "cvssV3Vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
     "epss": 0.975, "exploit": 1, "cwe_id": ["CWE-122"], "mitre_techniques": ["T1203"], "complexity": "low"},
    {"cve_id": "CVE-2017-11882", "cvss": 7.8, "cvssV3Vector": "CVSS:3.1/AV:L/AC:L/PR:N/UI:R/S:U/C:H/I:H/A:H",
     "epss": 0.974, "exploit": 1, "cwe_id": ["CWE-119"], "mitre_techniques": ["T1203"], "complexity": "medium"},
    {"cve_id": "CVE-2022-30190", "cvss": 7.8, "cvssV3Vector": "CVSS:3.1/AV:L/AC:L/PR:N/UI:R/S:U/C:H/I:H/A:H",
     "epss": 0.972, "exploit": 1, "cwe_id": ["CWE-610"], "mitre_techniques": ["T1566"], "complexity": "low"},
    {"cve_id": "CVE-2019-0708", "cvss": 9.8, "cvssV3Vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
     "epss": 0.975, "exploit": 1, "cwe_id": ["CWE-416"], "mitre_techniques": ["T1210"], "complexity": "low"},
    {"cve_id": "CVE-2023-44487", "cvss": 7.5, "cvssV3Vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H",
     "epss": 0.821, "exploit": 1, "cwe_id": ["CWE-400"], "mitre_techniques": ["T1499"], "complexity": "medium"},
    {"cve_id": "CVE-2021-44228", "cvss": 10.0, "cvssV3Vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
     "epss": 0.976, "exploit": 1, "cwe_id": ["CWE-502"], "mitre_techniques": ["T1190", "T1059"], "complexity": "low"},
    {"cve_id": "CVE-2022-26134", "cvss": 9.8, "cvssV3Vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
     "epss": 0.975, "exploit": 1, "cwe_id": ["CWE-917"], "mitre_techniques": ["T1190"], "complexity": "low"},
    {"cve_id": "CVE-2023-27997", "cvss": 9.8, "cvssV3Vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
     "epss": 0.462, "exploit": 1, "cwe_id": ["CWE-787"], "mitre_techniques": ["T1190"], "complexity": "medium"},
    {"cve_id": "CVE-2024-3400", "cvss": 10.0, "cvssV3Vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
     "epss": 0.943, "exploit": 1, "cwe_id": ["CWE-77"], "mitre_techniques": ["T1190", "T1059"], "complexity": "low"},
    # Lower-severity / no-exploit CVEs for variety
    {"cve_id": "CVE-2023-4966", "cvss": 7.5, "cvssV3Vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N",
     "epss": 0.534, "exploit": 0, "cwe_id": ["CWE-119"], "mitre_techniques": ["T1190"], "complexity": "medium"},
    {"cve_id": "CVE-2022-41040", "cvss": 8.8, "cvssV3Vector": "CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H",
     "epss": 0.965, "exploit": 1, "cwe_id": ["CWE-918"], "mitre_techniques": ["T1190"], "complexity": "medium"},
    {"cve_id": "CVE-2021-26855", "cvss": 9.8, "cvssV3Vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
     "epss": 0.975, "exploit": 1, "cwe_id": ["CWE-918"], "mitre_techniques": ["T1190", "T1059"], "complexity": "low"},
    {"cve_id": "CVE-2020-0688", "cvss": 8.8, "cvssV3Vector": "CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H",
     "epss": 0.974, "exploit": 1, "cwe_id": ["CWE-502"], "mitre_techniques": ["T1190"], "complexity": "medium"},
    {"cve_id": "CVE-2023-22515", "cvss": 10.0, "cvssV3Vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
     "epss": 0.971, "exploit": 1, "cwe_id": ["CWE-287"], "mitre_techniques": ["T1190"], "complexity": "low"},
]

# Asset-type templates with zone membership
_ZONE_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {
    "corporate_it": [
        {"type": "Domain Controller", "criticality": 5, "bv_range": (30000, 40000), "vulns": (4, 8), "sensitive": True},
        {"type": "File Server", "criticality": 4, "bv_range": (20000, 30000), "vulns": (3, 6), "sensitive": True},
        {"type": "Workstation", "criticality": 2, "bv_range": (8000, 15000), "vulns": (3, 8), "sensitive": False},
        {"type": "Email Server", "criticality": 4, "bv_range": (20000, 30000), "vulns": (3, 6), "sensitive": True},
    ],
    "dmz": [
        {"type": "Web Server", "criticality": 3, "bv_range": (15000, 25000), "vulns": (4, 7), "sensitive": False},
        {"type": "VPN Gateway", "criticality": 4, "bv_range": (20000, 30000), "vulns": (3, 6), "sensitive": False},
    ],
    "control_network": [
        {"type": "HMI Workstation", "criticality": 4, "bv_range": (20000, 30000), "vulns": (2, 5), "sensitive": False},
        {"type": "Engineering Workstation", "criticality": 4, "bv_range": (20000, 30000), "vulns": (3, 6), "sensitive": False},
        {"type": "Historian Database", "criticality": 3, "bv_range": (15000, 25000), "vulns": (2, 5), "sensitive": True},
    ],
    "field_devices": [
        {"type": "PLC", "criticality": 5, "bv_range": (30000, 40000), "vulns": (2, 3), "sensitive": False},
        {"type": "RTU", "criticality": 5, "bv_range": (25000, 35000), "vulns": (2, 3), "sensitive": False},
        {"type": "Sensor", "criticality": 2, "bv_range": (5000, 10000), "vulns": (1, 3), "sensitive": False},
    ],
}

_ZONE_FRACTIONS = {
    "corporate_it": 0.40,
    "dmz": 0.10,
    "control_network": 0.30,
    "field_devices": 0.20,
}


class EnterpriseTopologyGenerator:
    """Generate Purdue-model enterprise/ICS topologies at configurable scale."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._np_rng = __import__("numpy").random.default_rng(seed)

    def generate(self, num_assets: int) -> Dict[str, Any]:
        """
        Generate a topology JSON dict with *num_assets* assets (plus an
        ``internet`` node) and realistic zone-based connections.
        """
        assets_data: List[Dict] = []
        zone_assets: Dict[str, List[int]] = {z: [] for z in _ZONE_FRACTIONS}

        # Distribute assets across zones (ensure at least 1 per zone)
        zone_counts = self._distribute(num_assets)

        asset_id = 1
        for zone, count in zone_counts.items():
            templates = _ZONE_TEMPLATES[zone]
            for _ in range(count):
                tmpl = self.rng.choice(templates)
                a = self._make_asset(asset_id, tmpl)
                assets_data.append(a)
                zone_assets[zone].append(asset_id)
                asset_id += 1

        # Add internet node
        assets_data.append({
            "asset_id": "internet",
            "type": "External",
            "name": "Internet",
            "criticality_level": 0,
            "business_value": 0,
            "ip_address": "0.0.0.0",
            "mac_address": "00:00:00:00:00:00",
            "components": [],
        })

        connections = self._make_connections(zone_assets, num_assets)

        topology = {"Assets": assets_data, "Connections": connections}
        logger.info(
            f"Generated topology: {num_assets} assets, "
            f"{len(connections)} connections"
        )
        return topology

    def generate_and_save(self, num_assets: int, path: str) -> str:
        topology = self.generate(num_assets)
        with open(path, "w") as f:
            json.dump(topology, f, indent=2)
        logger.info(f"Saved topology to {path}")
        return path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _distribute(self, n: int) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        remaining = n
        zones = list(_ZONE_FRACTIONS.keys())
        for z in zones:
            counts[z] = max(1, round(n * _ZONE_FRACTIONS[z]))
        # Adjust to hit exactly n
        total = sum(counts.values())
        diff = n - total
        if diff != 0:
            z = zones[0] if diff > 0 else max(counts, key=counts.get)
            counts[z] += diff
        return counts

    def _make_asset(self, asset_id: int, tmpl: Dict) -> Dict:
        bv = self.rng.randint(*tmpl["bv_range"])
        # Clamp business value to data_loader validation range
        bv = max(5000, min(45000, bv))
        num_vulns = self.rng.randint(*tmpl["vulns"])
        vulns = self._sample_vulnerabilities(num_vulns, asset_id)

        ip_third = (asset_id // 254) + 1
        ip_fourth = (asset_id % 254) + 1

        return {
            "asset_id": asset_id,
            "type": tmpl["type"],
            "name": f"{tmpl['type']} {asset_id}",
            "criticality_level": tmpl["criticality"],
            "business_value": bv,
            "ip_address": f"192.168.{ip_third}.{ip_fourth}",
            "mac_address": f"00:11:22:33:{asset_id // 256:02x}:{asset_id % 256:02x}",
            "components": [
                {
                    "id": 1,
                    "type": "Operating System",
                    "vendor": "Generic",
                    "name": f"Component-{asset_id}-1",
                    "version": "1.0",
                    "vulnerabilities": vulns,
                }
            ],
        }

    def _sample_vulnerabilities(
        self, count: int, asset_id: int
    ) -> List[Dict]:
        pool = list(_CVE_POOL)
        self.rng.shuffle(pool)
        selected = pool[:count]
        vulns = []
        for i, tmpl in enumerate(selected):
            v = dict(tmpl)
            # Make CVE unique per asset to avoid key collisions
            v["cve_id"] = f"{tmpl['cve_id']}-A{asset_id}"
            v["component_id"] = 1
            v["likelihood"] = round(v["cvss"] / 5.0, 2)
            v["impact"] = round(v["cvss"] * 0.75, 2)
            v["scopeChanged"] = False
            v["ransomWare"] = 1 if v["cvss"] >= 9.0 else 0
            v["is_patched"] = False
            v["is_exploited"] = False
            v["exploitability"] = round(
                v["epss"] * 0.7 + v["cvss"] / 10.0 * 0.3, 4
            )
            vulns.append(v)
        return vulns

    def _make_connections(
        self, zone_assets: Dict[str, List[int]], num_assets: int
    ) -> List[Dict]:
        conns: List[Dict] = []
        seen: set = set()

        def add(src: int, dst: int, ctype: str):
            key = (src, dst)
            if key not in seen and src != dst:
                seen.add(key)
                conns.append({
                    "source_asset_id": src,
                    "destination_asset_id": dst,
                    "connection_type": ctype,
                })

        # Internet → DMZ
        for a in zone_assets["dmz"]:
            add("internet", a, "network")

        # DMZ → Corporate IT (a subset)
        for a in zone_assets["dmz"]:
            targets = self.rng.sample(
                zone_assets["corporate_it"],
                k=min(3, len(zone_assets["corporate_it"])),
            )
            for t in targets:
                add(a, t, "network")

        # DMZ → Control Network (historian / HMI via firewall)
        for a in zone_assets["dmz"]:
            targets = self.rng.sample(
                zone_assets["control_network"],
                k=min(2, len(zone_assets["control_network"])),
            )
            for t in targets:
                add(a, t, "data")

        # Intra Corporate IT
        it_ids = zone_assets["corporate_it"]
        for i, a in enumerate(it_ids):
            neighbours = self.rng.sample(
                it_ids, k=min(3, len(it_ids))
            )
            for b in neighbours:
                add(a, b, "network")

        # Corporate IT → Control Network (management)
        for a in self.rng.sample(it_ids, k=min(3, len(it_ids))):
            targets = self.rng.sample(
                zone_assets["control_network"],
                k=min(2, len(zone_assets["control_network"])),
            )
            for t in targets:
                add(a, t, "management")

        # Control Network → Field Devices
        ctrl = zone_assets["control_network"]
        field = zone_assets["field_devices"]
        for a in ctrl:
            targets = self.rng.sample(field, k=min(3, len(field)))
            for t in targets:
                add(a, t, "control")

        # Intra Control Network
        for a in ctrl:
            neighbours = self.rng.sample(ctrl, k=min(2, len(ctrl)))
            for b in neighbours:
                add(a, b, "management")

        # Intra Field Devices (sparse)
        for a in field:
            if self.rng.random() < 0.3 and len(field) > 1:
                b = self.rng.choice([x for x in field if x != a])
                add(a, b, "fieldbus")

        return conns
