"""
Canonical operational KPIs for APT3 simulation (lock to same definitions and scales as legacy table).

- Incidents: count of distinct security incidents per run (event count, not unique assets).
- MTTD: mean (t_detect - t_start) over detected incidents, in steps.
- MTTR: mean (t_recover - t_detect) over recovered incidents, in steps.
- Protected Asset Ratio (PAR): 100 * (1 - assets_compromised_any / assets_total), in 0-100.
"""

import math
from typing import List, Dict, Any, Optional


def compute_canonical_kpis(
    incident_rows: List[Dict[str, Any]],
    assets_total: int,
) -> Dict[str, Any]:
    """
    Compute canonical KPIs from per-incident rows.

    incident_rows: list of dicts with keys at least:
        asset_id, t_start, t_detect (int or None), t_recover (int or None), detected (bool), recovered (bool)
    assets_total: total number of assets in scope (denominator for PAR).

    Returns dict with:
        incidents_total, mttd_steps, mttr_steps, protected_asset_ratio_pct,
        assets_compromised_any, detected_count, recovered_count
    """
    incidents_total = len(incident_rows)
    assets_compromised_any = len(set(x["asset_id"] for x in incident_rows)) if incident_rows else 0
    par_pct = 100.0 * (1.0 - assets_compromised_any / assets_total) if assets_total else 0.0
    par_pct = max(0.0, min(100.0, par_pct))

    detected = [x for x in incident_rows if x.get("detected") and x.get("t_detect") is not None]
    recovered = [x for x in incident_rows if x.get("recovered") and x.get("t_detect") is not None and x.get("t_recover") is not None]

    if detected:
        mttd_steps = sum(x["t_detect"] - x["t_start"] for x in detected) / len(detected)
    else:
        mttd_steps = float("nan")

    if recovered:
        mttr_steps = sum(x["t_recover"] - x["t_detect"] for x in recovered) / len(recovered)
    else:
        mttr_steps = float("nan")

    return {
        "incidents_total": incidents_total,
        "assets_compromised_any": assets_compromised_any,
        "assets_total": assets_total,
        "protected_asset_ratio_pct": par_pct,
        "mttd_steps": mttd_steps,
        "mttr_steps": mttr_steps,
        "detected_count": len(detected),
        "recovered_count": len(recovered),
    }


def incident_row_for_log(incident: Dict[str, Any], run_id: int) -> Dict[str, Any]:
    """Produce one per-incident log line (JSON-serializable)."""
    return {
        "run_id": run_id,
        "incident_id": incident.get("incident_id", ""),
        "asset_id": incident.get("asset_id", ""),
        "stage": incident.get("stage", ""),
        "t_start": incident.get("t_start"),
        "t_detect": incident.get("t_detect"),   # null if never detected
        "t_recover": incident.get("t_recover"),  # null if never recovered
        "detected": incident.get("detected", False),
        "recovered": incident.get("recovered", False),
    }


def run_summary_for_log(summary: Dict[str, Any], run_id: int) -> Dict[str, Any]:
    """Produce one per-run summary line (JSON-serializable). mttd/mttr are NaN if not defined."""
    mttd = summary.get("mttd_steps")
    mttr = summary.get("mttr_steps")
    return {
        "run_id": run_id,
        "assets_total": summary.get("assets_total", 0),
        "assets_compromised_any": summary.get("assets_compromised_any", 0),
        "incidents_total": summary.get("incidents_total", 0),
        "mttd_steps": None if mttd is None or (isinstance(mttd, float) and math.isnan(mttd)) else mttd,
        "mttr_steps": None if mttr is None or (isinstance(mttr, float) and math.isnan(mttr)) else mttr,
        "protected_asset_ratio_pct": summary.get("protected_asset_ratio_pct", 0.0),
    }
