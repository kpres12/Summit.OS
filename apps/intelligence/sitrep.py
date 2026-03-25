"""
KOFA Situation Report (SITREP) generator.

Produces structured natural-language situation reports from the advisory stream.
Two paths:

  Template path (always available, <1ms):
    Groups advisories by class and severity, generates a concise SITREP
    with a headline, per-domain findings, and a recommended action.

  LLM path (optional, requires Ollama):
    Sends the template output + raw advisory context to the local LLM for a
    richer narrative. Falls back transparently to template if Ollama is down.

Output schema (also returned as JSON from the /sitrep endpoint):
  {
    "sitrep_id":          "uuid",
    "generated_at":       "ISO-8601",
    "time_window_s":      300,
    "advisory_count":     12,
    "summary":            "CRITICAL: active fire detected NE of coordinates ...",
    "findings": [
      {
        "domain":         "fire_smoke",
        "count":          3,
        "max_severity":   "CRITICAL",
        "avg_confidence": 0.88,
        "classes":        ["smoke", "fire"],
        "description":    "3 fire/smoke detections at high confidence ..."
      },
      ...
    ],
    "recommended_action": "Dispatch SURVEY UAV to highest-confidence fire location ...",
    "highest_risk":       "CRITICAL",
    "generated_by":       "kofa-template"   # or "kofa-llm"
  }
"""

import json
import logging
import os
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("kofa.sitrep")

OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

# ── severity ordering ──────────────────────────────────────────────────────────
_SEV_ORDER = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}

# ── domain keyword map (matches features.py groups) ───────────────────────────
_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "fire_smoke":       ["fire", "smoke", "flame", "wildfire", "ember", "hotspot", "burning", "blaze"],
    "person_sar":       ["person", "missing", "survivor", "victim", "casualty", "stranded", "mayday", "sos", "distress"],
    "flood_water":      ["flood", "inundation", "surge", "tsunami", "submerged", "rising water"],
    "structural":       ["collapse", "rubble", "earthquake", "landslide", "mudslide", "sinkhole"],
    "vehicle":          ["vehicle", "vessel", "aircraft", "drone", "uav", "boat", "ship"],
    "hazmat":           ["hazmat", "chemical", "spill", "leak", "toxic", "radiation", "nuclear", "plume"],
    "wildlife":         ["animal", "wildlife", "bear", "shark", "crocodile", "tiger", "elephant"],
    "infrastructure":   ["pipeline", "power line", "bridge", "dam", "substation", "tower"],
    "agricultural":     ["crop", "field", "blight", "drought", "pest", "irrigation"],
    "medical":          ["medical", "injury", "epidemic", "outbreak", "cardiac", "triage"],
    "security":         ["intrusion", "unauthorized", "trespass", "armed", "hostile", "breach"],
    "logistics":        ["delivery", "aid", "cargo", "resupply", "humanitarian"],
}

# ── recommended actions by domain × severity ──────────────────────────────────
_ACTIONS: Dict[str, Dict[str, str]] = {
    "fire_smoke": {
        "CRITICAL": "Dispatch SURVEY UAV immediately to confirm fire extent. Alert ground resources. Establish exclusion zone.",
        "HIGH":     "Dispatch SURVEY UAV for confirmation. Pre-position ground crews.",
        "default":  "Monitor with UAV. No immediate ground response required.",
    },
    "person_sar": {
        "CRITICAL": "Launch SEARCH grid pattern immediately. Notify SAR coordinator. Request medical standby.",
        "HIGH":     "Dispatch MONITOR UAV to last known position. Initiate contact attempts.",
        "default":  "Log and monitor. Escalate if no contact within 30 minutes.",
    },
    "flood_water": {
        "CRITICAL": "Dispatch fixed-wing SURVEY for area extent mapping. Alert evacuation teams.",
        "HIGH":     "UAV SURVEY to confirm water levels. Monitor levee/drainage infrastructure.",
        "default":  "Monitor water levels. No immediate action required.",
    },
    "hazmat": {
        "CRITICAL": "Establish PERIMETER immediately. Evacuate 500m radius. Notify hazmat response team.",
        "HIGH":     "PERIMETER UAV to map plume extent. Alert hazmat coordinators.",
        "default":  "Monitor with UAV. Assess wind direction for plume modeling.",
    },
    "structural": {
        "CRITICAL": "SEARCH pattern for trapped survivors. Coordinate heavy rescue assets.",
        "HIGH":     "SURVEY damage extent. Hold perimeter pending structural assessment.",
        "default":  "INSPECT with close-pass UAV. Engineer assessment recommended.",
    },
    "infrastructure": {
        "CRITICAL": "INSPECT immediately. Notify utility operator. Isolate affected segment.",
        "HIGH":     "INSPECT with UAV. Report findings to asset owner.",
        "default":  "Schedule INSPECT mission during next available window.",
    },
}

_DEFAULT_ACTION = "Continue monitoring. Escalate if situation develops."


def _classify_domain(cls_str: str) -> str:
    lower = cls_str.lower()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return domain
    return "other"


# ── data classes ───────────────────────────────────────────────────────────────

@dataclass
class Finding:
    domain:         str
    count:          int
    max_severity:   str
    avg_confidence: float
    classes:        List[str]
    description:    str
    locations:      List[Dict[str, float]] = field(default_factory=list)


@dataclass
class SitRep:
    sitrep_id:        str
    generated_at:     str
    time_window_s:    int
    advisory_count:   int
    summary:          str
    findings:         List[Finding]
    recommended_action: str
    highest_risk:     str
    generated_by:     str   # "kofa-template" | "kofa-llm"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["findings"] = [asdict(f) for f in self.findings]
        return d


# ── template generator ─────────────────────────────────────────────────────────

class SitRepGenerator:

    def from_advisories(
        self,
        advisories: List[Dict[str, Any]],
        time_window_s: int = 300,
    ) -> SitRep:
        """
        Build a SITREP from a list of advisory dicts.
        Each dict must have: risk_level, message, confidence, ts (optional).
        """
        if not advisories:
            return SitRep(
                sitrep_id       = str(uuid.uuid4()),
                generated_at    = datetime.now(timezone.utc).isoformat(),
                time_window_s   = time_window_s,
                advisory_count  = 0,
                summary         = "No active advisories in the current window.",
                findings        = [],
                recommended_action = "Continue normal operations.",
                highest_risk    = "LOW",
                generated_by    = "kofa-template",
            )

        # ── group by domain ────────────────────────────────────────────────
        domain_groups: Dict[str, List[Dict]] = defaultdict(list)
        for adv in advisories:
            # Extract class from advisory message (format: "RISK risk: CLASS detected ...")
            msg   = adv.get("message", "")
            parts = msg.split(":")
            cls   = parts[1].strip().split(" detected")[0].strip() if len(parts) > 1 else "unknown"
            domain = _classify_domain(cls)
            domain_groups[domain].append({**adv, "_cls": cls})

        # ── build per-domain findings ──────────────────────────────────────
        findings: List[Finding] = []
        for domain, items in domain_groups.items():
            severities  = [i.get("risk_level", "LOW") for i in items]
            confs       = [float(i.get("confidence", 0)) for i in items]
            classes     = list({i["_cls"] for i in items})
            max_sev     = max(severities, key=lambda s: _SEV_ORDER.get(s, 0))
            avg_conf    = sum(confs) / len(confs) if confs else 0.0

            description = _describe_finding(domain, items, max_sev, avg_conf)

            findings.append(Finding(
                domain         = domain,
                count          = len(items),
                max_severity   = max_sev,
                avg_confidence = round(avg_conf, 3),
                classes        = classes,
                description    = description,
            ))

        # Sort findings: CRITICAL first, then HIGH, then count
        findings.sort(
            key=lambda f: (_SEV_ORDER.get(f.max_severity, 0) * -1, -f.count)
        )

        # ── overall headline ───────────────────────────────────────────────
        highest_risk   = findings[0].max_severity if findings else "LOW"
        top_finding    = findings[0] if findings else None
        summary        = _make_summary(top_finding, len(advisories), highest_risk)

        # ── recommended action ─────────────────────────────────────────────
        recommended    = _make_recommendation(findings)

        return SitRep(
            sitrep_id          = str(uuid.uuid4()),
            generated_at       = datetime.now(timezone.utc).isoformat(),
            time_window_s      = time_window_s,
            advisory_count     = len(advisories),
            summary            = summary,
            findings           = findings,
            recommended_action = recommended,
            highest_risk       = highest_risk,
            generated_by       = "kofa-template",
        )

    async def enhance_with_llm(self, sitrep: SitRep, raw_advisories: List[Dict]) -> SitRep:
        """
        Send the template SITREP + raw context to Ollama for narrative enhancement.
        Returns original sitrep unchanged if Ollama is unavailable.
        """
        try:
            import httpx

            # Build a compact context string
            context_lines = [
                f"KOFA SITREP — {sitrep.generated_at}",
                f"Window: {sitrep.time_window_s}s | Advisories: {sitrep.advisory_count} | Highest risk: {sitrep.highest_risk}",
                "",
                "Findings:",
            ]
            for f in sitrep.findings:
                context_lines.append(
                    f"  [{f.max_severity}] {f.domain}: {f.count} detections "
                    f"({', '.join(f.classes[:3])}) — avg conf {f.avg_confidence:.0%}"
                )

            context_lines += [
                "",
                f"Template summary: {sitrep.summary}",
                f"Template action:  {sitrep.recommended_action}",
                "",
                "Task: Rewrite the summary and recommended_action as a crisp, "
                "professional military-style SITREP. Use active voice. "
                "Be specific about what was detected, where, and what to do next. "
                "Maximum 3 sentences for summary. Maximum 2 sentences for recommended_action. "
                "Return ONLY a JSON object with keys 'summary' and 'recommended_action'.",
            ]

            prompt = "\n".join(context_lines)

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model":  OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.15, "num_predict": 256},
                    },
                )
                if resp.status_code != 200:
                    return sitrep

                raw = resp.json().get("response", "")
                # Parse the JSON block from the LLM response
                start = raw.find("{")
                end   = raw.rfind("}") + 1
                if start >= 0 and end > start:
                    parsed = json.loads(raw[start:end])
                    llm_summary = parsed.get("summary", "").strip()
                    llm_action  = parsed.get("recommended_action", "").strip()
                    if llm_summary:
                        sitrep.summary             = llm_summary
                        sitrep.recommended_action  = llm_action or sitrep.recommended_action
                        sitrep.generated_by        = "kofa-llm"

        except Exception as exc:
            logger.debug("LLM SITREP enhancement failed (using template): %s", exc)

        return sitrep


# ── description helpers ────────────────────────────────────────────────────────

def _describe_finding(
    domain: str,
    items: List[Dict],
    max_sev: str,
    avg_conf: float,
) -> str:
    count = len(items)
    cls_list = ", ".join(sorted({i["_cls"] for i in items})[:3])

    _DOMAIN_LABELS = {
        "fire_smoke":     "fire/smoke",
        "person_sar":     "person/SAR",
        "flood_water":    "flood/water",
        "structural":     "structural damage",
        "vehicle":        "vehicle/vessel",
        "hazmat":         "hazmat/chemical",
        "wildlife":       "wildlife",
        "infrastructure": "infrastructure damage",
        "agricultural":   "agricultural anomaly",
        "medical":        "medical/health",
        "security":       "security/intrusion",
        "logistics":      "logistics request",
        "other":          "unclassified",
    }

    label = _DOMAIN_LABELS.get(domain, domain)
    return (
        f"{count} {label} detection{'s' if count > 1 else ''} "
        f"({cls_list}) at {avg_conf:.0%} avg confidence. "
        f"Highest severity: {max_sev}."
    )


def _make_summary(
    top: Optional[Finding],
    total: int,
    highest_risk: str,
) -> str:
    if top is None:
        return "No significant detections in current window."
    cls_str = top.classes[0] if top.classes else top.domain
    count_str = f"{total} advisory" if total == 1 else f"{total} advisories"
    return (
        f"{highest_risk}: {cls_str} is the highest-priority event. "
        f"KOFA recorded {count_str} in the current window across "
        f"{top.domain.replace('_', ' ')} and {len([top])} other domain(s). "
        f"Operator action required."
        if highest_risk in ("CRITICAL", "HIGH")
        else
        f"Situation nominal. {count_str} in current window. "
        f"Highest-priority detection: {cls_str} ({top.max_severity})."
    )


def _make_recommendation(findings: List[Finding]) -> str:
    if not findings:
        return _DEFAULT_ACTION

    top = findings[0]
    domain_actions = _ACTIONS.get(top.domain, {})
    action = (
        domain_actions.get(top.max_severity)
        or domain_actions.get("default")
        or _DEFAULT_ACTION
    )

    # Append secondary domain note if multiple high-priority domains
    secondary = [
        f for f in findings[1:]
        if _SEV_ORDER.get(f.max_severity, 0) >= _SEV_ORDER.get("HIGH", 2)
    ]
    if secondary:
        sec_domains = ", ".join(f.domain.replace("_", " ") for f in secondary[:2])
        action += f" Additionally, monitor active {sec_domains} situation(s)."

    return action


# ── singleton ──────────────────────────────────────────────────────────────────

_generator: SitRepGenerator | None = None

def get_sitrep_generator() -> SitRepGenerator:
    global _generator
    if _generator is None:
        _generator = SitRepGenerator()
    return _generator
