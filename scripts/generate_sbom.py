#!/usr/bin/env python3
"""
SBOM Generator — CycloneDX format

Generates a CycloneDX-1.5 JSON SBOM for the Heli.OS codebase. Walks the
Python and Node.js dependency manifests, plus enumerates ML model
artifacts and key first-party Python modules so they appear as
"application" components rather than just dependencies.

Used as a build/release-time artifact for DoD ATO / RMF / supply-chain
attestation pursuits (NIST 800-53 SR-3, SR-4, SR-11; CISA SBOM minimum
elements; SSDF PS.3.2).

Output: ./sbom/heli-os-{version}.cdx.json

Usage:
    python scripts/generate_sbom.py [--version 0.2.0] [--output sbom/]

Optional dependency: cyclonedx-python-lib for richer pip dependency walk.
This script falls back to a minimal pure-Python implementation that does
not require cyclonedx-python-lib so it can run in any environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("sbom")

REPO_ROOT = Path(__file__).resolve().parent.parent

# Components we always want listed as "application" (first-party)
FIRST_PARTY_DIRS = [
    "apps/console",
    "apps/api_gateway",
    "apps/tasking",
    "apps/fabric",
    "apps/agent",
    "packages/c2_intel",
    "packages/world",
    "packages/adapters",
    "packages/agent",
    "packages/security",
    "packages/policy",
    "packages/identity",
    "packages/threat_assessment",
    "packages/deconfliction",
    "packages/swarm",
    "packages/training",
    "packages/ml",
    "packages/sdk",
    "packages/domains",
    "packages/entities",
    "packages/schemas",
    "packages/observability",
]


def _sha256_path(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_dir(d: Path, max_files: int = 200) -> str:
    """Hash up to max_files files for an aggregate directory hash."""
    h = hashlib.sha256()
    files = sorted(d.rglob("*.py"))[:max_files]
    for f in files:
        h.update(f.relative_to(REPO_ROOT).as_posix().encode())
        try:
            h.update(_sha256_path(f).encode())
        except Exception:
            continue
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _python_deps_pip_freeze() -> list[dict]:
    """Walk installed Python packages via pip — runs in current venv."""
    components: list[dict] = []
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            text=True, stderr=subprocess.DEVNULL,
        )
        for pkg in json.loads(out):
            name = pkg["name"].lower()
            version = pkg["version"]
            components.append({
                "type": "library",
                "bom-ref": f"pypi:{name}@{version}",
                "name": name,
                "version": version,
                "purl": f"pkg:pypi/{name}@{version}",
                "scope": "required",
            })
    except Exception as e:
        logger.warning("[sbom] pip list failed: %s", e)
    return components


def _node_deps_package_json(path: Path, scope_hint: str) -> list[dict]:
    """Parse a package.json for dependencies (no network walk — direct deps only)."""
    components: list[dict] = []
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    for kind in ("dependencies", "devDependencies"):
        for name, version in (data.get(kind) or {}).items():
            ver = re.sub(r"^[\^~>=<]+", "", str(version))
            components.append({
                "type": "library",
                "bom-ref": f"npm:{name}@{ver}",
                "name": name,
                "version": ver,
                "purl": f"pkg:npm/{name}@{ver}",
                "scope": "required" if kind == "dependencies" else "optional",
                "properties": [
                    {"name": "heli:source", "value": str(path.relative_to(REPO_ROOT))},
                    {"name": "heli:scope_hint", "value": scope_hint},
                ],
            })
    return components


def _ml_model_components() -> list[dict]:
    """Enumerate trained ML model artifacts as components — provenance important."""
    components = []
    models_dir = REPO_ROOT / "packages" / "c2_intel" / "models"
    if not models_dir.exists():
        return components
    for meta_file in sorted(models_dir.glob("*_meta.json")):
        try:
            meta = json.loads(meta_file.read_text())
        except Exception:
            continue
        model_name = meta_file.stem.replace("_meta", "")
        binary = next((p for p in (
            models_dir / f"{model_name}.pt",
            models_dir / f"{model_name}.joblib",
        ) if p.exists()), None)
        sha = ""
        if binary is not None:
            try:
                sha = _sha256_path(binary)
            except Exception:
                pass
        sources = meta.get("data_sources") or [meta.get("data_source") or "unknown"]
        components.append({
            "type": "machine-learning-model",
            "bom-ref": f"heli-model:{model_name}",
            "name": model_name,
            "version": meta.get("trained_at", "unknown"),
            "description": meta.get("task") or meta.get("model") or "",
            "hashes": [{"alg": "SHA-256", "content": sha}] if sha else [],
            "properties": [
                {"name": "heli:model_type", "value": meta.get("model", "")},
                {"name": "heli:n_samples", "value": str(meta.get("n_samples", ""))},
                {"name": "heli:data_sources", "value": ", ".join(map(str, sources))},
                {"name": "heli:metrics", "value": json.dumps(meta.get("metrics") or {})},
            ],
        })
    return components


def _first_party_components(version: str) -> list[dict]:
    components = []
    for rel in FIRST_PARTY_DIRS:
        d = REPO_ROOT / rel
        if not d.exists():
            continue
        sha = _sha256_dir(d)
        components.append({
            "type": "application",
            "bom-ref": f"heli:{rel}",
            "name": rel,
            "version": version,
            "hashes": [{"alg": "SHA-256", "content": sha}],
            "properties": [
                {"name": "heli:first_party", "value": "true"},
            ],
        })
    return components


def build_sbom(version: str = "0.2.0") -> dict:
    """Assemble the full CycloneDX-1.5 JSON document."""
    serial = f"urn:uuid:{uuid.uuid4()}"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    git_sha = _git_commit()

    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": serial,
        "version": 1,
        "metadata": {
            "timestamp": timestamp,
            "tools": [{"vendor": "Branca.ai Inc.", "name": "heli-sbom",
                       "version": "0.1.0"}],
            "component": {
                "type": "application",
                "bom-ref": f"heli-os@{version}",
                "name": "heli-os",
                "version": version,
                "supplier": {"name": "Branca.ai Inc.",
                             "url": ["https://branca.ai"]},
                "licenses": [{"license": {
                    "name": "Proprietary — Branca.ai Inc.",
                    "url": "https://github.com/Branca-ai/Heli.OS/blob/main/LICENSE",
                }}],
                "properties": [
                    {"name": "git:commit", "value": git_sha},
                    {"name": "heli:variant", "value": "all-in-one"},
                ],
            },
            "supplier": {"name": "Branca.ai Inc."},
        },
        "components": [],
    }

    # Order matters for human readers: first-party, then ML models, then libraries
    sbom["components"].extend(_first_party_components(version))
    sbom["components"].extend(_ml_model_components())
    sbom["components"].extend(_python_deps_pip_freeze())
    sbom["components"].extend(
        _node_deps_package_json(REPO_ROOT / "package.json", "build_root")
    )
    if (REPO_ROOT / "apps/console/package.json").exists():
        sbom["components"].extend(
            _node_deps_package_json(REPO_ROOT / "apps/console/package.json", "console")
        )

    logger.info("[sbom] %d components total", len(sbom["components"]))
    return sbom


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="0.2.0")
    parser.add_argument("--output", default="sbom/")
    parser.add_argument("--print", action="store_true",
                        help="Print SBOM to stdout instead of writing")
    args = parser.parse_args()

    sbom = build_sbom(version=args.version)

    if args.print:
        print(json.dumps(sbom, indent=2))
        return

    out_dir = REPO_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"heli-os-{args.version}.cdx.json"
    out_file.write_text(json.dumps(sbom, indent=2))
    logger.info("[sbom] wrote %s (%d components)", out_file, len(sbom["components"]))


if __name__ == "__main__":
    main()
