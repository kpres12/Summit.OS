"""
Summit.OS Standalone Demo

No Docker. No services. Just Python + Ollama.

1. Fetches live aircraft from OpenSky Network (free, no account)
2. Builds a world context from real data
3. Sends to local Ollama with a mission objective
4. Prints what the AI brain decides to do

Usage:
    python scripts/demo.py
    python scripts/demo.py --mission "Find any aircraft that look unusual"
    python scripts/demo.py --bbox "32,-118,36,-114"   # LA area
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone

try:
    import httpx
except ImportError:
    print("ERROR: pip install httpx")
    sys.exit(1)

OLLAMA_URL = "http://localhost:11434"
OPENSKY_URL = "https://opensky-network.org/api/states/all"

# ── Fetch live aircraft ───────────────────────────────────────────────────────

def fetch_aircraft(bbox: str = "") -> list:
    params = {}
    if bbox:
        parts = [float(x) for x in bbox.split(",")]
        params = {"lamin": parts[0], "lomin": parts[1], "lamax": parts[2], "lomax": parts[3]}

    print(f"  Fetching live aircraft from OpenSky...", end="", flush=True)
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.get(OPENSKY_URL, params=params)
            r.raise_for_status()
            data = r.json()
            states = data.get("states") or []
            print(f" {len(states)} aircraft found")
            return states
    except Exception as e:
        print(f" FAILED: {e}")
        return []


def state_to_entity(sv: list) -> dict | None:
    try:
        icao = sv[0]
        callsign = (sv[1] or "").strip() or icao
        lon, lat = sv[5], sv[6]
        baro_alt = sv[7]
        on_ground = sv[8]
        velocity = sv[9]
        heading = sv[10]
        vert_rate = sv[11]

        if lat is None or lon is None:
            return None

        state = "ACTIVE"
        if baro_alt and baro_alt > 12000:
            state = "ACTIVE"
        if velocity and velocity > 300:
            state = "WARNING"   # Fast — flag it

        return {
            "entity_id": f"adsb-{icao}",
            "entity_type": "TRACK",
            "domain": "AERIAL",
            "state": state,
            "name": callsign,
            "kinematics": {
                "position": {"latitude": lat, "longitude": lon, "altitude": baro_alt},
                "velocity": {"speed": velocity, "heading": heading, "vertical_rate": vert_rate},
            },
            "aerial": {
                "flight_mode": "GROUND" if on_ground else "AIRBORNE",
                "icao24": icao,
            },
            "provenance": {"source_type": "adsb"},
        }
    except Exception:
        return None


# ── Build LLM context ─────────────────────────────────────────────────────────

def build_context(entities: list, mission: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Prioritise: WARNING first, then ACTIVE, cap at 30 entities
    entities.sort(key=lambda e: (0 if e.get("state") == "WARNING" else 1))
    entities = entities[:30]

    lines = [
        f"SUMMIT.OS OPERATOR AI | {ts}",
        f"CURRENT MISSION: {mission}",
        "",
        f"WORLD STATE ({len(entities)} aerial tracks):",
    ]

    for e in entities:
        pos = e.get("kinematics", {}).get("position", {})
        vel = e.get("kinematics", {}).get("velocity", {})
        aerial = e.get("aerial", {})
        lat = pos.get("latitude", 0)
        lon = pos.get("longitude", 0)
        alt = pos.get("altitude") or 0
        spd = vel.get("speed") or 0
        hdg = vel.get("heading") or 0
        vr  = vel.get("vertical_rate") or 0
        mode = aerial.get("flight_mode", "?")
        state = e.get("state", "ACTIVE")
        name = e.get("name", e.get("entity_id", "?"))

        lines.append(
            f"  [{state}] {name} | {mode} | "
            f"@{lat:.3f},{lon:.3f} alt={alt:.0f}m | "
            f"spd={spd:.0f}m/s hdg={hdg:.0f}° vr={vr:+.1f}m/s"
        )

    return "\n".join(lines)


# ── Ask Ollama ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are Summit.OS, an autonomous systems coordination AI. You monitor live sensor data \
and advise operators on what actions to take.

Given a mission objective and live world state, analyse the situation briefly. \
Then list any specific actions you would take (deploy an asset, raise an alert, etc.) \
or state that no action is needed and why.

Be concise. Lead with the most important finding. Max 200 words."""

def ask_brain(context: str, model: str) -> str:
    user_msg = f"{context}\n\nAnalyse the situation and recommend actions."

    print(f"  Sending to Ollama ({model})...", end="", flush=True)
    try:
        with httpx.Client(timeout=120.0) as client:
            r = client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model,
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 400},
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                },
            )
            r.raise_for_status()
            content = r.json().get("message", {}).get("content", "")
            print(" done")
            return content
    except Exception as e:
        print(f" FAILED: {e}")
        return ""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Summit.OS standalone demo")
    parser.add_argument("--mission", default="Monitor all aircraft and identify anything unusual or potentially concerning")
    parser.add_argument("--bbox", default="", help="lat_min,lon_min,lat_max,lon_max — leave blank for global")
    parser.add_argument("--model", default="gemma3:4b", help="Ollama model to use")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  SUMMIT.OS — Live Demo")
    print("="*60)
    print(f"  Mission : {args.mission}")
    print(f"  Model   : {args.model}")
    print(f"  Area    : {args.bbox or 'global (may be slow/large)'}")
    print("="*60 + "\n")

    # 1. Fetch live data
    states = fetch_aircraft(args.bbox)
    if not states:
        print("No aircraft data — check your connection or try a smaller bbox")
        sys.exit(1)

    # 2. Build entities
    entities = [e for sv in states if (e := state_to_entity(sv)) is not None]
    warnings = [e for e in entities if e["state"] == "WARNING"]
    print(f"  Entities built : {len(entities)} ({len(warnings)} flagged WARNING)")

    # 3. Build context
    context = build_context(entities, args.mission)

    # 4. Ask brain
    response = ask_brain(context, args.model)
    if not response:
        sys.exit(1)

    # 5. Print result
    print("\n" + "="*60)
    print("  AI BRAIN RESPONSE")
    print("="*60)
    print(response)
    print("="*60)

    if warnings:
        print(f"\n  [{len(warnings)} WARNING-state aircraft in dataset]")
        for w in warnings[:5]:
            pos = w["kinematics"]["position"]
            vel = w["kinematics"]["velocity"]
            print(f"    {w['name']} | alt={pos.get('altitude',0):.0f}m spd={vel.get('speed',0):.0f}m/s")

    print()


if __name__ == "__main__":
    main()
