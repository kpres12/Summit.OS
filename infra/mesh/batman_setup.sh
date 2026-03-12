#!/bin/bash
# Summit.OS — BATMAN-adv Mesh Network Setup
#
# Sets up a BATMAN-adv (Better Approach To Mobile Adhoc Networking) Layer 2
# mesh between Summit.OS nodes. Self-healing, topology-agnostic, no central
# router. Nodes can join/leave/fail and the mesh re-routes in milliseconds.
#
# Requirements:
#   - Linux kernel ≥ 3.10 with batman-adv module
#   - batctl (batman-adv userspace tools)
#   - Wireless interface in monitor/ad-hoc mode OR wired interface
#
# Usage:
#   sudo ./batman_setup.sh [OPTIONS]
#
# Options:
#   --interface IFACE   Physical interface to use (default: wlan0)
#   --ssid SSID         Ad-hoc SSID (default: summit-os-mesh)
#   --channel CH        WiFi channel (default: 6)
#   --bat-ip IP/CIDR    Batman interface IP (default: 10.10.0.X/24, X from last octet of MAC)
#   --mtu MTU           MTU for bat0 (default: 1532)
#   --no-dhcp           Skip DHCP server setup
#   --status            Show current mesh status and exit
#   --teardown          Tear down the mesh and exit
#
# Example — two nodes:
#   Node A: sudo ./batman_setup.sh --interface wlan0 --bat-ip 10.10.0.1/24
#   Node B: sudo ./batman_setup.sh --interface wlan0 --bat-ip 10.10.0.2/24
#
# After setup:
#   batctl n          — list direct neighbours
#   batctl o          — originator table (all reachable nodes)
#   batctl tl         — translation table (MAC → node mapping)
#   ip route          — verify bat0 route
#   ping 10.10.0.2    — ping another mesh node

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
INTERFACE="${BATMAN_IFACE:-wlan0}"
SSID="${BATMAN_SSID:-summit-os-mesh}"
CHANNEL="${BATMAN_CHANNEL:-6}"
BAT_IFACE="bat0"
MTU="${BATMAN_MTU:-1532}"
BAT_IP=""
SETUP_DHCP=true
ACTION="setup"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --interface) INTERFACE="$2"; shift 2 ;;
        --ssid)      SSID="$2"; shift 2 ;;
        --channel)   CHANNEL="$2"; shift 2 ;;
        --bat-ip)    BAT_IP="$2"; shift 2 ;;
        --mtu)       MTU="$2"; shift 2 ;;
        --no-dhcp)   SETUP_DHCP=false; shift ;;
        --status)    ACTION="status"; shift ;;
        --teardown)  ACTION="teardown"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[$(date -Iseconds)] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

require_root() {
    [[ $EUID -eq 0 ]] || die "Must run as root (sudo)"
}

require_cmd() {
    command -v "$1" &>/dev/null || die "Required command not found: $1 — install batctl / iw / iwconfig"
}

load_batman_module() {
    if ! lsmod | grep -q "^batman_adv"; then
        log "Loading batman_adv kernel module..."
        modprobe batman_adv || die "Failed to load batman_adv module. Ensure kernel headers are installed."
    fi
    log "batman_adv module loaded"
}

derive_bat_ip() {
    # Derive a /24 address from last two octets of MAC address
    local mac
    mac=$(cat "/sys/class/net/${INTERFACE}/address" 2>/dev/null || echo "02:00:00:00:00:01")
    local last_octet
    last_octet=$(printf "%d" "0x$(echo "$mac" | tr -d ':' | tail -c 3 | head -c 2)" 2>/dev/null || echo "1")
    last_octet=$((last_octet % 254 + 1))
    echo "10.10.0.${last_octet}/24"
}

# ── Status ────────────────────────────────────────────────────────────────────
show_status() {
    log "=== Summit.OS Mesh Status ==="
    if ip link show bat0 &>/dev/null; then
        echo "bat0 interface: UP"
        ip addr show bat0 | grep "inet " || echo "  (no IP assigned)"
        echo ""
        echo "--- Neighbours ---"
        batctl n 2>/dev/null || echo "  (batctl not available)"
        echo ""
        echo "--- Originator Table (all reachable nodes) ---"
        batctl o 2>/dev/null | head -20 || echo "  (batctl not available)"
        echo ""
        echo "--- Gateway ---"
        batctl gw_mode 2>/dev/null || true
    else
        echo "bat0 not active — mesh not running"
    fi
}

# ── Teardown ──────────────────────────────────────────────────────────────────
teardown() {
    log "Tearing down Summit.OS mesh..."

    if ip link show bat0 &>/dev/null 2>&1; then
        ip link set bat0 down || true
        batctl if del "$INTERFACE" 2>/dev/null || true
        ip link del bat0 2>/dev/null || true
    fi

    if ip link show "$INTERFACE" &>/dev/null 2>&1; then
        ip link set "$INTERFACE" down || true
        iwconfig "$INTERFACE" mode Managed 2>/dev/null || true
        ip link set "$INTERFACE" up || true
    fi

    log "Mesh teardown complete"
}

# ── Setup ─────────────────────────────────────────────────────────────────────
setup() {
    require_root
    require_cmd batctl
    require_cmd iw
    require_cmd iwconfig

    load_batman_module

    # Derive IP if not specified
    if [[ -z "$BAT_IP" ]]; then
        BAT_IP=$(derive_bat_ip)
        log "Auto-derived bat0 IP: $BAT_IP"
    fi

    log "Setting up mesh: interface=$INTERFACE, ssid=$SSID, channel=$CHANNEL, ip=$BAT_IP"

    # ── Bring physical interface down for reconfiguration
    ip link set "$INTERFACE" down

    # ── Set interface to ad-hoc mode
    iwconfig "$INTERFACE" mode ad-hoc 2>/dev/null || {
        # Try via iw if iwconfig fails
        iw dev "$INTERFACE" set type ibss 2>/dev/null || die "Cannot set $INTERFACE to ad-hoc/ibss mode"
    }

    ip link set "$INTERFACE" up

    # ── Join the ad-hoc cell
    iwconfig "$INTERFACE" essid "$SSID" channel "$CHANNEL" 2>/dev/null || \
        iw dev "$INTERFACE" ibss join "$SSID" $((2407 + CHANNEL * 5)) 2>/dev/null || \
        log "Warning: could not set SSID (may already be set)"

    # ── Add interface to BATMAN
    batctl if add "$INTERFACE" || log "Warning: $INTERFACE may already be in batman"

    # ── Configure bat0
    if ! ip link show bat0 &>/dev/null; then
        die "bat0 interface was not created — batman_adv setup failed"
    fi

    ip link set bat0 mtu "$MTU"
    ip link set bat0 up
    ip addr add "$BAT_IP" dev bat0 2>/dev/null || log "IP already assigned to bat0"

    # ── Tuning: routing algorithm and distributed ARP table
    batctl routing_algo BATMAN_IV 2>/dev/null || true
    batctl dat 1 2>/dev/null || true      # Enable distributed ARP table
    batctl bonding 0 2>/dev/null || true  # Disable bonding (single link)

    # ── Optional: lightweight DHCP for auto-assigning addresses to new nodes
    if [[ "$SETUP_DHCP" == true ]] && command -v dnsmasq &>/dev/null; then
        local subnet
        subnet=$(echo "$BAT_IP" | cut -d'.' -f1-3)
        log "Starting dnsmasq DHCP on bat0 for ${subnet}.100-${subnet}.200"
        dnsmasq \
            --interface=bat0 \
            --bind-interfaces \
            --dhcp-range="${subnet}.100,${subnet}.200,12h" \
            --no-daemon \
            --log-dhcp \
            --pid-file=/var/run/summit-mesh-dhcp.pid \
            &>/var/log/summit-mesh-dhcp.log &
        log "dnsmasq started (PID in /var/run/summit-mesh-dhcp.pid)"
    fi

    log "Summit.OS mesh setup complete"
    log ""
    show_status
    log ""
    log "MQTT bridge: set MQTT_HOST to a bat0 IP of your MQTT broker node"
    log "Next steps:"
    log "  batctl n               — check neighbours"
    log "  ping ${BAT_IP%%/*}     — verify self"
    log "  Set MESH_NODE_ID=$(hostname) in Summit.OS environment"
    log "  Set MESH_ENABLED=true"
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
case "$ACTION" in
    setup)    setup ;;
    status)   show_status ;;
    teardown) teardown ;;
esac
