#!/usr/bin/env bash
# Summit.OS Offline Tile Downloader
#
# Downloads map tiles for a geographic area so the operator console
# works with zero internet access. Tiles are served by the nginx
# tile-cache container.
#
# Two modes:
#   --warm-cache   Warms the nginx proxy cache by fetching tiles through it
#                  (requires tile-cache container to be running)
#
#   --pmtiles      Downloads a PMTiles archive for a region from Protomaps
#                  (self-contained single file, ~200MB for a US state)
#
# Usage:
#   # Warm the cache for an operational area (lat/lon bounding box)
#   ./download-tiles.sh --warm-cache \
#     --bbox "-122.5,37.5,-121.5,38.5" \
#     --zoom "0-16" \
#     --proxy "http://localhost:8089"
#
#   # Download a PMTiles region extract
#   ./download-tiles.sh --pmtiles --region us-west \
#     --output /data/tiles/us-west.pmtiles
#
# Requirements:
#   --warm-cache: curl
#   --pmtiles:    curl (downloads from Protomaps CDN)
#
# PMTiles regions available at: https://maps.protomaps.com/builds/
# Full planet: ~100GB | US: ~12GB | Individual states: ~200MB-2GB

set -euo pipefail

MODE="${1:-}"
BBOX="-180,-85,180,85"
ZOOM_MIN=0
ZOOM_MAX=14
PROXY="http://localhost:8089"
REGION="planet"
OUTPUT="/data/tiles/region.pmtiles"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --warm-cache) MODE="warm_cache" ;;
    --pmtiles)    MODE="pmtiles" ;;
    --bbox)       BBOX="$2"; shift ;;
    --zoom)
      IFS='-' read -r ZOOM_MIN ZOOM_MAX <<< "$2"
      shift ;;
    --proxy)      PROXY="$2"; shift ;;
    --region)     REGION="$2"; shift ;;
    --output)     OUTPUT="$2"; shift ;;
    *) ;;
  esac
  shift
done

# ── Mode: warm nginx cache ─────────────────────────────────────────────────────

warm_cache() {
  echo "Summit.OS tile cache warmer"
  echo "  Bounding box: $BBOX"
  echo "  Zoom levels:  $ZOOM_MIN–$ZOOM_MAX"
  echo "  Proxy:        $PROXY"
  echo ""

  IFS=',' read -r WEST SOUTH EAST NORTH <<< "$BBOX"

  total_tiles=0
  cached_tiles=0

  for zoom in $(seq "$ZOOM_MIN" "$ZOOM_MAX"); do
    # Convert lat/lon bbox to tile coordinates
    # Using Python for the math (more reliable than bash float ops)
    read -r X_MIN Y_MIN X_MAX Y_MAX <<< "$(python3 -c "
import math
def lon2tile(lon, z): return int((lon + 180) / 360 * 2**z)
def lat2tile(lat, z): return int((1 - math.log(math.tan(math.radians(lat)) + 1/math.cos(math.radians(lat))) / math.pi) / 2 * 2**z)
z = $zoom
west, south, east, north = $WEST, $SOUTH, $EAST, $NORTH
x_min = lon2tile(west, z)
x_max = lon2tile(east, z)
y_min = lat2tile(north, z)  # note: lat2tile is inverted
y_max = lat2tile(south, z)
print(x_min, y_min, x_max, y_max)
")"

    tiles_at_zoom=$(( (X_MAX - X_MIN + 1) * (Y_MAX - Y_MIN + 1) ))
    total_tiles=$((total_tiles + tiles_at_zoom))

    echo "Zoom $zoom: ${tiles_at_zoom} tiles (x=$X_MIN–$X_MAX, y=$Y_MIN–$Y_MAX)"

    for x in $(seq "$X_MIN" "$X_MAX"); do
      for y in $(seq "$Y_MIN" "$Y_MAX"); do
        url="${PROXY}/tiles/dark_all/${zoom}/${x}/${y}.png"
        status=$(curl -s -o /dev/null -w "%{http_code}" \
          -H "Referer: http://localhost:3000" \
          "$url" || echo "000")
        if [[ "$status" == "200" ]]; then
          cached_tiles=$((cached_tiles + 1))
        fi
      done
    done
  done

  echo ""
  echo "Done. Cached ${cached_tiles}/${total_tiles} tiles."
  echo "Tiles are now available offline via ${PROXY}"
}

# ── Mode: download PMTiles ─────────────────────────────────────────────────────

download_pmtiles() {
  echo "Summit.OS PMTiles downloader"
  echo "  Region: $REGION"
  echo "  Output: $OUTPUT"
  echo ""

  # Protomaps builds: https://maps.protomaps.com/builds/
  # Format: https://build.protomaps.com/{date}/{region}.pmtiles
  PMTILES_BASE="https://build.protomaps.com"

  # Find the latest build
  echo "Finding latest Protomaps build..."
  LATEST=$(curl -s "https://maps.protomaps.com/builds/" | grep -oP '\d{8}' | sort | tail -1)
  if [ -z "$LATEST" ]; then
    LATEST=$(date +%Y%m%d)
  fi

  URL="${PMTILES_BASE}/${LATEST}/${REGION}.pmtiles"
  echo "Downloading: $URL"
  echo "Output: $OUTPUT"
  echo ""

  mkdir -p "$(dirname "$OUTPUT")"
  curl -L --progress-bar -o "$OUTPUT" "$URL"

  size=$(du -sh "$OUTPUT" | cut -f1)
  echo ""
  echo "Downloaded $size PMTiles archive to $OUTPUT"
  echo ""
  echo "To use in Summit.OS:"
  echo "  1. Set NEXT_PUBLIC_TILE_URL=http://localhost:8089/pmtiles/${REGION}.pmtiles"
  echo "     in your .env or Coolify environment variables"
  echo "  2. Restart the console service"
  echo ""
  echo "Or use the PMTiles protocol directly in MapLibre:"
  echo "  pmtiles://http://localhost:8089/pmtiles/${REGION}.pmtiles"
}

# ── Run ────────────────────────────────────────────────────────────────────────

case "$MODE" in
  warm_cache) warm_cache ;;
  pmtiles)    download_pmtiles ;;
  *)
    echo "Summit.OS Offline Tile Downloader"
    echo ""
    echo "Usage:"
    echo "  $0 --warm-cache --bbox '-122.5,37.5,-121.5,38.5' --zoom 0-16"
    echo "  $0 --pmtiles --region us-west --output /data/tiles/us-west.pmtiles"
    echo ""
    echo "Before a field operation:"
    echo "  1. Start Summit.OS with internet"
    echo "  2. Run --warm-cache for your operational area"
    echo "  3. Go offline — tiles are cached"
    echo ""
    exit 1
    ;;
esac
