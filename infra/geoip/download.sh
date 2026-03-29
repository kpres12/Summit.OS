#!/usr/bin/env bash
# Download MaxMind GeoLite2-Country database for Summit.OS GeoBlock middleware.
#
# Requirements:
#   1. Free MaxMind account — sign up at https://www.maxmind.com/en/geolite2/signup
#   2. Generate a license key: Account → Manage License Keys → Generate New License Key
#   3. Export: MAXMIND_LICENSE_KEY=<your_key>
#   4. Run:    bash infra/geoip/download.sh

set -euo pipefail

if [[ -z "${MAXMIND_LICENSE_KEY:-}" ]]; then
  echo "ERROR: MAXMIND_LICENSE_KEY is not set."
  echo ""
  echo "  1. Sign up free at https://www.maxmind.com/en/geolite2/signup"
  echo "  2. Generate a license key: Account → Manage License Keys"
  echo "  3. Run: MAXMIND_LICENSE_KEY=your_key bash infra/geoip/download.sh"
  exit 1
fi

DEST="$(dirname "$0")/GeoLite2-Country.mmdb"
TMP="$(mktemp -d)"

echo "Downloading GeoLite2-Country database..."

curl -fsSL \
  "https://download.maxmind.com/app/geoip_download?edition_id=GeoLite2-Country&license_key=${MAXMIND_LICENSE_KEY}&suffix=tar.gz" \
  -o "${TMP}/geolite2.tar.gz"

tar -xzf "${TMP}/geolite2.tar.gz" -C "${TMP}" --wildcards "*.mmdb"
find "${TMP}" -name "GeoLite2-Country.mmdb" -exec mv {} "${DEST}" \;

rm -rf "${TMP}"

echo "Done. Database written to: ${DEST}"
echo "Restart the api-gateway container to pick it up."
