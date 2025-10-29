#!/bin/bash
# Generate CA and certificates for Summit.OS mTLS authentication
# Each client cert includes an OU (Organizational Unit) field that maps to org_id

set -e

CERT_DIR="$(cd "$(dirname "$0")/certs" && pwd)"
mkdir -p "$CERT_DIR"
cd "$CERT_DIR"

echo "Generating CA certificate..."
openssl genrsa -out ca.key 4096
openssl req -new -x509 -days 3650 -key ca.key -out ca.crt \
  -subj "/C=US/ST=CA/L=SF/O=Summit.OS/CN=Summit.OS Root CA"

echo "Generating server certificate..."
openssl genrsa -out server.key 2048
openssl req -new -key server.key -out server.csr \
  -subj "/C=US/ST=CA/L=SF/O=Summit.OS/CN=summit-api"

cat > server.ext << EOF
subjectAltName = DNS:summit-api,DNS:api-gateway,DNS:localhost,IP:127.0.0.1
extendedKeyUsage = serverAuth
EOF

openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
  -out server.crt -days 365 -extfile server.ext

echo "Generating client certificates for organizations..."

# Generate client cert for org1 (org_id=org1)
openssl genrsa -out client-org1.key 2048
openssl req -new -key client-org1.key -out client-org1.csr \
  -subj "/C=US/ST=CA/L=SF/O=Summit.OS/OU=org1/CN=client-org1"
openssl x509 -req -in client-org1.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
  -out client-org1.crt -days 365

# Generate client cert for org2 (org_id=org2)
openssl genrsa -out client-org2.key 2048
openssl req -new -key client-org2.key -out client-org2.csr \
  -subj "/C=US/ST=CA/L=SF/O=Summit.OS/OU=org2/CN=client-org2"
openssl x509 -req -in client-org2.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
  -out client-org2.crt -days 365

# Generate client cert for admin (org_id=admin)
openssl genrsa -out client-admin.key 2048
openssl req -new -key client-admin.key -out client-admin.csr \
  -subj "/C=US/ST=CA/L=SF/O=Summit.OS/OU=admin/CN=client-admin"
openssl x509 -req -in client-admin.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
  -out client-admin.crt -days 365

# Clean up CSRs and temp files
rm -f *.csr *.srl server.ext

echo "Certificates generated successfully!"
echo "CA cert: $CERT_DIR/ca.crt"
echo "Server cert: $CERT_DIR/server.crt"
echo "Server key: $CERT_DIR/server.key"
echo "Client certs: client-org1.crt, client-org2.crt, client-admin.crt"
echo ""
echo "To use a client cert with curl:"
echo "  curl --cacert ca.crt --cert client-org1.crt --key client-org1.key https://localhost:8443/health"
