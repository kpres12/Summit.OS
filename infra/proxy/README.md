# Summit.OS mTLS Configuration

This directory contains mTLS (mutual TLS) proxy configuration for securing Summit.OS services with client certificate authentication and organization-level multi-tenancy.

## Overview

The mTLS proxy layer:
- Validates client certificates against a trusted CA
- Extracts `org_id` from the certificate's OU (Organizational Unit) field
- Forwards the `org_id` to backend services via the `X-Org-ID` header
- Provides secure multi-tenant isolation at the network layer

## Architecture

```
Client (with cert) → Nginx mTLS Proxy → Backend Service
                           ↓
                  Extracts OU → org_id
                  Adds X-Org-ID header
```

## Quick Start

### 1. Generate Certificates

```bash
cd infra/proxy
./generate_certs.sh
```

This generates:
- `ca.crt` / `ca.key` - Certificate Authority
- `server.crt` / `server.key` - Server certificate
- `client-org1.crt` / `client-org1.key` - Client cert for org1
- `client-org2.crt` / `client-org2.key` - Client cert for org2
- `client-admin.crt` / `client-admin.key` - Client cert for admin

### 2. Start Services with mTLS

```bash
# Start all services including mTLS proxies
docker-compose -f infra/docker/docker-compose.yml --profile mtls up -d

# Or add to existing deployment
docker-compose -f infra/docker/docker-compose.yml --profile mtls up -d api-proxy fabric-proxy fusion-proxy intelligence-proxy tasking-proxy
```

### 3. Test mTLS Connection

```bash
cd infra/proxy/certs

# Test API Gateway with org1 credentials
curl --cacert ca.crt --cert client-org1.crt --key client-org1.key \
  https://localhost:8443/health

# Test with org2 credentials
curl --cacert ca.crt --cert client-org2.crt --key client-org2.key \
  https://localhost:8443/v1/worldstate

# Test Fabric service
curl --cacert ca.crt --cert client-org1.crt --key client-org1.key \
  https://localhost:8451/health
```

## Service Ports

When mTLS is enabled, services are available on these HTTPS ports:

- **API Gateway**: `https://localhost:8443` (proxies to 8000)
- **Fabric**: `https://localhost:8451` (proxies to 8001)
- **Fusion**: `https://localhost:8452` (proxies to 8002)
- **Intelligence**: `https://localhost:8453` (proxies to 8003)
- **Tasking**: `https://localhost:8454` (proxies to 8004)

## Multi-Tenancy with org_id

### Certificate Format

Client certificates MUST include the `OU` field in the Distinguished Name (DN):

```
/C=US/ST=CA/L=SF/O=Summit.OS/OU=org1/CN=client-org1
                               ^^^^^^
                               This becomes org_id
```

The nginx proxy extracts the OU value and forwards it as `X-Org-ID` header to backend services.

### Backend Integration

Backend services should:

1. Read the `X-Org-ID` header from incoming requests
2. Use it to filter/scope database queries and operations
3. Ensure users can only access data belonging to their organization

Example Python (FastAPI):

```python
from fastapi import Header, HTTPException

async def get_org_id(x_org_id: str = Header(None)) -> str:
    if not x_org_id:
        raise HTTPException(status_code=403, detail="Missing org_id")
    return x_org_id

@app.get("/missions")
async def list_missions(org_id: str = Depends(get_org_id)):
    # Only return missions for this org_id
    return db.query(Mission).filter_by(org_id=org_id).all()
```

## Generating Additional Client Certificates

To create a certificate for a new organization:

```bash
cd infra/proxy/certs

# Generate key and certificate signing request
openssl genrsa -out client-org3.key 2048
openssl req -new -key client-org3.key -out client-org3.csr \
  -subj "/C=US/ST=CA/L=SF/O=Summit.OS/OU=org3/CN=client-org3"

# Sign with CA
openssl x509 -req -in client-org3.csr -CA ca.crt -CAkey ca.key \
  -CAcreateserial -out client-org3.crt -days 365

# Clean up
rm client-org3.csr
```

## Security Considerations

1. **Protect Private Keys**: Never commit `*.key` files to version control
2. **CA Security**: The `ca.key` should be stored securely and offline in production
3. **Certificate Rotation**: Certificates expire after 365 days by default
4. **Revocation**: Implement CRL or OCSP for certificate revocation in production
5. **Health Endpoints**: Currently health checks bypass cert verification - consider implications

## Configuration Files

- `nginx.conf` - API Gateway proxy
- `nginx-fabric.conf` - Fabric service proxy
- `nginx-fusion.conf` - Fusion service proxy
- `nginx-intelligence.conf` - Intelligence service proxy
- `nginx-tasking.conf` - Tasking service proxy
- `generate_certs.sh` - Certificate generation script

## Troubleshooting

### Certificate Verification Failed

```
curl: (60) SSL certificate problem: unable to get local issuer certificate
```

Solution: Ensure you're using `--cacert ca.crt` and the CA cert is valid.

### 403 Forbidden

```
Client certificate must include OU field for org_id
```

Solution: Your client certificate is missing the OU field. Regenerate with OU specified.

### Connection Refused

Solution: Ensure mTLS proxies are running:

```bash
docker-compose -f infra/docker/docker-compose.yml ps | grep proxy
```

## Production Deployment

For production:

1. Use a trusted CA (e.g., Let's Encrypt, organization PKI)
2. Implement certificate revocation (CRL/OCSP)
3. Use hardware security modules (HSM) for CA keys
4. Enable certificate pinning in clients
5. Implement certificate rotation automation
6. Monitor certificate expiration
7. Enable request rate limiting
8. Add Web Application Firewall (WAF)
