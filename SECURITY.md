# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| latest (`main`) | Yes |
| older releases | Best effort |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Report security issues to: **security@bigmt.ai**

Please include:
- A description of the vulnerability and its potential impact
- Steps to reproduce or a proof-of-concept (if available)
- The affected component(s) (e.g., API Gateway, Fabric, console)
- Any suggested mitigations

We will acknowledge receipt within **48 hours** and aim to provide a resolution timeline within **7 days**. We will credit reporters in the release notes unless you prefer to remain anonymous.

## Security Defaults

Summit.OS ships with security controls **disabled by default** to simplify local development. Before deploying to any network-accessible environment, you must:

```bash
# Enable authentication (OIDC + API key enforcement)
OIDC_ENFORCE=true
OIDC_ISSUER=https://your-keycloak/realms/summit
API_KEY_ENFORCE=true
RBAC_ENFORCE=true

# Set strong secrets (never use the example defaults)
POSTGRES_PASSWORD=<strong-random-password>
FABRIC_JWT_SECRET=<64-random-hex-chars>
FIELD_ENCRYPTION_KEY=<openssl rand -base64 32>

# Use MQTT over TLS in production (port 8883, not 1883)
MQTT_PORT=8883
```

See `.env.example` and the README Configuration section for full details.

## Threat Model

Summit.OS is designed for self-hosted deployment on trusted infrastructure (private cloud, on-premises, or air-gapped edge). It is not designed to be exposed directly to the public internet without a reverse proxy and authentication enforcement.

Key trust boundaries:
- **API Gateway** is the single ingress point — all external traffic must go through it
- **MQTT broker** should be on a private network or require client certificate auth
- **WebSocket endpoint** (`/ws/{org_id}`) requires a valid JWT when `OIDC_ENFORCE=true`
- **OPA policies** in `infra/policy/` define role-based access for all mission operations

## Known Limitations

- Multi-org tenancy is not yet implemented; each deployment is single-tenant
- The Helm chart for Kubernetes is not yet available; production Kubernetes deployments should apply network policies manually
