# Secrets Management

## Overview

Summit.OS supports multiple secrets management patterns for production deployments. **Never commit secrets to version control.**

## Supported Patterns

### 1. Environment Variables (Development Only)

For local development, create a `.env` file (git-ignored):

```bash
cp .env.example .env
# Edit .env with your secrets
```

**⚠️ DO NOT use this in production.**

### 2. Docker Secrets (Recommended for Docker Swarm)

Create secrets:

```bash
echo "your_strong_password" | docker secret create postgres_password -
echo "your_jwt_secret_min_32_chars" | docker secret create fabric_jwt_secret -
echo "your_grafana_password" | docker secret create grafana_admin_password -
```

Update `docker-compose.yml` to use secrets:

```yaml
secrets:
  postgres_password:
    external: true
  fabric_jwt_secret:
    external: true

services:
  postgres:
    secrets:
      - postgres_password
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
```

### 3. Kubernetes Secrets (Recommended for K8s)

Create secrets:

```bash
kubectl create secret generic summit-secrets \
  --from-literal=postgres-password='your_password' \
  --from-literal=fabric-jwt-secret='your_jwt_secret' \
  --from-literal=grafana-admin-password='your_grafana_password'
```

Reference in deployment manifests:

```yaml
env:
  - name: POSTGRES_PASSWORD
    valueFrom:
      secretKeyRef:
        name: summit-secrets
        key: postgres-password
```

### 4. HashiCorp Vault (Enterprise)

Install Vault Agent injector and configure:

```yaml
apiVersion: v1
kind: Pod
metadata:
  annotations:
    vault.hashicorp.com/agent-inject: "true"
    vault.hashicorp.com/role: "summit-os"
    vault.hashicorp.com/agent-inject-secret-config: "secret/data/summit-os/config"
```

### 5. AWS Secrets Manager / Azure Key Vault / GCP Secret Manager

Use cloud-provider SDKs to fetch secrets at runtime:

```python
import boto3

client = boto3.client('secretsmanager', region_name='us-west-2')
response = client.get_secret_value(SecretId='summit-os/postgres-password')
password = response['SecretString']
```

## Rotation Policy

- **Postgres password**: Rotate every 90 days
- **JWT secrets**: Rotate every 180 days
- **Grafana admin password**: Rotate after initial setup
- **OIDC client secrets**: Follow IdP policy

## Least Privilege

- Each service should have its own DB user with minimal permissions
- Use separate secrets for dev/staging/prod
- Never reuse secrets across environments

## Audit & Monitoring

- Enable audit logs for secret access (Vault, AWS Secrets Manager, etc.)
- Alert on failed authentication attempts
- Track secret rotation dates

## Emergency Rotation

If a secret is compromised:

1. Immediately rotate the secret
2. Update all services
3. Review audit logs for unauthorized access
4. Revoke compromised tokens/sessions
5. Document incident in security log

## See Also

- [Docker Secrets](https://docs.docker.com/engine/swarm/secrets/)
- [Kubernetes Secrets](https://kubernetes.io/docs/concepts/configuration/secret/)
- [HashiCorp Vault](https://www.vaultproject.io/)
- [AWS Secrets Manager](https://aws.amazon.com/secrets-manager/)
