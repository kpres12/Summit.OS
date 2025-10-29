# External Integration Guide

This guide shows how to run Summit.OS apps against external infrastructure without deploying that infra yourself.

## 1) Create your .env

Copy `.env.example` to `.env` and fill in values for the providers you have:

- OIDC: NEXT_PUBLIC_OIDC_ISSUER, NEXT_PUBLIC_OIDC_CLIENT_ID, NEXT_PUBLIC_OIDC_REDIRECT_URI, OIDC_CLIENT_SECRET (optional)
- OPA: OPA_URL
- Messaging: MQTT_BROKER, MQTT_PORT, MQTT_USERNAME/PASSWORD (optional)
- Redis: REDIS_URL
- Database: POSTGRES_URL
- Upstream services (optional): FABRIC_URL, FUSION_URL, INTELLIGENCE_URL, TASKING_URL
- Console: NEXT_PUBLIC_API_URL, NEXT_PUBLIC_WS_URL

Notes:
- If you set FABRIC_URL/FUSION_URL/INTELLIGENCE_URL/TASKING_URL, API Gateway will call those external endpoints; otherwise it will use local containers.
- X-Org-ID is enforced end-to-end; your front proxy (nginx/Traefik) should set it, or call services directly with that header.

## 2) Start apps in external mode

```
make dev-external
```

This uses docker-compose.external.yml to:
- Remove local depends_on on internal Redis/Postgres/MQTT
- Inject your .env variables into app services

## 3) Stop

```
make stop
```

## Troubleshooting
- 401s: ensure OIDC_ENFORCE=false while testing, or set OIDC_ISSUER/AUDIENCE/JWKS properly.
- Policy denials: update infra/policy/*.rego on your OPA, or set OPA_URL to a reachable server.
- Missing data: set NEXT_PUBLIC_API_URL to your API Gateway URL and ensure the gateway points to the right FABRIC/FUSION/INTELLIGENCE/TASKING.
