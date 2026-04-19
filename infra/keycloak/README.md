# Keycloak — Identity Provider for Summit.OS

Keycloak is the OIDC Identity Provider for Summit.OS. It handles user authentication (login/password), issues JWTs signed with RS256, enforces brute-force protection and password policy, and is what the Next.js console's OIDC flow talks to.

The `summit` realm is automatically imported on first startup from `realm-export.json`.

---

## Starting Keycloak

Keycloak is provided as an override compose file that layers on top of the main stack. Start the full stack with Keycloak included:

```bash
docker-compose \
  -f infra/docker/docker-compose.yml \
  -f infra/docker/docker-compose.keycloak.yml \
  up -d
```

Keycloak depends on the `keycloak-db-init` init container (defined in the base compose) which creates the `keycloak` database in Postgres before Keycloak starts.

---

## Admin Console

- URL: http://localhost:8080/admin
- Username: `admin`
- Password: `REDACTED` (override with `KC_BOOTSTRAP_ADMIN_PASSWORD` env var)

---

## OIDC Discovery URL

```
http://localhost:8080/realms/summit/.well-known/openid-configuration
```

---

## Environment Variables — apps/console/.env.local

Add the following to `apps/console/.env.local` to wire the Next.js console to Keycloak:

```
OIDC_ISSUER=http://localhost:8080/realms/summit
OIDC_CLIENT_ID=summit-console
OIDC_CLIENT_SECRET=REDACTED
OIDC_REDIRECT_URI=http://localhost:3000/api/auth/callback
```

---

## Default Dev User

| Field    | Value                   |
|----------|-------------------------|
| Email    | kyle@branca.ai   |
| Password | REDACTED         |
| Role     | OPERATOR                |

---

## Realm Roles

| Role              | Description                                        |
|-------------------|----------------------------------------------------|
| VIEWER            | Read-only access to situational awareness data     |
| OPERATOR          | Full operator access including dispatch and tasking|
| MISSION_COMMANDER | Mission planning, handoff briefs, command access   |
| ADMIN             | Administrative access to platform configuration   |
| SUPER_ADMIN       | Full unrestricted platform access                  |

---

## Security Notes

WARNING: The following secrets are defaults for local development only. Change all of them before any non-local deployment:

- `KC_BOOTSTRAP_ADMIN_PASSWORD` — Keycloak admin password
- `KEYCLOAK_DB_PASSWORD` — Keycloak Postgres user password
- `summit-console-secret` — The `summit-console` OIDC client secret (set in realm-export.json and must match OIDC_CLIENT_SECRET)

In production, set `KC_HTTP_ENABLED=false`, configure TLS, and set `KC_HOSTNAME` to the real hostname. Do not use `start-dev` in production — use `start` instead.
