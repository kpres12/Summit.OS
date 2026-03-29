# Summit.OS Community Edition

Summit.OS is an open platform for autonomous systems coordination — free to use,
self-host, fork, and build on. Think Linux for autonomous operations: the core OS
is open, the ecosystem is yours to shape.

---

## The Model

Summit.OS follows the open-core model:

| | Community | Enterprise |
|---|---|---|
| **License** | Apache 2.0 | Apache 2.0 + Commercial |
| **Cost** | Free forever | $1,500 / month |
| **Deployment** | Self-hosted | Self-hosted or BigMT.ai managed |
| **Organizations** | Single (`default`) | Multi-org, fully isolated |
| **Entity tracking** | Unlimited (self-hosted) | Unlimited + hosted limits by tier |
| **Missions & tasking** | Full | Full |
| **ML inference** | Full | Full |
| **Sensor fusion** | Full | Full |
| **WebSocket streaming** | Full | Full + per-org isolation |
| **Adapters** | All built-in + custom | All built-in + custom |
| **OIDC / SSO** | Optional | Enforced |
| **RBAC** | Optional | Enforced |
| **Audit logging** | Optional | Enforced, append-only |
| **Field encryption** | Optional | Enforced |
| **Org management API** | — | SUPER_ADMIN CRUD |
| **Support** | GitHub Issues | Priority + SLA |

---

## What You Can Build

Summit.OS is designed to be a foundation, not a ceiling. Community edition is
Apache 2.0 — you can build products, services, and integrations on top of it
without restriction, as long as you comply with the license.

Examples of things people build on Summit.OS:

- **Wildfire response platforms** — custom alert ingestion, dispatch automation,
  crew coordination overlays
- **Commercial UAV fleet management** — multi-drone tasking, geofencing, telemetry
  dashboards for specific industries
- **Search and rescue tools** — terrain analysis overlays, resource tracking,
  handoff briefing generators
- **Municipal emergency management** — multi-agency coordination, resource
  deconfliction, situational awareness
- **Custom sensor adapters** — hardware integrations for specific sensors, radios,
  or data sources

If you build something on Summit.OS, we'd love to know about it. Open a GitHub
Discussion or reach out at hello@bigmt.ai.

---

## Community vs. Enterprise: The Key Distinction

**Community** is for individuals, researchers, small teams, and anyone self-hosting
for non-commercial or low-stakes use. You get full access to every capability —
the only difference is that multi-org tenancy and the enterprise enforcement
features are not available.

**Enterprise** is for organizations that need:
- Multiple isolated tenants on a single deployment
- Guaranteed auth, RBAC, and audit enforcement (not optional)
- SLA-backed support
- Managed hosting by BigMT.ai

Enterprise features are gated by `ENTERPRISE_MULTI_TENANT=true` in your
deployment configuration. They are fully implemented in the open source codebase
— the distinction is operational, not technical.

---

## Getting Started (Community)

```bash
git clone https://github.com/BigMT-Ai/Summit.OS
cd Summit.OS
cp .env.example .env
# Edit .env with your settings
make dev
```

The stack comes up at:
- Console: http://localhost:3002
- API Gateway: http://localhost:8000
- Grafana: http://localhost:3001

See [README.md](./README.md) for full setup instructions.

---

## Enterprise Inquiries

Email **enterprise@bigmt.ai** or visit **bigmt.ai** to discuss enterprise
licensing, managed hosting, or custom deployment support.

---

## Contributing

Summit.OS grows through community contributions. See [CONTRIBUTING.md](./CONTRIBUTING.md)
for how to get involved — whether that's a new adapter, a bug fix, a new ML model,
or documentation.
