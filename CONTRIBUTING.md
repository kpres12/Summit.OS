# Contributing to Summit.OS

Thank you for your interest in contributing. This guide covers setup, conventions, and the PR process.

## Getting Started

```bash
git clone <repo-url>
cd Summit.OS
make dev          # starts infrastructure + all services
make health       # verify everything is up
```

**Prerequisites:** Docker, Docker Compose, Node.js 20+, Python 3.11+.

## Repository Structure

- `apps/` — Microservices (Python/FastAPI) and console (Next.js)
- `packages/` — Shared Python libraries (entities, world, mesh, ai, etc.)
- `infra/` — Docker Compose, Prometheus, Grafana configs
- `tests/` — Integration and E2E tests
- `models/` — ONNX model files

## Development Workflow

### Branching

- `main` — stable, all CI passes
- `feature/<name>` — new features
- `fix/<name>` — bug fixes
- `docs/<name>` — documentation changes

### Making Changes

1. Create a branch from `main`
2. Make your changes
3. Run tests: `make test`
4. Run lint: `make lint`
5. Open a PR against `main`

### Running Tests

```bash
# All unit tests
make test

# Single service
cd apps/fabric && python -m pytest tests/ -v

# E2E tests (requires running stack)
make test-e2e

# Smoke test (requires running stack)
make smoke
```

### Code Style

**Python:**
- Formatter: `black` (line length 120)
- Linter: `flake8` (line length 120)
- Type hints encouraged
- `make format` to auto-format, `make lint` to check

**TypeScript/React:**
- ESLint + Prettier
- `cd apps/console && npm run lint`
- `cd apps/console && npm run format`

## Pull Request Process

1. Fill out the PR template
2. Ensure CI passes (lint, tests, typecheck)
3. Keep PRs focused — one logical change per PR
4. Add tests for new functionality
5. Update `CHANGELOG.md` under `[Unreleased]` if the change is user-facing

## Adding a New Service

1. Create `apps/<service>/main.py` with a FastAPI app and `/health` endpoint
2. Create `apps/<service>/Dockerfile`
3. Add to `infra/docker/docker-compose.yml`
4. Add to `Makefile` health/test/lint targets
5. Add to `.github/workflows/ci.yml`

## Adding a New SDK Adapter

Summit.OS uses a base adapter class. Any hardware or data source can be integrated in ~30 minutes.

```python
# packages/adapters/my_sensor.py
from packages.sdk import SummitAdapter, AdapterManifest, EntityBuilder

class MySensorAdapter(SummitAdapter):
    async def get_telemetry(self):
        # Return current position, status, etc.
        return {"lat": ..., "lon": ..., "alt": ...}

    async def handle_command(self, cmd):
        # Execute missions, waypoints, etc.
        pass
```

1. Subclass `SummitAdapter` in `packages/adapters/` or your own repo
2. Implement `get_telemetry()` and `handle_command()`
3. Register via `AdapterRegistry.add()` or `adapters.json` config
4. Test with `python -m pytest packages/adapters/tests/`

See `docs/INTEGRATION_GUIDE.md` and `examples/quickstart_adapter.py` for a complete working example.

## Database Migrations

Summit.OS uses [Alembic](https://alembic.sqlalchemy.org/) for schema migrations, wired to the Fabric service.

```bash
# Apply all pending migrations (fresh install or upgrade)
make db-migrate

# Check current revision
make db-status

# Roll back one migration
make db-rollback
```

Migration files live in `apps/fabric/alembic/versions/`. To create a new migration after changing the schema:

```bash
cd apps/fabric
POSTGRES_URL=postgresql://summit:summit_password@localhost:5433/summit_os \
  python -m alembic revision --autogenerate -m "describe your change"
```

**Rules:**
- Every schema change needs a migration file — never modify existing migration files
- Always include a `downgrade()` function
- Test both `upgrade` and `downgrade` before opening a PR

## Retraining the ML Models

The mission classifier and risk scorer ship as pre-trained ONNX files. To retrain on your own data:

```bash
# 1. Download fresh public training data (NASA FIRMS, NOAA, GBIF)
python packages/ml/download_real_data.py --years 2022 2023 2024

# 2. Train — optionally blend in your operator-approved mission history from Postgres
python packages/ml/train_mission_classifier.py \
  --real-csv packages/ml/data/real_combined.csv \
  --real-data postgresql://summit:password@localhost:5432/summit_os

# 3. New .onnx files drop into packages/ml/models/ automatically
# 4. Inference picks them up without a restart (hot-swap supported)
```

To contribute improved base models back upstream, open a PR with the new `.onnx` files and the training metrics from the run (accuracy, F1 per class, training data size).

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be respectful.

## License

By contributing, you agree that your contributions will be licensed under the GNU Affero General Public License v3 (AGPL v3).
