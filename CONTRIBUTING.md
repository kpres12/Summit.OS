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

See `INTEGRATION_GUIDE.md` and `examples/quickstart_adapter.py`.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be respectful.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
