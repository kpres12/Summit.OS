# Contracts (schemas) for Summit.OS

This package contains versioned JSON Schemas that define the structure of messages and REST payloads across the platform.

Guidelines
- All schemas are semver versioned via $id and $schema with embedded version.
- Backward-compatible changes: add optional fields only; never change meaning of existing fields.
- Breaking changes require a new major version and dual-read adapters in services.

Schemas
- jsonschemas/observation.schema.json: canonical Observation payload used on Redis Stream "observations_stream" and persisted by Fusion.
- Future: mission.schema.json, node.schema.json, advisory.schema.json, coverage.schema.json.
