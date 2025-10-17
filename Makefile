# Summit.OS Development Environment
# Big Mountain Technologies - Distributed Intelligence Fabric

.PHONY: help dev dev-services dev-apps dev-console dev-backend clean test lint format install-deps

# Default target
help:
	@echo "Summit.OS Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Development:"
	@echo "  make dev           - Start full development environment"
	@echo "  make dev-services  - Start infrastructure services only"
	@echo "  make dev-apps      - Start Summit.OS applications"
	@echo "  make dev-console   - Start FireLine Console only"
	@echo "  make dev-backend   - Start backend services only"
	@echo "  make sim           - Launch SITL/HITL sim executor (local)"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean         - Clean up containers and volumes"
	@echo "  make test          - Run all tests"
	@echo "  make lint          - Run linting"
	@echo "  make format        - Format code"
	@echo "  make install-deps  - Install all dependencies"
	@echo ""
	@echo "Services:"
	@echo "  - FireLine Console: http://localhost:3000"
	@echo "  - API Gateway: http://localhost:8000"
	@echo "  - Data Fabric: http://localhost:8001"
	@echo "  - Sensor Fusion: http://localhost:8002"
	@echo "  - Intelligence: http://localhost:8003"
	@echo "  - Mission Tasking: http://localhost:8004"
	@echo "  - Grafana: http://localhost:3001"
	@echo "  - Prometheus: http://localhost:9090"

# Full development environment
dev: install-deps dev-services
	@echo "Starting Summit.OS development environment..."
	@echo "Services will be available at:"
	@echo "  - FireLine Console: http://localhost:3000"
	@echo "  - API Gateway: http://localhost:8000"
	@echo "  - Grafana: http://localhost:3001"
	@echo ""
	@echo "Starting applications in 10 seconds..."
	@sleep 10
	@$(MAKE) dev-apps

# Infrastructure services only
dev-services:
	@echo "Starting infrastructure services..."
	@docker-compose -f infra/docker/docker-compose.yml up -d redis postgres mqtt prometheus grafana
	@echo "Waiting for services to be ready..."
	@sleep 15
	@echo "Infrastructure services started!"

# Summit.OS applications
dev-apps:
	@echo "Starting Summit.OS applications..."
	@docker-compose -f infra/docker/docker-compose.yml up -d fabric fusion intelligence tasking api-gateway console
	@echo "Applications started! Check logs with: docker-compose -f infra/docker/docker-compose.yml logs -f"

# FireLine Console only
dev-console:
	@echo "Starting FireLine Console..."
	@cd apps/console && npm install
	@cd apps/console && npm run dev

# Backend services only
dev-backend:
	@echo "Starting backend services..."
	@docker-compose -f infra/docker/docker-compose.yml up -d fabric fusion intelligence tasking api-gateway

# Clean up
clean:
	@echo "Cleaning up Summit.OS environment..."
	@docker-compose -f infra/docker/docker-compose.yml down -v
	@docker system prune -f
	@echo "Cleanup complete!"

# Install dependencies
install-deps:
	@echo "Installing dependencies..."
	@cd apps/console && npm install
	@echo "Dependencies installed!"

# Local simulation helper (SITL/HITL)
sim:
	@echo "Starting local simulation executor..."
	@python apps/tasking/sim_executor.py --asset drone-001=udp:127.0.0.1:14550 --register-assets --arm --takeoff-alt 20 --loiter-center 37.422,-122.084 --loiter-radius 150 --speed 5 --start

# Run tests
test:
	@echo "Running tests..."
	@cd apps/fabric && python -m pytest tests/ -v
	@cd apps/fusion && python -m pytest tests/ -v
	@cd apps/intelligence && python -m pytest tests/ -v
	@cd apps/tasking && python -m pytest tests/ -v
	@cd apps/console && npm test
	@echo "Tests completed!"

# Linting
lint:
	@echo "Running linting..."
	@cd apps/fabric && python -m flake8 .
	@cd apps/fusion && python -m flake8 .
	@cd apps/intelligence && python -m flake8 .
	@cd apps/tasking && python -m flake8 .
	@cd apps/console && npm run lint
	@echo "Linting completed!"

# Format code
format:
	@echo "Formatting code..."
	@cd apps/fabric && python -m black .
	@cd apps/fusion && python -m black .
	@cd apps/intelligence && python -m black .
	@cd apps/tasking && python -m black .
	@cd apps/console && npm run format
	@echo "Code formatted!"

# Development database setup
db-setup:
	@echo "Setting up development database..."
	@docker-compose -f infra/docker/docker-compose.yml exec postgres psql -U summit -d summit_os -c "CREATE EXTENSION IF NOT EXISTS postgis;"
	@echo "Database setup complete!"

# Generate mock data
mock-data:
	@echo "Generating mock data..."
	@python scripts/generate_mock_data.py
	@echo "Mock data generated!"

# Demo mission
demo:
	@echo "Starting demo mission..."
	@python scripts/demo_mission.py
	@echo "Demo mission started!"

# Comprehensive demo (full P0/P1 feature showcase)
demo-full:
	@echo "Starting comprehensive Summit.OS demonstration..."
	@python scripts/demo_full.py
	@echo "Demo complete!"

# Publish sample smoke observation to MQTT
smoke-detection:
	@echo "Publishing sample smoke observation to MQTT..."
	@python scripts/publish_smoke_detection.py
	@echo "Published sample smoke observation."

# Health check
health:
	@echo "Checking Summit.OS health..."
	@curl -s http://localhost:8000/health | jq .
	@curl -s http://localhost:8001/health | jq .
	@curl -s http://localhost:8002/health | jq .
	@curl -s http://localhost:8003/health | jq .
	@curl -s http://localhost:8004/health | jq .
	@echo "Health check complete!"

# Logs
logs:
	@docker-compose -f infra/docker/docker-compose.yml logs -f

# Stop all services
stop:
	@echo "Stopping Summit.OS services..."
	@docker-compose -f infra/docker/docker-compose.yml down
	@echo "Services stopped!"

# Toggle direct autopilot fallback (TASKING service)
tasking-direct-on:
	@echo "Enabling TASKING_DIRECT_AUTOPILOT and restarting tasking service..."
	@TASKING_DIRECT_AUTOPILOT=true docker-compose -f infra/docker/docker-compose.yml up -d --no-deps --build tasking
	@echo "Direct autopilot enabled."

tasking-direct-off:
	@echo "Disabling TASKING_DIRECT_AUTOPILOT and restarting tasking service..."
	@TASKING_DIRECT_AUTOPILOT=false docker-compose -f infra/docker/docker-compose.yml up -d --no-deps --build tasking
	@echo "Direct autopilot disabled."
