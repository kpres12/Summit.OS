"""Unit tests for the Heli.OS Secret Client."""
import importlib.util
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Load packages/secrets/client.py directly to avoid collision with stdlib `secrets`.
_client_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "packages", "secret_store", "client.py")
)
_spec = importlib.util.spec_from_file_location("summit_secrets_client", _client_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
SecretClient = _mod.SecretClient


class TestSecretClientBackendSelection:

    def test_no_vault_addr_uses_env(self):
        client = SecretClient(vault_addr="", vault_token="", backend="auto")
        assert client._use_vault is False

    def test_vault_addr_with_token_uses_vault(self):
        client = SecretClient(
            vault_addr="https://vault.example.com",
            vault_token="s.abc123",
            backend="auto",
        )
        assert client._use_vault is True

    def test_explicit_env_backend_ignores_vault_addr(self):
        client = SecretClient(
            vault_addr="https://vault.example.com",
            vault_token="s.abc123",
            backend="env",
        )
        assert client._use_vault is False

    def test_explicit_vault_backend_forces_vault(self):
        client = SecretClient(vault_addr="https://vault.example.com", backend="vault")
        assert client._use_vault is True

    def test_approle_creds_trigger_vault(self):
        client = SecretClient(
            vault_addr="https://vault.example.com",
            role_id="my-role",
            secret_id="my-secret",
            backend="auto",
        )
        assert client._use_vault is True

    def test_vault_addr_without_creds_does_not_use_vault(self):
        client = SecretClient(
            vault_addr="https://vault.example.com",
            vault_token="",
            role_id="",
            secret_id="",
            backend="auto",
        )
        assert client._use_vault is False


class TestSecretClientEnvFallback:

    @pytest.mark.asyncio
    async def test_returns_env_var_when_no_vault(self, monkeypatch):
        monkeypatch.setenv("MY_TEST_SECRET", "super_secret_value")
        client = SecretClient(vault_addr="", backend="env")
        result = await client.get("MY_TEST_SECRET")
        assert result == "super_secret_value"

    @pytest.mark.asyncio
    async def test_returns_default_when_not_found(self):
        client = SecretClient(vault_addr="", backend="env")
        result = await client.get("NONEXISTENT_SECRET_XYZ", default="fallback")
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found_no_default(self):
        client = SecretClient(vault_addr="", backend="env")
        result = await client.get("TOTALLY_MISSING_SECRET_12345")
        assert result is None

    @pytest.mark.asyncio
    async def test_require_raises_on_missing(self):
        client = SecretClient(vault_addr="", backend="env")
        with pytest.raises(ValueError, match="NONEXISTENT_REQUIRED_SECRET"):
            await client.require("NONEXISTENT_REQUIRED_SECRET")

    @pytest.mark.asyncio
    async def test_require_returns_value_when_found(self, monkeypatch):
        monkeypatch.setenv("REQUIRED_KEY", "present")
        client = SecretClient(vault_addr="", backend="env")
        result = await client.require("REQUIRED_KEY")
        assert result == "present"


class TestSecretClientVaultMock:

    @pytest.mark.asyncio
    async def test_vault_hit_returns_value(self):
        """Mock a successful Vault KV v2 response."""
        client = SecretClient(
            vault_addr="https://vault.test",
            vault_token="s.testtoken",
            backend="vault",
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {"data": {"POSTGRES_PASSWORD": "mock-vault-value"}}
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=None)
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_http

            result = await client.get("POSTGRES_PASSWORD")

        assert result == "mock-vault-value"

    @pytest.mark.asyncio
    async def test_vault_404_falls_back_to_env(self, monkeypatch):
        """Vault 404 should fall through to env var."""
        monkeypatch.setenv("MISSING_IN_VAULT", "from_env")
        client = SecretClient(
            vault_addr="https://vault.test",
            vault_token="s.testtoken",
            backend="vault",
        )
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=None)
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_http

            result = await client.get("MISSING_IN_VAULT")

        assert result == "from_env"

    @pytest.mark.asyncio
    async def test_vault_network_error_falls_back_to_env(self, monkeypatch):
        """Network error reaching Vault falls back to env var."""
        monkeypatch.setenv("NETWORK_FALLBACK_KEY", "env_value")
        client = SecretClient(
            vault_addr="https://vault.test",
            vault_token="s.testtoken",
            backend="vault",
        )

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=None)
            mock_http.get = AsyncMock(side_effect=Exception("Connection refused"))
            mock_cls.return_value = mock_http

            result = await client.get("NETWORK_FALLBACK_KEY")

        assert result == "env_value"
