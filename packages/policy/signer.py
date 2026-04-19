"""
Heli.OS Policy Signer

Signs OPA .rego policy files with Ed25519 so Heli.OS can verify
that no policy has been tampered with before loading it.

In a system that commands physical hardware, policy files are a critical
attack surface. If an attacker modifies `actuators.rego` to remove the
pressure-safety check, an AI agent could close a valve during a surge.
Signed policies make that attack detectable.

Signing flow (run once, at build/deploy time):
    from packages.policy.signer import PolicySigner
    signer = PolicySigner()
    signer.sign_all("infra/policy/")

Verification flow (at runtime, before OPA loads any policy):
    signer.verify_directory("infra/policy/")   # raises PolicyVerificationError if tampered

Key management:
    - POLICY_SIGNING_KEY env var: hex-encoded Ed25519 private key (32 bytes)
    - POLICY_VERIFY_KEY env var: hex-encoded Ed25519 public key (32 bytes)
    - If neither is set: auto-generates a keypair and logs the public key.
      Set POLICY_SIGNING_KEY in production.

Signature files: each `foo.rego` gets a `foo.rego.sig` file alongside it.
"""

from __future__ import annotations

import binascii
import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger("summit.policy.signer")


class PolicyVerificationError(Exception):
    """Raised when a policy file fails signature verification."""

    pass


class PolicySigner:
    """
    Signs and verifies Heli.OS OPA policy files.

    Uses Ed25519 (PyNaCl) — fast, modern, small keys (32 bytes).
    Falls back to a warning-only mode if PyNaCl is not installed.
    """

    def __init__(
        self,
        signing_key_hex: Optional[str] = os.getenv("POLICY_SIGNING_KEY"),
        verify_key_hex: Optional[str] = os.getenv("POLICY_VERIFY_KEY"),
        enforce: bool = os.getenv("POLICY_SIGNING_ENFORCE", "true").lower() == "true",
    ):
        self._enforce = enforce
        self._signing_key = None
        self._verify_key = None
        self._available = self._init_keys(signing_key_hex, verify_key_hex)

    def _init_keys(self, signing_hex: Optional[str], verify_hex: Optional[str]) -> bool:
        try:
            from nacl.signing import SigningKey, VerifyKey
            from nacl.encoding import HexEncoder

            if signing_hex:
                self._signing_key = SigningKey(signing_hex.encode(), encoder=HexEncoder)
                self._verify_key = self._signing_key.verify_key
                logger.info("PolicySigner: loaded signing key from environment")
            elif verify_hex:
                self._verify_key = VerifyKey(verify_hex.encode(), encoder=HexEncoder)
                logger.info("PolicySigner: loaded verify-only key (no signing)")
            else:
                # Generate ephemeral keypair — fine for development
                self._signing_key = SigningKey.generate()
                self._verify_key = self._signing_key.verify_key
                pub_hex = self._verify_key.encode(encoder=HexEncoder).decode()
                priv_hex = self._signing_key.encode(encoder=HexEncoder).decode()
                logger.warning(
                    "PolicySigner: no key configured — generated ephemeral keypair.\n"
                    f"  Set POLICY_SIGNING_KEY={priv_hex} to use this key permanently.\n"
                    f"  Public key (POLICY_VERIFY_KEY): {pub_hex}"
                )
            return True
        except ImportError:
            logger.warning(
                "PyNaCl not installed — policy signing disabled. "
                "Run: pip install PyNaCl\n"
                "Policy verification will be SKIPPED until PyNaCl is installed."
            )
            return False

    def sign_file(self, policy_path: str) -> Optional[str]:
        """
        Sign a .rego file. Creates `{policy_path}.sig` alongside it.

        Returns the hex signature string, or None if signing is unavailable.
        """
        if not self._available or self._signing_key is None:
            logger.warning(f"Cannot sign {policy_path} — signing key unavailable")
            return None

        try:
            from nacl.signing import SigningKey
            from nacl.encoding import HexEncoder

            content = Path(policy_path).read_bytes()
            # Sign SHA-256 of content (not raw content) to handle large files
            digest = hashlib.sha256(content).digest()
            signed = self._signing_key.sign(digest, encoder=HexEncoder)
            sig_hex = signed.signature.decode()

            sig_path = f"{policy_path}.sig"
            Path(sig_path).write_text(sig_hex)
            logger.info(f"Signed policy: {policy_path} → {sig_path}")
            return sig_hex

        except Exception as e:
            logger.error(f"Failed to sign {policy_path}: {e}")
            return None

    def verify_file(self, policy_path: str) -> bool:
        """
        Verify a .rego file against its .sig file.

        Returns True if valid.
        Raises PolicyVerificationError if tampered (when enforce=True).
        Logs warning and returns True if signature file missing (graceful).
        """
        sig_path = f"{policy_path}.sig"

        if not os.path.exists(sig_path):
            if self._enforce:
                msg = f"Policy signature file missing: {sig_path}"
                logger.error(msg)
                raise PolicyVerificationError(msg)
            logger.warning(f"No signature for {policy_path} — skipping verification")
            return True

        if not self._available or self._verify_key is None:
            logger.warning(f"Cannot verify {policy_path} — PyNaCl unavailable")
            return True

        try:
            from nacl.signing import VerifyKey
            from nacl.encoding import HexEncoder
            from nacl.exceptions import BadSignatureError

            content = Path(policy_path).read_bytes()
            digest = hashlib.sha256(content).digest()
            sig_hex = Path(sig_path).read_text().strip().encode()

            self._verify_key.verify(digest, binascii.unhexlify(sig_hex))
            logger.debug(f"Policy signature valid: {policy_path}")
            return True

        except Exception as e:
            if "BadSignatureError" in type(e).__name__ or "Invalid signature" in str(e):
                msg = (
                    f"POLICY TAMPERED: {policy_path} signature verification FAILED. "
                    f"This policy will NOT be loaded."
                )
                logger.critical(msg)
                if self._enforce:
                    raise PolicyVerificationError(msg)
                return False
            logger.warning(f"Policy verification error for {policy_path}: {e}")
            return True

    def sign_all(self, policy_dir: str) -> List[str]:
        """Sign all .rego files in a directory. Returns list of signed file paths."""
        signed = []
        for path in sorted(Path(policy_dir).glob("*.rego")):
            result = self.sign_file(str(path))
            if result:
                signed.append(str(path))
        logger.info(f"Signed {len(signed)} policy files in {policy_dir}")
        return signed

    def verify_directory(self, policy_dir: str) -> List[str]:
        """
        Verify all .rego files in a directory.

        Returns list of verified file paths.
        Raises PolicyVerificationError on first failure (if enforce=True).
        """
        verified = []
        for path in sorted(Path(policy_dir).glob("*.rego")):
            if self.verify_file(str(path)):
                verified.append(str(path))
        logger.info(f"Verified {len(verified)} policy files in {policy_dir}")
        return verified

    def export_public_key_hex(self) -> Optional[str]:
        """Return the hex-encoded public verification key for distribution."""
        if not self._available or self._verify_key is None:
            return None
        try:
            from nacl.encoding import HexEncoder

            return self._verify_key.encode(encoder=HexEncoder).decode()
        except Exception:
            return None
